//
//  GGUFTokenizer.swift
//  Code
//
//  BPE tokenizer extracted from GGUF metadata (tokenizer.ggml.*)
//  Supports Qwen chat template (<|im_start|>, <|im_end|>)
//

import Foundation

class GGUFTokenizer {

    // Vocabulary
    private var tokens: [String] = []           // id → token string
    private var scores: [Float] = []            // id → merge score
    private var tokenToId: [String: Int] = [:]  // token → id
    private var merges: [(String, String)] = []  // BPE merge pairs

    // Special tokens
    private(set) var bosTokenId: Int = -1
    private(set) var eosTokenId: Int = -1
    private(set) var padTokenId: Int = -1
    private(set) var unknownTokenId: Int = 0

    // Qwen special tokens
    private var imStartId: Int = -1
    private var imEndId: Int = -1
    private var nlTokenId: Int = -1

    private(set) var vocabSize: Int = 0

    // MARK: - Loading from GGUF Metadata

    func loadFromGGUF(metadata: [String: Any]) {
        // Extract token list
        if let tokenList = metadata["tokenizer.ggml.tokens"] as? [Any] {
            tokens = tokenList.compactMap { $0 as? String }
            vocabSize = tokens.count
            for (i, tok) in tokens.enumerated() {
                tokenToId[tok] = i
            }
        }

        // Extract scores (for merge priority)
        if let scoreList = metadata["tokenizer.ggml.scores"] as? [Any] {
            scores = scoreList.compactMap {
                if let f = $0 as? Float { return f }
                if let d = $0 as? Double { return Float(d) }
                return 0
            }
        }

        // Extract merge pairs
        if let mergeList = metadata["tokenizer.ggml.merges"] as? [Any] {
            merges = mergeList.compactMap { item -> (String, String)? in
                guard let str = item as? String else { return nil }
                let parts = str.split(separator: " ", maxSplits: 1)
                guard parts.count == 2 else { return nil }
                return (String(parts[0]), String(parts[1]))
            }
        }

        // Special token IDs
        if let v = metadata["tokenizer.ggml.bos_token_id"] {
            bosTokenId = intValue(v) ?? -1
        }
        if let v = metadata["tokenizer.ggml.eos_token_id"] {
            eosTokenId = intValue(v) ?? -1
        }
        if let v = metadata["tokenizer.ggml.padding_token_id"] {
            padTokenId = intValue(v) ?? -1
        }
        if let v = metadata["tokenizer.ggml.unknown_token_id"] {
            unknownTokenId = intValue(v) ?? 0
        }

        // Find Qwen special tokens
        imStartId = tokenToId["<|im_start|>"] ?? -1
        imEndId = tokenToId["<|im_end|>"] ?? -1
        nlTokenId = tokenToId["\n"] ?? -1
    }

    // MARK: - BPE Encoding

    func encode(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }

        // Convert text to initial byte-level tokens
        var symbols = textToInitialTokens(text)

        // Apply BPE merges
        applyBPEMerges(&symbols)

        return symbols
    }

    /// Encode with Qwen chat template
    func encodeChat(messages: [LLMMessage], addGeneration: Bool = true) -> [Int] {
        var tokens: [Int] = []

        for message in messages {
            // <|im_start|>role\ncontent<|im_end|>\n
            if imStartId >= 0 { tokens.append(imStartId) }

            let roleStr = message.role.rawValue
            tokens.append(contentsOf: encode(roleStr))
            if nlTokenId >= 0 { tokens.append(nlTokenId) }
            else { tokens.append(contentsOf: encode("\n")) }

            tokens.append(contentsOf: encode(message.content))

            if imEndId >= 0 { tokens.append(imEndId) }
            if nlTokenId >= 0 { tokens.append(nlTokenId) }
            else { tokens.append(contentsOf: encode("\n")) }
        }

        // Add generation prompt
        if addGeneration {
            if imStartId >= 0 { tokens.append(imStartId) }
            tokens.append(contentsOf: encode("assistant"))
            if nlTokenId >= 0 { tokens.append(nlTokenId) }
            else { tokens.append(contentsOf: encode("\n")) }
        }

        return tokens
    }

    // MARK: - Decoding

    func decode(_ ids: [Int], skipSpecial: Bool = true) -> String {
        var result = ""
        for id in ids {
            guard id >= 0, id < tokens.count else { continue }
            let tok = tokens[id]
            if skipSpecial && isSpecialToken(id) { continue }
            result += unescapeToken(tok)
        }
        return result
    }

    func decodeOne(_ id: Int) -> String {
        guard id >= 0, id < tokens.count else { return "" }
        let tok = tokens[id]
        if isSpecialToken(id) { return "" }
        return unescapeToken(tok)
    }

    func isEOS(_ id: Int) -> Bool {
        return id == eosTokenId || id == imEndId
    }

    func isSpecialToken(_ id: Int) -> Bool {
        return id == bosTokenId || id == eosTokenId || id == padTokenId
            || id == imStartId || id == imEndId
    }

    // MARK: - Private BPE Implementation

    private func textToInitialTokens(_ text: String) -> [Int] {
        let bytes = Array(text.utf8)
        var result: [Int] = []

        var i = 0
        while i < bytes.count {
            var bestLen = 0
            var bestId = unknownTokenId

            // Try longest match first (up to 32 bytes)
            let maxLen = min(32, bytes.count - i)
            for len in stride(from: maxLen, through: 1, by: -1) {
                let slice = Array(bytes[i..<(i + len)])
                guard let sub = String(bytes: slice, encoding: .utf8) else { continue }
                if let id = tokenToId[sub] {
                    bestLen = len
                    bestId = id
                    break
                }
            }

            if bestLen > 0 {
                result.append(bestId)
                i += bestLen
            } else {
                // Single byte fallback using hex escape
                let byteStr = String(format: "<0x%02X>", bytes[i])
                if let id = tokenToId[byteStr] {
                    result.append(id)
                } else {
                    result.append(unknownTokenId)
                }
                i += 1
            }
        }

        return result
    }

    private func applyBPEMerges(_ symbols: inout [Int]) {
        guard !merges.isEmpty else { return }

        // Build merge priority lookup
        var mergePriority: [String: Int] = [:]
        for (i, merge) in merges.enumerated() {
            let key = merge.0 + " " + merge.1
            mergePriority[key] = i
        }

        // Iteratively apply best merge
        while symbols.count > 1 {
            var bestIdx = -1
            var bestPriority = Int.max
            var bestMergedId = -1

            for i in 0..<(symbols.count - 1) {
                let left = symbols[i] < tokens.count ? tokens[symbols[i]] : ""
                let right = symbols[i + 1] < tokens.count ? tokens[symbols[i + 1]] : ""
                let key = left + " " + right
                if let priority = mergePriority[key], priority < bestPriority {
                    let merged = left + right
                    if let mergedId = tokenToId[merged] {
                        bestIdx = i
                        bestPriority = priority
                        bestMergedId = mergedId
                    }
                }
            }

            if bestIdx < 0 { break }

            symbols[bestIdx] = bestMergedId
            symbols.remove(at: bestIdx + 1)
        }
    }

    private func unescapeToken(_ token: String) -> String {
        var s = token
        // Handle Qwen/GPT-style byte escape sequences
        if s.hasPrefix("<0x") && s.hasSuffix(">") {
            let hex = String(s.dropFirst(3).dropLast(1))
            if let byte = UInt8(hex, radix: 16) {
                return String(bytes: [byte], encoding: .utf8) ?? ""
            }
        }
        // Handle SentencePiece-style space marker
        s = s.replacingOccurrences(of: "▁", with: " ")
        s = s.replacingOccurrences(of: "Ġ", with: " ")
        s = s.replacingOccurrences(of: "Ċ", with: "\n")
        return s
    }

    private func intValue(_ v: Any) -> Int? {
        if let i = v as? UInt32 { return Int(i) }
        if let i = v as? Int32 { return Int(i) }
        if let i = v as? UInt64 { return Int(i) }
        if let i = v as? Int64 { return Int(i) }
        if let i = v as? Int { return i }
        return nil
    }
}

// MARK: - Qwen Chat Template Extension for LLMTokenizer

extension LLMTokenizer {
    /// Format for Qwen-style chat models using ChatML template
    func formatQwenPrompt(messages: [LLMMessage]) -> String {
        var prompt = ""

        for message in messages {
            prompt += "<|im_start|>\(message.role.rawValue)\n"
            prompt += "\(message.content)<|im_end|>\n"
        }

        // Add assistant generation prompt
        prompt += "<|im_start|>assistant\n"

        return prompt
    }
}
