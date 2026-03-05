//
//  ANEInferenceEngine.swift
//  Code
//
//  Qwen3.5 hybrid Mamba-2 + Attention inference engine
//  18 Mamba layers + 6 pure attention layers, all CPU with Accelerate
//

import Foundation
import Accelerate

class ANEInferenceEngine {

    let compiler: ANEModelCompiler
    let config: GGUFModelConfig

    // Generation parameters
    var maxTokens: Int = 512
    var temperature: Float = 0.7
    var topK: Int = 40
    var topP: Float = 0.9

    // State
    private var position: Int = 0

    // Mamba state: conv buffer + SSM recurrent state
    // convState[mambaIdx] = flattened [(K-1) * channels]
    private var convStates: [[Float]] = []
    // ssmState[mambaIdx] = flattened [nGroups * dState * dInnerPerGroup]
    private var ssmStates: [[Float]] = []

    // Attention state: KV cache
    // kvCache[attnIdx] = flat [maxSeq * 2 * kvDim] with layout [pos][K_or_V][kvDim]
    private var kvCaches: [[Float]] = []

    // Derived constants
    private let dim: Int
    private let hiddenDim: Int
    private let nHeads: Int
    private let nKVHeads: Int
    private let headDim: Int
    private let kvDim: Int
    private let ropeDim: Int
    private let ropeTheta: Float
    private let rmsEps: Float
    private let nGroups: Int
    private let dState: Int
    private let ssmInner: Int
    private let dInnerPerGroup: Int
    private let convKernel: Int
    private let totalConvChannels: Int
    private let vocabSize: Int
    private let maxSeqLen: Int

    init(compiler: ANEModelCompiler) {
        self.compiler = compiler
        self.config = compiler.config

        dim = config.dim
        hiddenDim = config.hiddenDim
        nHeads = config.nHeads
        nKVHeads = config.nKVHeads
        headDim = config.headDim
        kvDim = config.kvDim
        ropeDim = config.ropeDim
        ropeTheta = config.ropeTheta
        rmsEps = config.rmsNormEps
        nGroups = config.ssmGroupCount
        dState = config.ssmStateSize
        ssmInner = config.ssmInnerSize
        dInnerPerGroup = ssmInner / max(nGroups, 1)
        convKernel = config.ssmConvKernel
        totalConvChannels = ssmInner * 3
        vocabSize = config.vocabSize
        maxSeqLen = min(config.contextLength, 2048)

        initializeState()
    }

    private func initializeState() {
        convStates = []
        ssmStates = []
        kvCaches = []

        for l in 0..<config.nLayers {
            if config.isFullAttentionLayer(l) {
                kvCaches.append([Float](repeating: 0, count: maxSeqLen * 2 * kvDim))
            } else {
                convStates.append([Float](repeating: 0, count: (convKernel - 1) * totalConvChannels))
                ssmStates.append([Float](repeating: 0, count: nGroups * dState * dInnerPerGroup))
            }
        }
        position = 0
    }

    func resetCache() {
        initializeState()
    }

    // MARK: - Forward Pass (Single Token)

    func forwardStep(tokenId: Int) -> [Float] {
        var x = GGUFDequantizer.q8_0EmbedLookup(
            w: compiler.embeddingPtr, tokenId: tokenId, dim: dim)

        var mambaIdx = 0
        var attnIdx = 0

        for l in 0..<config.nLayers {
            switch compiler.layerWeights[l] {
            case .mamba(let w):
                x = mambaForward(x: x, w: w, mambaIdx: mambaIdx)
                mambaIdx += 1
            case .attention(let w):
                x = attentionForward(x: x, w: w, attnIdx: attnIdx)
                attnIdx += 1
            }
        }

        // Final RMSNorm
        x = rmsNorm(x, weight: compiler.rmsFinalWeight)

        // Classifier (tied with embedding weights)
        var logits = [Float](repeating: 0, count: vocabSize)
        x.withUnsafeBufferPointer { xBuf in
            logits.withUnsafeMutableBufferPointer { yBuf in
                GGUFDequantizer.q8_0Matvec(
                    w: compiler.embeddingPtr,
                    x: xBuf.baseAddress!, y: yBuf.baseAddress!,
                    outDim: vocabSize, inDim: dim)
            }
        }

        position += 1
        return logits
    }

    // MARK: - Mamba Layer Forward

    private func mambaForward(x: [Float], w: MambaLayerWeights, mambaIdx: Int) -> [Float] {
        let xNorm = rmsNorm(x, weight: w.attnNorm)

        // Project to xBC space
        let xBC = w.attnQKV.matvec(xNorm)

        // Causal 1D convolution
        let convOut = causalConv1d(input: xBC, mambaIdx: mambaIdx, weight: w.ssmConv1d)

        // Split: x_ssm, B, C
        var xSsm = Array(convOut[0..<ssmInner])
        let bFlat = Array(convOut[ssmInner..<(2 * ssmInner)])
        let cFlat = Array(convOut[(2 * ssmInner)..<(3 * ssmInner)])

        // SiLU on x_ssm
        for i in 0..<ssmInner {
            xSsm[i] = xSsm[i] / (1.0 + exp(-xSsm[i]))
        }

        // Compute dt
        let dtRaw = w.ssmAlpha.matvec(xNorm)
        var dt = [Float](repeating: 0, count: nGroups)
        for g in 0..<nGroups {
            dt[g] = log(1.0 + exp(dtRaw[g] + w.ssmDtBias[g]))  // softplus
        }

        // SSM selective scan
        var y = ssmScan(xSsm: xSsm, B: bFlat, C: cFlat, dt: dt,
                        A: w.ssmA, mambaIdx: mambaIdx)

        // Group RMSNorm
        y = groupRMSNorm(y, weight: w.ssmNorm)

        // Gate
        let gateRaw = w.attnGate.matvec(xNorm)
        for i in 0..<ssmInner {
            y[i] *= 1.0 / (1.0 + exp(-gateRaw[i]))
        }

        // Output projection + residual
        let out = w.ssmOut.matvec(y)
        var result = x
        for i in 0..<dim { result[i] += out[i] }

        // FFN
        let ffnNorm = rmsNorm(result, weight: w.postAttnNorm)
        let ffnOut = ffnForward(ffnNorm, gate: w.ffnGate, up: w.ffnUp, down: w.ffnDown)
        for i in 0..<dim { result[i] += ffnOut[i] }

        return result
    }

    // MARK: - Causal Conv1D

    private func causalConv1d(input: [Float], mambaIdx: Int, weight: [Float]) -> [Float] {
        let K = convKernel
        let C = totalConvChannels

        // Shift conv state left by C, append new input
        if K > 2 {
            for i in 0..<((K - 2) * C) {
                convStates[mambaIdx][i] = convStates[mambaIdx][i + C]
            }
        }
        for i in 0..<C {
            convStates[mambaIdx][(K - 2) * C + i] = input[i]
        }

        // Compute depthwise convolution
        // weight layout: [channels, kernel] with ne0=K, ne1=C
        var output = [Float](repeating: 0, count: C)
        for ch in 0..<C {
            var sum: Float = 0
            for k in 0..<(K - 1) {
                sum += weight[ch * K + k] * convStates[mambaIdx][k * C + ch]
            }
            sum += weight[ch * K + (K - 1)] * input[ch]
            output[ch] = sum
        }

        return output
    }

    // MARK: - SSM Selective Scan

    private func ssmScan(xSsm: [Float], B: [Float], C: [Float],
                         dt: [Float], A: [Float], mambaIdx: Int) -> [Float] {
        var y = [Float](repeating: 0, count: ssmInner)

        for g in 0..<nGroups {
            let aVal = -exp(A[g])
            let dtVal = dt[g]
            let dA = exp(dtVal * aVal)

            for j in 0..<dInnerPerGroup {
                let ch = g * dInnerPerGroup + j
                let xVal = xSsm[ch]
                var yVal: Float = 0

                for s in 0..<dState {
                    let si = g * dState * dInnerPerGroup + s * dInnerPerGroup + j
                    let bVal = B[g * dState + s]
                    ssmStates[mambaIdx][si] = dA * ssmStates[mambaIdx][si] + dtVal * bVal * xVal
                    yVal += C[g * dState + s] * ssmStates[mambaIdx][si]
                }
                y[ch] = yVal
            }
        }

        return y
    }

    // MARK: - Attention Layer Forward

    private func attentionForward(x: [Float], w: AttentionLayerWeights, attnIdx: Int) -> [Float] {
        let xNorm = rmsNorm(x, weight: w.attnNorm)

        // Q/K/V projections
        var q = w.attnQ.matvec(xNorm)
        var k = w.attnK.matvec(xNorm)
        let v = w.attnV.matvec(xNorm)

        // Per-head Q norm
        for h in 0..<nHeads {
            let off = h * headDim
            var slice = Array(q[off..<(off + headDim)])
            slice = rmsNorm(slice, weight: w.attnQNorm)
            for i in 0..<headDim { q[off + i] = slice[i] }
        }
        // Per-head K norm
        for h in 0..<nKVHeads {
            let off = h * headDim
            var slice = Array(k[off..<(off + headDim)])
            slice = rmsNorm(slice, weight: w.attnKNorm)
            for i in 0..<headDim { k[off + i] = slice[i] }
        }

        // RoPE (partial)
        applyRoPE(&q, nHeadsCount: nHeads)
        applyRoPE(&k, nHeadsCount: nKVHeads)

        // Store K/V in cache
        let pos = position
        for i in 0..<kvDim {
            kvCaches[attnIdx][pos * 2 * kvDim + i] = k[i]
            kvCaches[attnIdx][pos * 2 * kvDim + kvDim + i] = v[i]
        }

        // GQA Attention
        let gqaGroupSize = nHeads / max(nKVHeads, 1)
        var attnOut = [Float](repeating: 0, count: dim)
        let scale = 1.0 / sqrt(Float(headDim))

        for h in 0..<nHeads {
            let kvHead = h / gqaGroupSize
            let qOff = h * headDim

            // Scores
            var scores = [Float](repeating: 0, count: pos + 1)
            for p in 0...pos {
                let kOff = p * 2 * kvDim + kvHead * headDim
                var dot: Float = 0
                for d in 0..<headDim {
                    dot += q[qOff + d] * kvCaches[attnIdx][kOff + d]
                }
                scores[p] = dot * scale
            }

            softmax(&scores)

            // Weighted V
            for p in 0...pos {
                let vOff = p * 2 * kvDim + kvDim + kvHead * headDim
                for d in 0..<headDim {
                    attnOut[qOff + d] += scores[p] * kvCaches[attnIdx][vOff + d]
                }
            }
        }

        // Output projection + residual
        let projected = w.attnOutput.matvec(attnOut)
        var result = x
        for i in 0..<dim { result[i] += projected[i] }

        // FFN
        let ffnNorm = rmsNorm(result, weight: w.postAttnNorm)
        let ffnOut = ffnForward(ffnNorm, gate: w.ffnGate, up: w.ffnUp, down: w.ffnDown)
        for i in 0..<dim { result[i] += ffnOut[i] }

        return result
    }

    // MARK: - FFN (SwiGLU)

    private func ffnForward(_ x: [Float], gate: QWeight, up: QWeight, down: QWeight) -> [Float] {
        let gateOut = gate.matvec(x)
        let upOut = up.matvec(x)

        var hidden = [Float](repeating: 0, count: hiddenDim)
        for i in 0..<hiddenDim {
            let silu = gateOut[i] / (1.0 + exp(-gateOut[i]))
            hidden[i] = silu * upOut[i]
        }

        return down.matvec(hidden)
    }

    // MARK: - RoPE

    private func applyRoPE(_ vec: inout [Float], nHeadsCount: Int) {
        let halfRope = ropeDim / 2
        for h in 0..<nHeadsCount {
            let off = h * headDim
            for i in 0..<halfRope {
                let freq = 1.0 / pow(ropeTheta, Float(2 * i) / Float(ropeDim))
                let theta = Float(position) * freq
                let cosVal = cos(theta)
                let sinVal = sin(theta)
                let r0 = vec[off + 2 * i]
                let r1 = vec[off + 2 * i + 1]
                vec[off + 2 * i] = r0 * cosVal - r1 * sinVal
                vec[off + 2 * i + 1] = r0 * sinVal + r1 * cosVal
            }
        }
    }

    // MARK: - Helpers

    private func rmsNorm(_ x: [Float], weight: [Float]) -> [Float] {
        let n = x.count
        var sumSq: Float = 0
        for i in 0..<n { sumSq += x[i] * x[i] }
        let rms = sqrt(sumSq / Float(n) + rmsEps)
        var result = [Float](repeating: 0, count: n)
        for i in 0..<n { result[i] = (x[i] / rms) * weight[i] }
        return result
    }

    private func groupRMSNorm(_ x: [Float], weight: [Float]) -> [Float] {
        var result = x
        for g in 0..<nGroups {
            let off = g * dInnerPerGroup
            var sumSq: Float = 0
            for j in 0..<dInnerPerGroup { sumSq += result[off + j] * result[off + j] }
            let rms = sqrt(sumSq / Float(dInnerPerGroup) + rmsEps)
            for j in 0..<dInnerPerGroup {
                result[off + j] = (result[off + j] / rms) * weight[j]
            }
        }
        return result
    }

    private func softmax(_ x: inout [Float]) {
        let maxVal = x.max() ?? 0
        var sumExp: Float = 0
        for i in 0..<x.count {
            x[i] = exp(x[i] - maxVal)
            sumExp += x[i]
        }
        for i in 0..<x.count { x[i] /= sumExp }
    }

    // MARK: - Token Generation

    func generate(promptTokens: [Int], tokenizer: GGUFTokenizer,
                  onToken: (Int, String) -> Bool) -> String {
        resetCache()
        var output = ""

        // Prefill
        var lastLogits = [Float]()
        for token in promptTokens {
            lastLogits = forwardStep(tokenId: token)
        }

        // Autoregressive generation
        for _ in 0..<maxTokens {
            let nextToken = sampleToken(logits: lastLogits)
            if tokenizer.isEOS(nextToken) { break }
            if position >= maxSeqLen - 1 { break }

            let text = tokenizer.decodeOne(nextToken)
            output += text
            if !onToken(nextToken, text) { break }

            lastLogits = forwardStep(tokenId: nextToken)
        }

        return output
    }

    // MARK: - Sampling

    private func sampleToken(logits: [Float]) -> Int {
        if temperature <= 0 {
            return logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        }

        var scaled = logits.map { $0 / temperature }

        // Top-K
        if topK > 0 && topK < scaled.count {
            let sorted = scaled.enumerated().sorted { $0.element > $1.element }
            let threshold = sorted[topK - 1].element
            for i in 0..<scaled.count {
                if scaled[i] < threshold { scaled[i] = -Float.infinity }
            }
        }

        // Softmax
        let maxVal = scaled.max() ?? 0
        var probs = scaled.map { exp($0 - maxVal) }
        let sumProbs = probs.reduce(0, +)
        for i in 0..<probs.count { probs[i] /= sumProbs }

        // Top-P
        if topP < 1.0 {
            let sorted = probs.enumerated().sorted { $0.element > $1.element }
            var cumProb: Float = 0
            var cutoff = probs.count
            for (idx, (_, p)) in sorted.enumerated() {
                cumProb += p
                if cumProb >= topP { cutoff = idx + 1; break }
            }
            let allowed = Set(sorted.prefix(cutoff).map { $0.offset })
            for i in 0..<probs.count {
                if !allowed.contains(i) { probs[i] = 0 }
            }
            let newSum = probs.reduce(0, +)
            if newSum > 0 { for i in 0..<probs.count { probs[i] /= newSum } }
        }

        // Multinomial
        let r = Float.random(in: 0..<1)
        var cum: Float = 0
        for i in 0..<probs.count {
            cum += probs[i]
            if cum >= r { return i }
        }
        return probs.count - 1
    }
}
