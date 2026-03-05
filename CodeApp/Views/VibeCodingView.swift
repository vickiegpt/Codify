//
//  VibeCodingView.swift
//  Code
//
//  Created by Claude on 18/02/2026.
//
//  Vibe coding integration helpers used by ChatView and MonacoImplementation.
//  The vibe coding UI is integrated directly into the AI Chat panel.
//

import Foundation

enum VibeCodingNotification {
    static let editSelection = Notification.Name("vibeCoding.editSelection")
    static let selectedCodeKey = "selectedCode"
}
