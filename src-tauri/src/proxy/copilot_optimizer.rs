//! Copilot 请求优化器
//!
//! 解决 GitHub Copilot 代理消耗量异常问题（Issue #1813）。
//!
//! Copilot 使用 `x-initiator` 请求头区分「用户发起」和「agent 续写」：
//! - `user`：计为一次 premium interaction（扣额度）
//! - `agent`：视为上一次交互的延续（不额外扣费）
//!
//! 参考实现: https://github.com/caozhiyuan/copilot-api

use std::collections::HashSet;

use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// 请求分类结果
#[derive(Debug, Clone)]
pub struct CopilotClassification {
    /// "user" 或 "agent" — 映射到 x-initiator 请求头
    pub initiator: &'static str,
    /// 是否为 warmup/探针请求（可降级到小模型）
    pub is_warmup: bool,
    /// 是否为上下文压缩请求
    pub is_compact: bool,
    /// 是否为 Claude Code 子代理请求（Agent tool 生成的 subagent）
    /// 子代理请求应设置 x-interaction-type=conversation-subagent，不计 premium interaction
    pub is_subagent: bool,
}

/// Classify the latest user/tool input without applying Claude-specific heuristics.
pub fn classify_responses_request(body: &Value) -> CopilotClassification {
    let initiator_for_item = |item: &Value| {
        let item_type = item.get("type").and_then(Value::as_str);
        if item_type.is_some_and(super::providers::codex_chat_history::is_call_output_item_type) {
            return Some("agent");
        }
        match item.get("role").and_then(Value::as_str) {
            Some("user") => Some("user"),
            Some("assistant" | "system" | "developer") => None,
            _ => match item_type {
                Some("reasoning" | "function_call" | "custom_tool_call" | "tool_search_call") => {
                    None
                }
                _ => Some("user"),
            },
        }
    };
    // Full-history requests can contain old tool outputs before a new user turn.
    let initiator = match body.get("input") {
        Some(Value::Array(items)) => items.iter().rev().find_map(initiator_for_item),
        Some(item @ Value::Object(_)) => initiator_for_item(item),
        _ => None,
    }
    .unwrap_or("user");

    CopilotClassification {
        initiator,
        is_warmup: false,
        is_compact: false,
        is_subagent: false,
    }
}

/// 分类 Anthropic 格式的请求体，决定 Copilot 请求头。
///
/// 分类算法（只检查最后一条消息，与参考实现 caozhiyuan/copilot-api 对齐）：
/// 1. 无消息 → "user"（安全默认，首次请求）
/// 2. 最后消息 role=user：
///    - content 中存在非 tool_result 类型 block → "user"
///    - content 全部是 tool_result → "agent"
///    - 匹配 compact 模式 → "agent"
/// 3. 最后消息 role 非 user → "user"（安全默认）
///
/// Warmup 检测（与参考实现对齐）：
/// - 请求头中有 `anthropic-beta` + 无 tools + 非 compact → warmup
///
/// `compact_detection`：是否启用 compact 检测。为 false 时跳过，
/// 确保 `CopilotOptimizerConfig.compact_detection` 开关真正生效。
///
/// `subagent_detection`：是否启用子代理检测。为 true 时，会扫描首条用户消息
/// 中的 `__SUBAGENT_MARKER__` 标记，将子代理请求标记为不计费。
pub fn classify_request(
    body: &Value,
    has_anthropic_beta: bool,
    compact_detection: bool,
    subagent_detection: bool,
) -> CopilotClassification {
    let is_compact = compact_detection && is_compact_request(body);
    let is_subagent = subagent_detection && detect_subagent(body);

    let messages = match body.get("messages").and_then(|m| m.as_array()) {
        Some(msgs) if !msgs.is_empty() => msgs,
        _ => {
            return CopilotClassification {
                initiator: "user",
                is_warmup: is_warmup_request(body, has_anthropic_beta, false),
                is_compact: false,
                is_subagent,
            }
        }
    };

    let last_msg = &messages[messages.len() - 1];
    let role = last_msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

    // 只有 role=user 的消息需要细分
    if role != "user" {
        return CopilotClassification {
            initiator: if is_subagent { "agent" } else { "user" },
            is_warmup: false,
            is_compact,
            is_subagent,
        };
    }

    // 判定逻辑（与 copilot-api 的 merge-then-classify 效果对齐）：
    // 只要 content 数组中包含 tool_result → 视为工具续写 → agent
    // 这覆盖了 skill/edit hook/plan follow-up 等常见场景，
    // 它们的 content 通常是 [tool_result, text] 混合形态。
    // copilot-api 通过先 merge（text 吸收进 tool_result）再 classify 实现同等效果；
    // 直接在分类层处理更稳健，不依赖 merge 启用状态和执行顺序。
    let is_user_initiated = match last_msg.get("content") {
        Some(Value::Array(blocks)) => {
            // 含有 tool_result → 工具续写（agent），否则 → 用户发起（user）
            !blocks
                .iter()
                .any(|block| block.get("type").and_then(|t| t.as_str()) == Some("tool_result"))
        }
        Some(Value::String(_)) => true,
        _ => false,
    };

    // 子代理请求始终标记为 agent（即使首条消息包含用户文本）
    let initiator = if is_subagent || !is_user_initiated || is_compact {
        "agent"
    } else {
        "user"
    };

    CopilotClassification {
        initiator,
        is_warmup: initiator == "user" && is_warmup_request(body, has_anthropic_beta, is_compact),
        is_compact,
        is_subagent,
    }
}

/// 检测是否为 warmup/探针请求（适合降级到小模型）。
///
/// 与参考实现对齐，三个条件同时满足：
/// 1. 请求头有 `anthropic-beta`（Claude Code warmup 探针的标志）
/// 2. 无 tools 定义
/// 3. 非 compact 请求
fn is_warmup_request(body: &Value, has_anthropic_beta: bool, is_compact: bool) -> bool {
    if !has_anthropic_beta || is_compact {
        return false;
    }
    // 无工具定义
    body.get("tools")
        .and_then(|tools| tools.as_array())
        .is_none_or(|tools| tools.is_empty())
}

/// 检测是否为 Claude Code 上下文压缩/compact 请求。
///
/// 只匹配 Claude Code **内部生成**的机器特征，不匹配用户可能手动输入的通用短语，
/// 避免将真实用户请求误标为 agent。
///
/// 强特征来源：
/// 1. system prompt — Claude Code compact 模式会设置专用 system prompt，用户无法手动设置
/// 2. "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools." — 机器指令
/// 3. 同时包含 "Pending Tasks:" 和 "Current Work:" — Claude Code compact 的结构标记
pub(super) fn is_compact_request(body: &Value) -> bool {
    // 信号 1: system prompt 以 Claude Code compact 专用前缀开头
    // 用户在 Claude Code 中无法直接控制 system prompt，这是最可靠的信号
    let system_text = extract_system_text(body);
    if system_text
        .starts_with("You are a helpful AI assistant tasked with summarizing conversations")
    {
        return true;
    }

    // 信号 2 & 3: 检查最后一条用户消息中的机器生成特征
    let messages = match body.get("messages").and_then(|m| m.as_array()) {
        Some(msgs) => msgs,
        None => return false,
    };

    if let Some(last_msg) = messages.last() {
        if last_msg.get("role").and_then(|r| r.as_str()) != Some("user") {
            return false;
        }

        let text = extract_text_from_message(last_msg);

        // 信号 2: Claude Code compact 的机器指令（大小写敏感，精确匹配）
        if text.contains("CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.") {
            return true;
        }

        // 信号 3: Claude Code compact 的结构标记（两个同时出现才算）
        if text.contains("Pending Tasks:") && text.contains("Current Work:") {
            return true;
        }
    }

    false
}

/// 合并用户消息中的 tool_result 和 text block。
///
/// 与参考实现 `mergeToolResultForClaude` 对齐：
///
/// **消息内部合并**（核心）：在单条 user 消息内，将 text block 吸收进 tool_result block，
/// 使整条消息只剩 tool_result 类型 block。这样 Copilot 不会将其视为用户发起的交互。
///
/// 场景：Claude Code 在 skill 调用、edit hook、plan 提醒等场景下，会发送混合了
/// tool_result + text 的用户消息。text block 的存在让 Copilot 将其计为 premium request。
///
/// **跨消息合并**（补充）：连续的 tool_result-only 用户消息合并为一条。
#[cfg(test)]
pub fn merge_tool_results(body: Value) -> Value {
    merge_tool_results_with_changes(body).0
}

pub(crate) fn merge_tool_results_with_changes(mut body: Value) -> (Value, bool) {
    let mut changed = false;
    let messages = match body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        Some(msgs) if !msgs.is_empty() => msgs,
        _ => return (body, changed),
    };

    // Phase 1: 消息内部合并 — 将 text block 吸收进 tool_result block
    for msg in messages.iter_mut() {
        if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
            continue;
        }
        let content = match msg.get("content").and_then(|c| c.as_array()) {
            Some(blocks) => blocks,
            None => continue,
        };

        // 分离 tool_result 和 text block
        let mut tool_results: Vec<Value> = Vec::new();
        let mut text_blocks: Vec<Value> = Vec::new();
        let mut valid = true;

        for block in content {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("tool_result") => tool_results.push(block.clone()),
                Some("text") => text_blocks.push(block.clone()),
                _ => {
                    // 存在其他类型 block → 跳过此消息
                    valid = false;
                    break;
                }
            }
        }

        // 必须同时有 tool_result 和 text 才需要合并
        if !valid || tool_results.is_empty() || text_blocks.is_empty() {
            continue;
        }

        // 合并策略（与参考实现对齐）
        let merged = merge_blocks_into_tool_results(tool_results, text_blocks);
        msg["content"] = Value::Array(merged);
        changed = true;
    }

    // Phase 2: 跨消息合并 — 连续的 tool_result-only 用户消息合并
    if !messages
        .windows(2)
        .any(|pair| is_tool_result_only_message(&pair[0]) && is_tool_result_only_message(&pair[1]))
    {
        return (body, changed);
    }

    let mut merged_msgs: Vec<Value> = Vec::with_capacity(messages.len());
    let mut messages = std::mem::take(messages).into_iter().peekable();
    while let Some(mut message) = messages.next() {
        if is_tool_result_only_message(&message)
            && messages.peek().is_some_and(is_tool_result_only_message)
        {
            let mut combined_content = std::mem::take(message["content"].as_array_mut().unwrap());
            while messages.peek().is_some_and(is_tool_result_only_message) {
                let mut next = messages.next().unwrap();
                combined_content.append(next["content"].as_array_mut().unwrap());
            }
            message = serde_json::json!({
                "role": "user",
                "content": combined_content
            });
            changed = true;
        }
        merged_msgs.push(message);
    }

    body["messages"] = Value::Array(merged_msgs);
    (body, changed)
}

/// 基于最后一条用户消息内容生成确定性 Request ID。
///
/// CC Switch 额外策略（参考项目 copilot-api 使用随机 UUID）：
/// - 哈希输入: sessionId + lastUserContent（排除 tool_result 和 cache_control）
/// - 相同内容产生相同 ID，可能帮助 Copilot 去重
/// - 找不到用户内容时退化为随机 UUID
/// - 使用 UUID v4 格式
pub fn deterministic_request_id(body: &Value, session_id: &str) -> String {
    let last_user_content = find_last_user_content(body);

    match last_user_content {
        Some(content) => {
            let mut hasher = Sha256::new();
            hasher.update(session_id.as_bytes());
            hasher.update(content.as_bytes());
            let result = hasher.finalize();

            let mut bytes = [0u8; 16];
            bytes.copy_from_slice(&result[..16]);
            // UUID v4 版本位和变体位（与参考实现一致）
            bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
            bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 1

            Uuid::from_bytes(bytes).to_string()
        }
        None => Uuid::new_v4().to_string(),
    }
}

/// 基于 session ID 生成稳定的 Interaction ID。
///
/// 与参考实现（copilot-api session.ts）对齐：
/// - 同一主对话的所有请求共享同一个 interaction ID
/// - 哈希输入: 仅 session ID（不包含消息内容，与 request ID 不同）
/// - Copilot 用此 ID 将请求聚合为同一个 "interaction"，影响 premium 计费归属
/// - 空 session ID 时返回 None（不应注入随机值，避免 interaction 碎片化）
pub fn deterministic_interaction_id(session_id: &str) -> Option<String> {
    if session_id.is_empty() {
        return None;
    }

    let mut hasher = Sha256::new();
    hasher.update(b"interaction:");
    hasher.update(session_id.as_bytes());
    let result = hasher.finalize();

    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&result[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 1

    Some(Uuid::from_bytes(bytes).to_string())
}

/// 检测请求是否来自 Claude Code 子代理（Agent tool 生成的 subagent）。
///
/// Claude Code 的 Agent tool 会在子代理首条用户消息的 `<system-reminder>` 标签中
/// 注入 `__SUBAGENT_MARKER__` JSON 标记，格式如：
/// ```json
/// {"__SUBAGENT_MARKER__": {"session_id": "...", "agent_id": "...", "agent_type": "..."}}
/// ```
///
/// 扫描策略（与 copilot-api 的 subagent-marker.ts 对齐）：
/// 1. 遍历所有 user 消息（不仅是第一条，因为 context 压缩可能重排消息）
/// 2. 在消息文本中查找 `__SUBAGENT_MARKER__` 关键字
/// 3. 找到即判定为子代理请求
fn detect_subagent(body: &Value) -> bool {
    // 信号 1: 显式 __SUBAGENT_MARKER__（Claude Code 2.x+ 自动注入）
    if extract_system_text(body).contains("__SUBAGENT_MARKER__") {
        return true;
    }

    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
                continue;
            }
            let text = extract_text_from_message(msg);
            if text.contains("__SUBAGENT_MARKER__") {
                return true;
            }
        }
    }

    // 信号 2（fallback）: metadata.user_id 包含子代理标识
    // Claude Code 的 Agent tool 会将 subagent session 标记为
    // "parentSessionId_agent_agentId" 格式，检测 "_agent_" 后缀
    if let Some(user_id) = body.pointer("/metadata/user_id").and_then(|v| v.as_str()) {
        // "_agent_" 是 Claude Code Agent tool 的内部标记
        if user_id.contains("_agent_") {
            return true;
        }
    }

    // 信号 3（fallback）: system prompt 包含 Claude Code 子代理的典型框架文本
    // Agent tool 生成的子代理会在 system prompt 中包含由 Agent tool 注入的任务描述，
    // 但主对话的 system prompt 由 Claude Code CLI 直接生成，两者格式不同
    // 这个信号不够可靠（用户 prompt 也可能包含这些词），因此只作为辅助判据
    // 暂不启用，预留接口

    false
}

/// 清理孤立的 tool_result — 没有对应 tool_use 的 tool_result 转为 text block。
///
/// 场景：上下文压缩、消息截断等可能导致 assistant 消息中的 tool_use 被删除，
/// 但后续 user 消息中的 tool_result 仍在。上游 API 可能因不匹配而报错/重试。
///
/// 与 copilot-api 的 `sanitizeOrphanToolResults` 对齐。
#[cfg(test)]
pub fn sanitize_orphan_tool_results(body: Value) -> Value {
    sanitize_orphan_tool_results_with_changes(body).0
}

pub(crate) fn sanitize_orphan_tool_results_with_changes(mut body: Value) -> (Value, bool) {
    let mut changed = false;
    let messages = match body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        Some(msgs) if msgs.len() >= 2 => msgs,
        _ => return (body, changed),
    };

    // Anthropic 协议要求 tool_result 紧跟其对应 tool_use 所在的 assistant turn。
    // 只检查 messages[i-1]（紧邻上一条 assistant）来判定是否 orphan，
    // 与参考实现 sanitizeOrphanToolResults 对齐。
    for i in 1..messages.len() {
        if messages[i].get("role").and_then(|r| r.as_str()) != Some("user") {
            continue;
        }

        // 收集紧邻上一条 assistant 的 tool_use id
        let prev_tool_use_ids: HashSet<String> =
            if messages[i - 1].get("role").and_then(|r| r.as_str()) == Some("assistant") {
                messages[i - 1]
                    .get("content")
                    .and_then(|c| c.as_array())
                    .map(|blocks| {
                        blocks
                            .iter()
                            .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
                            .filter_map(|b| b.get("id").and_then(|i| i.as_str()).map(String::from))
                            .collect()
                    })
                    .unwrap_or_default()
            } else {
                // 上一条不是 assistant → 这条 user 中的所有 tool_result 都是 orphan
                HashSet::new()
            };

        let content = match messages[i]
            .get_mut("content")
            .and_then(|c| c.as_array_mut())
        {
            Some(blocks) => blocks,
            None => continue,
        };

        for block in content.iter_mut() {
            if block.get("type").and_then(|t| t.as_str()) != Some("tool_result") {
                continue;
            }
            let tool_use_id = block
                .get("tool_use_id")
                .and_then(|id| id.as_str())
                .unwrap_or("");
            // 空 tool_use_id 或不在紧邻 assistant 的 tool_use 中 → orphan
            if tool_use_id.is_empty() || !prev_tool_use_ids.contains(tool_use_id) {
                let content_text = match block.get("content") {
                    Some(Value::String(text)) => text.clone(),
                    Some(Value::Array(blocks)) => blocks
                        .iter()
                        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    _ => String::new(),
                };
                changed = true;
                *block = serde_json::json!({
                    "type": "text",
                    "text": format!("[Tool result for {}]: {}", tool_use_id, content_text)
                });
            }
        }
    }

    (body, changed)
}

/// 请求前主动剥离所有 assistant 消息里的 thinking / redacted_thinking block
///
/// Copilot 的三条目标端点（`/chat/completions`、`/v1/responses`、`/v1/chat/completions`）
/// 均为 OpenAI 兼容格式，不识别 Anthropic 的 thinking block。若原样转发，上游会
/// 拒绝并返回 invalid_request_error —— 届时 `thinking_rectifier` 才做反应式清理并
/// 重试。那次已经失败的请求依旧消耗一次 premium quota，所以此处提前剥离。
///
/// 与 `thinking_rectifier::rectify_anthropic_request` 的区别：
/// - 本函数只剥 thinking / redacted_thinking 两类 block，不触碰 signature，也不
///   移除顶层 thinking 字段——那些是错误路径上的激进整流，常规路径不需要。
/// - 保持与 `merge_tool_results` / `sanitize_orphan_tool_results` 一致的"消费 body、
///   返回新 body"签名，便于接入 forwarder 管道。
pub fn strip_thinking_blocks(mut body: Value) -> Value {
    let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return body;
    };

    for msg in messages.iter_mut() {
        if msg.get("role").and_then(|r| r.as_str()) != Some("assistant") {
            continue;
        }
        let Some(content) = msg.get_mut("content").and_then(|c| c.as_array_mut()) else {
            continue;
        };
        content.retain(|block| {
            !matches!(
                block.get("type").and_then(|t| t.as_str()),
                Some("thinking") | Some("redacted_thinking")
            )
        });
    }

    body
}

// ─── 内部辅助 ─────────────────────────────────

/// 从请求体的 `system` 字段提取文本（处理 string/array 两种格式）。
fn extract_system_text(body: &Value) -> String {
    match body.get("system") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// 查找最后一条 user 消息的非 tool_result 文本内容。
///
/// 与参考实现的 `findLastUserContent` 对齐：
/// - 从后往前遍历消息
/// - 排除 tool_result block
/// - 排除 cache_control 字段
fn find_last_user_content(body: &Value) -> Option<String> {
    let messages = body.get("messages").and_then(|m| m.as_array())?;

    for msg in messages.iter().rev() {
        if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
            continue;
        }
        let content = msg.get("content")?;

        if let Some(s) = content.as_str() {
            return Some(s.to_string());
        }

        if let Some(blocks) = content.as_array() {
            // 过滤 tool_result，保留其他 block（去掉 cache_control）
            let filtered: Vec<Value> = blocks
                .iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) != Some("tool_result"))
                .map(|b| {
                    let mut b = b.clone();
                    if let Some(obj) = b.as_object_mut() {
                        obj.remove("cache_control");
                    }
                    b
                })
                .collect();

            if !filtered.is_empty() {
                return Some(serde_json::to_string(&filtered).unwrap_or_default());
            }
        }
    }

    None
}

/// 将 text block 合并进 tool_result block。
///
/// 两种合并策略（与参考实现对齐）：
/// - 数量相等：一一对应，text 追加到对应 tool_result 的 content 中
/// - 数量不等：所有 text 追加到最后一个 tool_result 的 content 中
fn merge_blocks_into_tool_results(
    mut tool_results: Vec<Value>,
    text_blocks: Vec<Value>,
) -> Vec<Value> {
    if tool_results.len() == text_blocks.len() {
        // 一一对应合并
        for (tr, tb) in tool_results.iter_mut().zip(text_blocks.iter()) {
            append_text_to_tool_result(tr, tb);
        }
    } else {
        // 所有 text 追加到最后一个 tool_result
        if let Some(last_tr) = tool_results.last_mut() {
            for tb in &text_blocks {
                append_text_to_tool_result(last_tr, tb);
            }
        }
    }
    tool_results
}

/// 将 text block 的内容追加到 tool_result 的 content 中
fn append_text_to_tool_result(tool_result: &mut Value, text_block: &Value) {
    let text = text_block
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or("");
    if text.trim().is_empty() {
        return;
    }

    // tool_result 的 content 可以是字符串或数组
    match tool_result.get_mut("content") {
        Some(Value::String(existing)) => {
            existing.push('\n');
            existing.push_str(text);
        }
        Some(Value::Array(arr)) => {
            arr.push(serde_json::json!({"type": "text", "text": text}));
        }
        _ => {
            // content 缺失或 null — 直接设置
            tool_result["content"] = Value::String(text.to_string());
        }
    }
}

/// 从消息中提取文本内容
fn extract_text_from_message(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| {
                if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                    block.get("text").and_then(|t| t.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// 判断消息是否为 tool_result-only 的用户消息
fn is_tool_result_only_message(msg: &Value) -> bool {
    if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
        return false;
    }
    match msg.get("content").and_then(|c| c.as_array()) {
        Some(blocks) if !blocks.is_empty() => blocks
            .iter()
            .all(|block| block.get("type").and_then(|t| t.as_str()) == Some("tool_result")),
        _ => false,
    }
}

// ─── 测试 ─────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn classify_responses_request_cases() {
        let tool = json!({
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "done"
        });
        let user = json!({"role": "user", "content": "A new question"});
        for (input, expected) in [
            (json!([user.clone(), tool.clone()]), "agent"),
            (json!([tool.clone(), user.clone()]), "user"),
            (
                json!([
                    user.clone(),
                    tool.clone(),
                    {"type": "reasoning", "summary": []},
                    {"role": "developer", "content": "Use the output"}
                ]),
                "agent",
            ),
            (
                json!([
                    tool.clone(),
                    user,
                    {"role": "system", "content": "Continue"}
                ]),
                "user",
            ),
            (tool.clone(), "agent"),
            (json!("A new question"), "user"),
            (json!([]), "user"),
            (json!(null), "user"),
            (json!([tool, {"type": "unknown"}]), "user"),
        ] {
            let body = json!({"input": input, "previous_response_id": "resp-previous"});
            let result = classify_responses_request(&body);
            assert_eq!(result.initiator, expected, "{body}");
            assert!(!result.is_warmup && !result.is_compact && !result.is_subagent);
        }

        // 不套用 Claude messages 启发式（compact / tool_result / subagent 标记一律忽略）
        for body in [
            json!({}),
            json!({
                "input": "A new question",
                "system": "You are a helpful AI assistant tasked with summarizing conversations",
                "messages": [{"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": "done"},
                    {"type": "text", "text": "__SUBAGENT_MARKER__"}
                ]}]
            }),
        ] {
            let result = classify_responses_request(&body);
            assert_eq!(result.initiator, "user");
            assert!(!result.is_warmup && !result.is_compact && !result.is_subagent);
        }
    }

    #[test]
    fn classify_request_initiator_and_warmup_rules() {
        let tools = json!([{"name": "Read", "description": "Read a file", "input_schema": {}}]);
        for (case, body, has_beta, expected_initiator, expected_warmup) in [
            (
                "user_text",
                json!({"messages": [{"role": "user", "content": "Hello, write code"}]}),
                false,
                "user",
                false,
            ),
            (
                "user_text_array",
                json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "Explain"}]}]}),
                false,
                "user",
                false,
            ),
            (
                "tool_result_only_with_tools",
                json!({
                    "tools": tools,
                    "messages": [
                        {"role": "user", "content": "Read the file"},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": "Reading."},
                            {"type": "tool_use", "id": "toolu_123", "name": "Read", "input": {}}
                        ]},
                        {"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_123", "content": "ok"}
                        ]}
                    ]
                }),
                true,
                "agent",
                false,
            ),
            (
                "tool_result_with_text_block",
                json!({"messages": [{"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_123", "content": "file"},
                    {"type": "text", "text": "Now refactor"}
                ]}]}),
                false,
                "agent",
                false,
            ),
            (
                "empty_messages",
                json!({"messages": []}),
                false,
                "user",
                false,
            ),
            (
                "no_messages",
                json!({"model": "claude-sonnet-4-20250514"}),
                false,
                "user",
                false,
            ),
            // Warmup: has_anthropic_beta && no tools && initiator == user
            (
                "warmup_beta_no_tools",
                json!({"messages": [{"role": "user", "content": "Hello"}]}),
                true,
                "user",
                true,
            ),
            (
                "not_warmup_with_tools",
                json!({"tools": tools, "messages": [{"role": "user", "content": "Hello"}]}),
                true,
                "user",
                false,
            ),
            (
                "not_warmup_when_agent",
                json!({"messages": [{"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_123", "content": "ok"}
                ]}]}),
                true,
                "agent",
                false,
            ),
        ] {
            let result = classify_request(&body, has_beta, true, false);
            assert_eq!(result.initiator, expected_initiator, "{case}");
            assert_eq!(result.is_warmup, expected_warmup, "{case}");
            assert!(!result.is_compact, "{case}");
        }
    }

    #[test]
    fn compact_detection_and_toggle_rules() {
        for (case, body, expected_compact) in [
            (
                "system_prompt_string",
                json!({
                    "system": "You are a helpful AI assistant tasked with summarizing conversations. Please summarize.",
                    "messages": [{"role": "user", "content": "Here is the history..."}]
                }),
                true,
            ),
            (
                "system_prompt_array",
                json!({
                    "system": [{"type": "text", "text": "You are a helpful AI assistant tasked with summarizing conversations."}],
                    "messages": [{"role": "user", "content": "Summarize"}]
                }),
                true,
            ),
            (
                "critical_marker_block",
                json!({"messages": [{"role": "user", "content": [
                    {"type": "text", "text": "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools. Summarize."}
                ]}]}),
                true,
            ),
            (
                "structural_markers",
                json!({"messages": [{"role": "user", "content": "Summary:\n\nPending Tasks:\n- Fix\n\nCurrent Work:\n- Feat"}]}),
                true,
            ),
            // 防误报：普通用户请求总结不算 compact
            (
                "user_summarize_not_compact",
                json!({"messages": [{"role": "user", "content": "Please summarize the conversation so far into a concise summary."}]}),
                false,
            ),
            (
                "generic_summary_not_compact",
                json!({"messages": [{"role": "user", "content": "Your task is to create a detailed summary of the conversation so far."}]}),
                false,
            ),
            (
                "ordinary_question",
                json!({"messages": [{"role": "user", "content": "What is the weather today?"}]}),
                false,
            ),
        ] {
            assert_eq!(is_compact_request(&body), expected_compact, "{case}");
            let classified = classify_request(&body, false, true, false);
            assert_eq!(classified.is_compact, expected_compact, "{case}");
            assert_eq!(
                classified.initiator,
                if expected_compact { "agent" } else { "user" },
                "{case}"
            );
        }

        // compact_detection=false 必须关闭检测
        let compact_body = json!({
            "system": "You are a helpful AI assistant tasked with summarizing conversations.",
            "messages": [{"role": "user", "content": "Summarize"}]
        });
        let disabled = classify_request(&compact_body, false, false, false);
        assert_eq!(disabled.initiator, "user");
        assert!(!disabled.is_compact);
    }

    #[test]
    fn subagent_detection_and_toggle_rules() {
        let user_marker_body = json!({"messages": [{"role": "user", "content": [{
            "type": "text",
            "text": "<system-reminder>\n{\"__SUBAGENT_MARKER__\":{\"session_id\":\"abc\",\"agent_id\":\"explore-1\"}}\n</system-reminder>\nSearch"
        }]}]});
        for (case, body, expected_subagent) in [
            ("user_marker", user_marker_body.clone(), true),
            (
                "system_marker",
                json!({
                    "system": "Agent {\"__SUBAGENT_MARKER__\":{\"agent_id\":\"plan-1\"}}",
                    "messages": [{"role": "user", "content": "Design plan"}]
                }),
                true,
            ),
            (
                "metadata_user_id_agent",
                json!({
                    "metadata": {"user_id": "session_abc123_agent_explore-1"},
                    "messages": [{"role": "user", "content": "Search"}]
                }),
                true,
            ),
            (
                "normal_message",
                json!({"messages": [{"role": "user", "content": "Hello"}]}),
                false,
            ),
            (
                "normal_metadata_user_id",
                json!({
                    "metadata": {"user_id": "session_abc123"},
                    "messages": [{"role": "user", "content": "Hello"}]
                }),
                false,
            ),
        ] {
            assert_eq!(detect_subagent(&body), expected_subagent, "{case}");
            let classified = classify_request(&body, false, true, true);
            assert_eq!(classified.is_subagent, expected_subagent, "{case}");
            assert_eq!(
                classified.initiator,
                if expected_subagent { "agent" } else { "user" },
                "{case}"
            );
        }

        // subagent_detection=false 必须关闭检测
        let disabled = classify_request(&user_marker_body, false, true, false);
        assert_eq!(disabled.initiator, "user");
        assert!(!disabled.is_subagent);
    }

    #[test]
    fn merge_tool_results_intra_message_and_normal_cases() {
        // 单消息内 tool_result + text -> text 吸收进 tool_result
        let merged = merge_tool_results(json!({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "file contents"},
                {"type": "text", "text": "skill output here"}
            ]}]
        }));
        let content = merged["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "tool_result");
        let text = content[0]["content"].as_str().unwrap();
        assert!(text.contains("file contents") && text.contains("skill output here"));

        // 数量相等：一一对应合并
        let equal = merge_tool_results(json!({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "result1"},
                {"type": "text", "text": "text1"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "result2"},
                {"type": "text", "text": "text2"}
            ]}]
        }));
        let content = equal["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert!(content[0]["content"].as_str().unwrap().contains("text1"));
        assert!(content[1]["content"].as_str().unwrap().contains("text2"));

        // 空 text block 不追加内容
        let empty_text = merge_tool_results(json!({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "result"},
                {"type": "text", "text": ""}
            ]}]
        }));
        let content = empty_text["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["content"], "result");

        // 含非 tool_result/text 块（image）-> 跳过整条消息
        let with_image = json!({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "result"},
                {"type": "image", "source": {"data": "..."}},
                {"type": "text", "text": "caption"}
            ]}]
        });
        assert_eq!(merge_tool_results(with_image.clone()), with_image);

        // 普通消息不变
        let normal = json!({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"}
            ]
        });
        assert_eq!(merge_tool_results(normal.clone()), normal);
    }

    #[test]
    fn merge_tool_results_with_changes_tracks_merges_without_cloning_history() {
        // 孤立 tool_result 不触发合并，不重分配大历史字符串，且保留 metadata
        let isolated = json!({"messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "x".repeat(1024 * 1024)},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "one", "content": "result"}], "metadata": {"keep": true}},
            {"role": "assistant", "content": "next"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "two", "content": "result"}]}
        ]});
        let expected = isolated.clone();
        let history_ptr = isolated["messages"][1]["content"]
            .as_str()
            .unwrap()
            .as_ptr();
        let (result, changed) = merge_tool_results_with_changes(isolated);
        assert!(!changed);
        assert_eq!(result, expected);
        assert_eq!(
            result["messages"][1]["content"].as_str().unwrap().as_ptr(),
            history_ptr
        );

        // 连续 tool_result-only 用户消息跨消息合并并标记 changed=true，同时保留前面孤立消息的 metadata
        let single = json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "one", "content": "result"}
        ], "metadata": {"keep": true}});
        let (result, changed) = merge_tool_results_with_changes(json!({"messages": [
            single,
            {"role": "assistant", "content": "next"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "two", "content": "second"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "three", "content": "third"}]}
        ]}));
        assert!(changed);
        assert_eq!(result["messages"].as_array().unwrap().len(), 3);
        assert_eq!(result["messages"][0], single);
        assert_eq!(result["messages"][2]["content"][0]["tool_use_id"], "two");
        assert_eq!(result["messages"][2]["content"][1]["tool_use_id"], "three");

        // 消息内合并同样必须标记 changed=true（后续需继续做隐私过滤）
        let (result, changed) = merge_tool_results_with_changes(json!({"messages": [{
            "role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "one", "content": "result"},
                {"type": "text", "text": "hook reminder"}
            ]
        }]}));
        assert!(changed);
        assert_eq!(
            result["messages"][0]["content"].as_array().unwrap().len(),
            1
        );
    }

    #[test]
    fn deterministic_request_and_interaction_id_invariants() {
        let hello = json!({"messages": [{"role": "user", "content": "Hello"}]});
        let goodbye = json!({"messages": [{"role": "user", "content": "Goodbye"}]});

        let req1 = deterministic_request_id(&hello, "session1");
        assert!(Uuid::parse_str(&req1).is_ok());
        assert_eq!(req1, deterministic_request_id(&hello, "session1"));
        assert_ne!(req1, deterministic_request_id(&goodbye, "session1"));
        assert_ne!(req1, deterministic_request_id(&hello, "session2"));

        // tool_result 内容变化但最终用户文本相同 -> 相同 request ID
        let with_tool = |v: &str| {
            json!({"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": v}]},
                {"role": "user", "content": "do something"}
            ]})
        };
        assert_eq!(
            deterministic_request_id(&with_tool("version_A"), "s"),
            deterministic_request_id(&with_tool("version_B"), "s")
        );

        // 无用户文本时退化为随机 UUID
        let assistant_only = json!({"messages": [{"role": "assistant", "content": "Hi"}]});
        assert_ne!(
            deterministic_request_id(&assistant_only, "s"),
            deterministic_request_id(&assistant_only, "s")
        );

        // Interaction ID：同 session 稳定、跨 session 不同、与 request ID 域隔离、空 session 返回 None
        let inter1 = deterministic_interaction_id("session1").unwrap();
        assert!(Uuid::parse_str(&inter1).is_ok());
        assert_eq!(
            Some(inter1.clone()),
            deterministic_interaction_id("session1")
        );
        assert_ne!(
            Some(inter1.clone()),
            deterministic_interaction_id("session2")
        );
        assert_ne!(inter1, req1);
        assert!(deterministic_interaction_id("").is_none());
    }

    #[test]
    fn sanitize_orphan_tool_results_rules() {
        // 匹配保留，孤立转为 text
        let mixed = sanitize_orphan_tool_results(json!({
            "messages": [
                {"role": "user", "content": "Help"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tool_1", "name": "read", "input": {}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": "ok"},
                    {"type": "tool_result", "tool_use_id": "tool_orphan", "content": "orphan"}
                ]}
            ]
        }));
        let last = mixed["messages"][2]["content"].as_array().unwrap();
        assert_eq!(last[0]["type"], "tool_result");
        assert_eq!(last[1]["type"], "text");
        assert!(last[1]["text"].as_str().unwrap().contains("tool_orphan"));

        // 无孤立时不变
        let clean = json!({
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tool_1", "name": "read", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "ok"}]}
            ]
        });
        assert_eq!(sanitize_orphan_tool_results(clean.clone()), clean);

        // tool_use 在更早的 assistant 而非紧邻 assistant -> orphan
        let non_adjacent = sanitize_orphan_tool_results(json!({
            "messages": [
                {"role": "user", "content": "step 1"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "old_tool", "name": "s", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "old_tool", "content": "found"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "thinking"}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "old_tool", "content": "stale"}]}
            ]
        }));
        assert_eq!(
            non_adjacent["messages"][2]["content"][0]["type"],
            "tool_result"
        );
        assert_eq!(non_adjacent["messages"][4]["content"][0]["type"], "text");

        // 紧邻上一条非 assistant（user -> user）-> orphan
        let prev_user = sanitize_orphan_tool_results(json!({
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "data"}]}
            ]
        }));
        assert_eq!(prev_user["messages"][1]["content"][0]["type"], "text");

        // 空或缺失 tool_use_id -> orphan
        let missing_id = sanitize_orphan_tool_results(json!({
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tool_1", "name": "read", "input": {}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "", "content": "empty"},
                    {"type": "tool_result", "content": "missing"}
                ]}
            ]
        }));
        assert_eq!(missing_id["messages"][1]["content"][0]["type"], "text");
        assert_eq!(missing_id["messages"][1]["content"][1]["type"], "text");
    }

    #[test]
    fn classify_must_run_before_sanitize_for_orphan_tool_results() {
        // 无论纯 orphan tool_result 还是 orphan tool_result + text，
        // 分类必须在 sanitize 之前执行才能正确判为 agent（sanitize 会把它们转成 text -> user）
        for (case, content) in [
            (
                "orphan_only",
                json!([
                    {"type": "tool_result", "tool_use_id": "orphan_1", "content": "a"},
                    {"type": "tool_result", "tool_use_id": "orphan_2", "content": "b"}
                ]),
            ),
            (
                "orphan_with_text",
                json!([
                    {"type": "tool_result", "tool_use_id": "orphan_1", "content": "a"},
                    {"type": "text", "text": "output"}
                ]),
            ),
        ] {
            let body = json!({
                "messages": [
                    {"role": "assistant", "content": "Processing..."},
                    {"role": "user", "content": content}
                ]
            });
            assert_eq!(
                classify_request(&body, false, false, false).initiator,
                "agent",
                "{case}: before sanitize"
            );
            let sanitized = sanitize_orphan_tool_results(body);
            assert_eq!(
                classify_request(&sanitized, false, false, false).initiator,
                "user",
                "{case}: after sanitize"
            );
        }
    }

    #[test]
    fn strip_thinking_blocks_cases() {
        // 多轮 assistant 移除 thinking / redacted_thinking，保持块顺序，保留非 thinking 块 signature，
        // 仅 thinking 的消息留空数组，不动 user 消息和字符串 content
        let result = strip_thinking_blocks(json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "thinking", "thinking": "keep user"},
                    {"type": "text", "text": "q1"}
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "pre", "signature": "sig"},
                    {"type": "text", "text": "A"},
                    {"type": "tool_use", "id": "t1", "name": "read", "input": {}, "signature": "keep_sig"},
                    {"type": "redacted_thinking", "data": "mid"},
                    {"type": "text", "text": "B"}
                ]},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "solo"}]},
                {"role": "assistant", "content": "plain text response"}
            ]
        }));
        let msgs = result["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["content"].as_array().unwrap().len(), 2);
        let a1 = msgs[1]["content"].as_array().unwrap();
        assert_eq!(a1.len(), 3);
        assert_eq!(a1[0]["text"], "A");
        assert_eq!(a1[1]["type"], "tool_use");
        assert_eq!(a1[1]["signature"], "keep_sig");
        assert_eq!(a1[2]["text"], "B");
        assert_eq!(msgs[2]["content"].as_array().unwrap().len(), 0);
        assert_eq!(msgs[3]["content"], "plain text response");

        let no_messages = json!({ "model": "claude-3-5-sonnet" });
        assert_eq!(strip_thinking_blocks(no_messages.clone()), no_messages);
    }
}
