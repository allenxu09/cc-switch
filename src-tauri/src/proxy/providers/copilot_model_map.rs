//! GitHub Copilot 模型 ID 归一化与 live-list 解析
//!
//! Copilot upstream 仅接受 dot 形式的 Claude 4.x 模型 ID（如 `claude-sonnet-4.6`），
//! 而 Claude Code 客户端发出 dash 形式（如 `claude-sonnet-4-6`、`claude-sonnet-4-6[1m]`）。
//! 不归一化会触发上游 400 `model_not_supported`。
//!
//! 仅做语法归一化不够：账号订阅级别可能不开放某个具体模型。
//! `resolve_against_models` 用 `/models` live 列表做精确匹配，找不到时
//! 按 family（haiku/sonnet/opus）+ 最高版本号 fallback。

use super::copilot_auth::CopilotModel;
use crate::provider::CodexCopilotApiFormat;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopilotProtocol {
    Messages,
    Responses,
    Chat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopilotTransport {
    pub protocol: CopilotProtocol,
    pub endpoint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCopilotModel {
    pub id: String,
    pub vendor: String,
    pub transport: Option<CopilotTransport>,
}

/// 归一化客户端 model ID 为 Copilot upstream 接受的形式。
/// 返回 `None` 表示无需变换（已归一化、非 Claude 4.x 系列、或空输入）。
pub(super) fn normalize_to_copilot_id(client_id: &str) -> Option<String> {
    let trimmed = client_id.trim();
    let bytes = trimmed.as_bytes();

    if bytes.len() < 8 || !bytes[..7].eq_ignore_ascii_case(b"claude-") {
        return None;
    }

    let has_one_m_bracket = ends_with_ascii_ci(bytes, b"[1m]");

    // Fast path: 已含点 + 不带 [1m] → 已归一化（绝大多数请求走这里）
    if trimmed.contains('.') && !has_one_m_bracket {
        return None;
    }

    let (base, has_1m_suffix) = split_one_m_suffix(trimmed);
    let stripped = strip_trailing_date(base);
    let dotted = dashes_to_dot_in_last_version(stripped);

    if dotted.is_none() && !has_1m_suffix {
        return None;
    }

    let mut candidate = dotted.unwrap_or_else(|| stripped.to_string());
    if has_1m_suffix {
        candidate.push_str("-1m");
    }
    (candidate != trimmed).then_some(candidate)
}

/// 在请求体中应用 model ID 归一化。
pub fn apply_copilot_model_normalization(mut body: Value) -> Value {
    let Some(orig) = body.get("model").and_then(|v| v.as_str()) else {
        return body;
    };
    if let Some(normalized) = normalize_to_copilot_id(orig) {
        log::debug!("[CopilotNormalizer] {orig} → {normalized}");
        body["model"] = Value::String(normalized);
    }
    body
}

fn ends_with_ascii_ci(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.len() >= needle.len()
        && haystack[haystack.len() - needle.len()..].eq_ignore_ascii_case(needle)
}

fn split_one_m_suffix(id: &str) -> (&str, bool) {
    let bytes = id.as_bytes();
    if ends_with_ascii_ci(bytes, b"[1m]") {
        return (&id[..bytes.len() - 4], true);
    }
    if ends_with_ascii_ci(bytes, b"-1m") {
        return (&id[..bytes.len() - 3], true);
    }
    (id, false)
}

fn strip_trailing_date(id: &str) -> &str {
    let Some(last_dash) = id.rfind('-') else {
        return id;
    };
    let suffix = &id[last_dash + 1..];
    if suffix.len() == 8 && suffix.bytes().all(|b| b.is_ascii_digit()) {
        &id[..last_dash]
    } else {
        id
    }
}

/// 把 `…-X-Y`（X、Y 都是纯数字的末两段）变成 `…-X.Y`。
/// 返回 `None` 表示模式不匹配（保守策略避免误伤 `claude-3-5-sonnet` 等历史 ID）。
fn dashes_to_dot_in_last_version(id: &str) -> Option<String> {
    let last_dash = id.rfind('-')?;
    let last_segment = &id[last_dash + 1..];
    if last_segment.is_empty() || !last_segment.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let head = &id[..last_dash];
    let prev_dash = head.rfind('-')?;
    let prev_segment = &head[prev_dash + 1..];
    if prev_segment.is_empty() || !prev_segment.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(format!("{head}.{last_segment}"))
}

/// 用 Copilot live 模型列表确认/降级 model ID。
///
/// 流程：
/// 1. 先做语法归一化（dash→dot、`[1m]`→`-1m`）
/// 2. 在 `models` 中精确匹配；找到则使用归一化后的 ID
/// 3. 找不到时按 family（haiku/sonnet/opus）取最高版本号 fallback
///    （优先保留 `-1m` 标志；都没有则取 base 版）
///
/// 返回 `None` 表示无需变换或无可降级的 family 候选（保留原 ID 让上游决定，
/// 让用户拿到明确的 `model_not_supported` 而非被静默替换）。
pub fn resolve_against_models(client_id: &str, models: &[CopilotModel]) -> Option<String> {
    match_copilot_model(client_id, models, None).and_then(|matched| matched.replacement_id)
}

/// Exact-or-family match returning the full model record (vendor + endpoints).
pub fn resolve_model_record<'a>(
    client_id: &str,
    models: &'a [CopilotModel],
) -> Option<&'a CopilotModel> {
    match_copilot_model(client_id, models, None).map(|matched| matched.model)
}

pub fn resolve_model(client_id: &str, models: &[CopilotModel]) -> Option<ResolvedCopilotModel> {
    // Legacy/vendor lookups must not filter out models without Codex transports.
    let model = match_copilot_model(client_id, models, None)?.model;
    Some(ResolvedCopilotModel {
        id: model.id.clone(),
        vendor: model.vendor.clone(),
        transport: transport_for(&model.supported_endpoints, CodexCopilotApiFormat::Auto),
    })
}

/// Only family fallback is transport-filtered; exact matches retain their
/// identity even when the requested format has no supported transport.
pub fn resolve_model_with_format(
    client_id: &str,
    models: &[CopilotModel],
    api_format: CodexCopilotApiFormat,
) -> Option<ResolvedCopilotModel> {
    let model = match_copilot_model(client_id, models, Some(api_format))?.model;
    Some(ResolvedCopilotModel {
        id: model.id.clone(),
        vendor: model.vendor.clone(),
        transport: transport_for(&model.supported_endpoints, api_format),
    })
}

struct CopilotModelMatch<'a> {
    model: &'a CopilotModel,
    // Exact matches retain the caller's spelling; a resolved model uses the
    // catalog's ID. Keep that distinction separate from selecting the model.
    replacement_id: Option<String>,
}

fn match_copilot_model<'a>(
    client_id: &str,
    models: &'a [CopilotModel],
    fallback_format: Option<CodexCopilotApiFormat>,
) -> Option<CopilotModelMatch<'a>> {
    let normalized = normalize_to_copilot_id(client_id);
    let target = normalized.as_deref().unwrap_or(client_id);
    if let Some(model) = models
        .iter()
        .find(|model| model.id.eq_ignore_ascii_case(target))
    {
        return Some(CopilotModelMatch {
            model,
            replacement_id: normalized.filter(|id| id != client_id),
        });
    }

    let Some(fallback) = family_fallback(target, models, fallback_format) else {
        // 没有系列退回可用（如 Fable）时，带 1M 的请求退回同版本的普通模型：
        // Copilot 列表里多半没有 `-1m` 变体，直接发会被 400 拒掉。
        let (base, true) = split_one_m_suffix(target) else {
            return None;
        };
        let model = models
            .iter()
            .find(|model| model.id.eq_ignore_ascii_case(base))?;
        return Some(CopilotModelMatch {
            model,
            replacement_id: (!model.id.eq_ignore_ascii_case(client_id)).then(|| model.id.clone()),
        });
    };
    let model = models
        .iter()
        .find(|model| model.id.eq_ignore_ascii_case(&fallback))?;
    Some(CopilotModelMatch {
        model,
        replacement_id: (!fallback.eq_ignore_ascii_case(client_id)).then_some(fallback),
    })
}

pub fn transport_for(
    supported_endpoints: &[String],
    api_format: CodexCopilotApiFormat,
) -> Option<CopilotTransport> {
    let protocols: &[CopilotProtocol] = match api_format {
        CodexCopilotApiFormat::Auto => &[CopilotProtocol::Responses, CopilotProtocol::Chat],
        CodexCopilotApiFormat::OpenaiResponses => &[CopilotProtocol::Responses],
        CodexCopilotApiFormat::OpenaiChat => &[CopilotProtocol::Chat],
    };
    advertised_transport(supported_endpoints, protocols)
}

pub(crate) fn advertised_transport(
    supported_endpoints: &[String],
    protocols: &[CopilotProtocol],
) -> Option<CopilotTransport> {
    protocols.iter().find_map(|&protocol| {
        let paths: &[&str] = match protocol {
            CopilotProtocol::Messages => &["/v1/messages"],
            CopilotProtocol::Responses => &["/responses", "/v1/responses"],
            CopilotProtocol::Chat => &["/chat/completions", "/v1/chat/completions"],
        };
        supported_endpoints.iter().find_map(|endpoint| {
            let path = endpoint.split('?').next()?.trim_end_matches('/');
            paths
                .iter()
                .any(|p| p.eq_ignore_ascii_case(path))
                .then(|| CopilotTransport {
                    protocol,
                    endpoint: path.to_string(),
                })
        })
    })
}

fn detect_family(id: &str) -> Option<&'static str> {
    let lower = id.to_ascii_lowercase();
    if lower.contains("haiku") {
        Some("haiku")
    } else if lower.contains("sonnet") {
        Some("sonnet")
    } else if lower.contains("opus") {
        Some("opus")
    } else {
        None
    }
}

/// 提取 family 后第一段 `MAJOR.MINOR` 版本号。
/// 例：`claude-sonnet-4.6` → (4, 6)；`claude-sonnet-4.6-1m` → (4, 6)。
fn extract_major_minor(id: &str) -> Option<(u32, u32)> {
    let lower = id.to_ascii_lowercase();
    let family = detect_family(&lower)?;
    let after = &lower[lower.find(family)? + family.len()..];
    let after = after.strip_prefix('-')?;
    let segment = after.split(['-', '[', ' ']).next()?;
    let mut parts = segment.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}

fn family_fallback(
    target: &str,
    models: &[CopilotModel],
    api_format: Option<CodexCopilotApiFormat>,
) -> Option<String> {
    let family = detect_family(target)?;
    let want_1m = target.ends_with("-1m");

    let pick_best = |require_1m: bool| -> Option<String> {
        models
            .iter()
            .filter(|m| {
                let lower = m.id.to_ascii_lowercase();
                lower.contains(family) && lower.ends_with("-1m") == require_1m
            })
            .filter(|m| {
                api_format.is_none_or(|format| {
                    // Duplicate IDs resolve to their first catalog metadata.
                    models
                        .iter()
                        .find(|model| model.id.eq_ignore_ascii_case(&m.id))
                        .is_some_and(|model| {
                            transport_for(&model.supported_endpoints, format).is_some()
                        })
                })
            })
            .filter_map(|m| extract_major_minor(&m.id).map(|v| (m, v)))
            .max_by_key(|(_, v)| *v)
            .map(|(m, _)| m.id.clone())
    };

    if want_1m {
        pick_best(true).or_else(|| pick_best(false))
    } else {
        pick_best(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn model(id: &str) -> CopilotModel {
        CopilotModel {
            id: id.to_string(),
            name: id.to_string(),
            vendor: "anthropic".to_string(),
            model_picker_enabled: true,
            context_window: None,
            supported_endpoints: Vec::new(),
            supports_parallel_tool_calls: None,
            supports_vision: None,
            reasoning_effort: None,
            billing: None,
        }
    }

    fn model_with_endpoints(id: &str, endpoints: &[&str]) -> CopilotModel {
        let mut m = model(id);
        m.supported_endpoints = endpoints.iter().map(|e| (*e).to_string()).collect();
        m
    }

    #[test]
    fn normalize_to_copilot_id_and_apply_body_cases() {
        for (input, expected) in [
            ("claude-sonnet-4-6", Some("claude-sonnet-4.6")),
            ("claude-opus-4-6", Some("claude-opus-4.6")),
            ("claude-haiku-4-5", Some("claude-haiku-4.5")),
            ("claude-sonnet-4-6[1m]", Some("claude-sonnet-4.6-1m")),
            ("claude-opus-4-6[1m]", Some("claude-opus-4.6-1m")),
            ("claude-haiku-4-5-20251001", Some("claude-haiku-4.5")),
            ("claude-sonnet-4-5-20250929", Some("claude-sonnet-4.5")),
            ("Claude-Sonnet-4-6", Some("Claude-Sonnet-4.6")),
            ("claude-sonnet-4-6[1M]", Some("claude-sonnet-4.6-1m")),
            ("claude-sonnet-4.6[1m]", Some("claude-sonnet-4.6-1m")),
            ("claude-haiku-4-5-20251001[1m]", Some("claude-haiku-4.5-1m")),
            ("claude-sonnet-4.6", None),
            ("claude-opus-4.6-1m", None),
            ("claude-haiku-4.5", None),
            ("gpt-5", None),
            ("gpt-4o-mini", None),
            ("o3", None),
            ("", None),
            ("claude-3-5-sonnet", None),
            ("claude-3-5-sonnet-20241022", None),
        ] {
            assert_eq!(
                normalize_to_copilot_id(input).as_deref(),
                expected,
                "{input}"
            );
        }

        let rewritten = apply_copilot_model_normalization(
            json!({"model": "claude-sonnet-4-6", "max_tokens": 1024}),
        );
        assert_eq!(rewritten["model"], "claude-sonnet-4.6");
        assert_eq!(rewritten["max_tokens"], 1024);
        assert_eq!(
            apply_copilot_model_normalization(json!({"model": "claude-sonnet-4.6"}))["model"],
            "claude-sonnet-4.6"
        );
        assert!(apply_copilot_model_normalization(json!({"messages": []}))
            .get("model")
            .is_none());
    }

    #[test]
    fn resolve_against_models_and_catalog_semantics() {
        let models = [
            model("claude-sonnet-4.5"),
            model("claude-sonnet-4.6"),
            model("claude-sonnet-4.6-1m"),
            model("claude-opus-4.5"),
            model("claude-opus-4.6"),
            model("claude-haiku-4.5"),
            model("gpt-5.6"),
        ];
        for (requested, replacement, catalog_id) in [
            ("claude-sonnet-4.6", None, Some("claude-sonnet-4.6")),
            ("CLAUDE-SONNET-4.6", None, Some("claude-sonnet-4.6")),
            (
                "CLAUDE-SONNET-4-6",
                Some("CLAUDE-SONNET-4.6"),
                Some("claude-sonnet-4.6"),
            ),
            (
                "claude-sonnet-4-6",
                Some("claude-sonnet-4.6"),
                Some("claude-sonnet-4.6"),
            ),
            (
                "claude-sonnet-4.8",
                Some("claude-sonnet-4.6"),
                Some("claude-sonnet-4.6"),
            ),
            (
                "claude-opus-4.8",
                Some("claude-opus-4.6"),
                Some("claude-opus-4.6"),
            ),
            (
                "claude-sonnet-4-6[1M]",
                Some("claude-sonnet-4.6-1m"),
                Some("claude-sonnet-4.6-1m"),
            ),
            ("gpt-5.6", None, Some("gpt-5.6")),
            ("gpt-5", None, None),
            ("unknown", None, None),
        ] {
            assert_eq!(
                resolve_against_models(requested, &models).as_deref(),
                replacement,
                "{requested}"
            );
            assert_eq!(
                resolve_model(requested, &models)
                    .as_ref()
                    .map(|resolved| resolved.id.as_str()),
                catalog_id,
                "{requested}"
            );
        }

        // 账号无 -1m 变体时降级到 base；完全没有该 family 时返回 None 让上游报错
        let sonnet_only = [model("claude-sonnet-4.6"), model("claude-haiku-4.5")];
        assert_eq!(
            resolve_against_models("claude-sonnet-4-6[1m]", &sonnet_only).as_deref(),
            Some("claude-sonnet-4.6")
        );
        assert_eq!(
            resolve_against_models("claude-opus-4.6", &sonnet_only),
            None
        );

        // 重复 ID 回退保留首条目录元数据
        let first = model("claude-sonnet-4.6");
        let mut second = model("CLAUDE-SONNET-4.6");
        second.vendor = "second".to_string();
        let dup_models = [first, second];
        assert_eq!(
            resolve_against_models("claude-sonnet-4.8", &dup_models).as_deref(),
            Some("CLAUDE-SONNET-4.6")
        );
        let resolved = resolve_model("claude-sonnet-4.8", &dup_models).unwrap();
        assert_eq!(resolved.id, "claude-sonnet-4.6");
        assert_eq!(resolved.vendor, "anthropic");
    }

    #[test]
    fn one_m_without_family_falls_back_to_same_version_base() {
        // Fable 不在系列退回里；Copilot 只列普通 id，带 1M 时退回同版本，不升到 5.2。
        let models = [
            model("claude-fable-5.1"),
            model("claude-fable-5.2"),
            model("gpt-6-astra"),
        ];
        for client in [
            "claude-fable-5-1[1M]",
            "claude-fable-5.1[1m]",
            "claude-fable-5.1-1M",
        ] {
            assert_eq!(
                resolve_against_models(client, &models),
                Some("claude-fable-5.1".to_string()),
                "{client}"
            );
            assert_eq!(
                resolve_model(client, &models).unwrap().id,
                "claude-fable-5.1"
            );
        }
        // 列表里有 1M 变体时照用；同版本也没有时照旧找不到。
        let with_variant = [model("claude-fable-5.1"), model("claude-fable-5.1-1m")];
        assert_eq!(
            resolve_against_models("claude-fable-5-1[1M]", &with_variant),
            Some("claude-fable-5.1-1m".to_string())
        );
        assert_eq!(
            resolve_against_models("claude-fable-5-3[1M]", &models),
            None
        );
        assert_eq!(
            resolve_against_models("gpt-6-astra-1m", &models),
            Some("gpt-6-astra".to_string())
        );
    }

    #[test]
    fn transport_selection_and_unsupported_endpoint_metadata() {
        for (endpoint, protocol) in [
            ("/responses", CopilotProtocol::Responses),
            ("/v1/responses", CopilotProtocol::Responses),
            ("/chat/completions", CopilotProtocol::Chat),
            ("/v1/chat/completions", CopilotProtocol::Chat),
        ] {
            let advertised = model_with_endpoints("gpt-5.6", &[endpoint]);
            assert_eq!(
                resolve_model("gpt-5.6", &[advertised]).unwrap().transport,
                Some(CopilotTransport {
                    protocol,
                    endpoint: endpoint.to_string(),
                })
            );
        }

        // 同时支持时优先 Responses，否则选 Chat
        let mut both = model_with_endpoints("gpt-5.6", &["/chat/completions", "/responses"]);
        both.vendor = "OpenAI".to_string();
        assert_eq!(
            resolve_model("gpt-5.6", &[both]).unwrap().transport,
            Some(CopilotTransport {
                protocol: CopilotProtocol::Responses,
                endpoint: "/responses".to_string(),
            })
        );

        // 仅 /v1/messages 或无端点元数据 -> Codex bridge 不支持（transport=None）
        for candidate in [
            model_with_endpoints("claude-opus-4.6", &["/v1/messages"]),
            model("o3"),
            model("claude-sonnet-4.6"),
        ] {
            let id = candidate.id.clone();
            assert_eq!(
                resolve_model(&id, &[candidate]).unwrap().transport,
                None,
                "{id}"
            );
        }
    }

    #[test]
    fn protocol_family_fallback_filters_by_format_and_preserves_legacy() {
        // 显式格式跳过不兼容最新版，选最高兼容版本
        for (format, protocol, endpoint, incompatible_endpoint) in [
            (
                CodexCopilotApiFormat::OpenaiChat,
                CopilotProtocol::Chat,
                "/v1/chat/completions",
                "/responses",
            ),
            (
                CodexCopilotApiFormat::OpenaiResponses,
                CopilotProtocol::Responses,
                "/v1/responses",
                "/chat/completions",
            ),
        ] {
            let models = [
                model_with_endpoints("claude-opus-4.8", &[incompatible_endpoint]),
                model_with_endpoints("claude-opus-4.7", &[endpoint]),
                model_with_endpoints("claude-opus-4.6", &[endpoint]),
            ];
            let resolved = resolve_model_with_format("claude-opus-5.0", &models, format).unwrap();
            assert_eq!(resolved.id, "claude-opus-4.7", "{format:?}");
            assert_eq!(
                resolved.transport,
                Some(CopilotTransport {
                    protocol,
                    endpoint: endpoint.to_string(),
                })
            );
        }

        // Auto 跳过 /v1/messages 与空端点模型，且不影响 legacy 解析路径
        let models = [
            model_with_endpoints("claude-opus-4.8", &["/v1/messages"]),
            model("claude-opus-4.7"),
            model_with_endpoints("claude-opus-4.6", &["/chat/completions"]),
            model_with_endpoints("claude-opus-4.5", &["/responses"]),
        ];
        let resolved =
            resolve_model_with_format("claude-opus-5.0", &models, CodexCopilotApiFormat::Auto)
                .unwrap();
        assert_eq!(resolved.id, "claude-opus-4.6");
        assert_eq!(
            resolved.transport,
            Some(CopilotTransport {
                protocol: CopilotProtocol::Chat,
                endpoint: "/chat/completions".to_string(),
            })
        );
        assert_eq!(
            resolve_against_models("claude-opus-5.0", &models).as_deref(),
            Some("claude-opus-4.8")
        );
        let legacy = resolve_model("claude-opus-5.0", &models).unwrap();
        assert_eq!(legacy.id, "claude-opus-4.8");
        assert_eq!(legacy.vendor, "anthropic");
        assert_eq!(legacy.transport, None);

        // 无兼容候选或空目录返回 None
        for (format, incompatible_endpoint) in [
            (CodexCopilotApiFormat::Auto, "/v1/messages"),
            (CodexCopilotApiFormat::OpenaiChat, "/responses"),
            (CodexCopilotApiFormat::OpenaiResponses, "/chat/completions"),
        ] {
            let mut gpt = model_with_endpoints("gpt-6-astra", &["/responses", "/chat/completions"]);
            gpt.vendor = "OpenAI".to_string();
            let catalog = [
                model_with_endpoints("claude-opus-4.8", &[incompatible_endpoint]),
                model("claude-opus-4.7"),
                model_with_endpoints("claude-sonnet-4.6", &["/responses", "/chat/completions"]),
                gpt,
            ];
            for requested in [
                "claude-opus-5.0",
                "claude-haiku-4.5",
                "gpt-5.6-sol",
                "unknown",
                "",
            ] {
                assert_eq!(
                    resolve_model_with_format(requested, &catalog, format),
                    None,
                    "{requested}: {format:?}"
                );
            }
            assert_eq!(
                resolve_model_with_format("claude-opus-4.8", &[], format),
                None
            );
        }
    }

    #[test]
    fn exact_unsupported_models_and_duplicate_metadata_precedence() {
        for (id, vendor, requests) in [
            (
                "claude-opus-4.7",
                "anthropic",
                vec!["claude-opus-4.7", "CLAUDE-OPUS-4.7", "CLAUDE-OPUS-4-7"],
            ),
            (
                "claude-opus-4.7-1m",
                "anthropic",
                vec![
                    "claude-opus-4.7-1m",
                    "CLAUDE-OPUS-4.7-1M",
                    "claude-opus-4-7[1M]",
                ],
            ),
            ("gpt-6-astra", "OpenAI", vec!["gpt-6-astra", "GPT-6-ASTRA"]),
        ] {
            for (format, unsupported) in [
                (CodexCopilotApiFormat::Auto, vec!["/v1/messages"]),
                (CodexCopilotApiFormat::Auto, vec![]),
                (CodexCopilotApiFormat::OpenaiChat, vec!["/responses"]),
                (
                    CodexCopilotApiFormat::OpenaiResponses,
                    vec!["/chat/completions"],
                ),
            ] {
                let mut duplicate = model_with_endpoints(
                    &id.to_ascii_uppercase(),
                    &["/responses", "/chat/completions"],
                );
                duplicate.vendor = "second".to_string();
                let mut models = [
                    model_with_endpoints(id, &unsupported),
                    duplicate,
                    model_with_endpoints("claude-opus-4.6", &["/responses", "/chat/completions"]),
                ];
                models[0].vendor = vendor.to_string();
                for requested in &requests {
                    let resolved = resolve_model_with_format(requested, &models, format).unwrap();
                    assert_eq!(resolved.id, id, "{requested}: {format:?}");
                    assert_eq!(resolved.vendor, vendor);
                    assert_eq!(resolved.transport, None);
                }
            }
        }

        // 重复 ID 的协议资格以目录首条元数据为准
        for (first_endpoint, duplicate_endpoint, expected_id) in [
            ("/v1/messages", "/chat/completions", "claude-opus-4.6"),
            ("/chat/completions", "/v1/messages", "claude-opus-4.7"),
        ] {
            let first = model_with_endpoints("claude-opus-4.7", &[first_endpoint]);
            let mut duplicate = model_with_endpoints("CLAUDE-OPUS-4.7", &[duplicate_endpoint]);
            duplicate.vendor = "second".to_string();
            let older = model_with_endpoints("claude-opus-4.6", &["/chat/completions"]);
            let resolved = resolve_model_with_format(
                "claude-opus-4.8",
                &[first, duplicate, older],
                CodexCopilotApiFormat::OpenaiChat,
            )
            .unwrap();
            assert_eq!(resolved.id, expected_id, "{first_endpoint}");
            assert_eq!(resolved.vendor, "anthropic");
            assert_eq!(
                resolved.transport,
                Some(CopilotTransport {
                    protocol: CopilotProtocol::Chat,
                    endpoint: "/chat/completions".to_string(),
                })
            );
        }
    }

    #[test]
    fn one_m_and_large_context_family_fallback_rules() {
        let endpoints = ["/responses", "/chat/completions"];
        let formats = [
            CodexCopilotApiFormat::Auto,
            CodexCopilotApiFormat::OpenaiChat,
            CodexCopilotApiFormat::OpenaiResponses,
        ];

        // 1M 请求：优先兼容的显式 -1m，缺失或不兼容时回退到自然大上下文 base
        for format in formats {
            let mut natural = model_with_endpoints("claude-opus-4.7", &endpoints);
            natural.context_window = Some(1_048_576);
            let mut models = [
                model_with_endpoints("claude-opus-4.8-1m", &["/v1/messages"]),
                model_with_endpoints("claude-opus-4.6-1m", &endpoints),
                natural,
            ];
            assert_eq!(
                resolve_model_with_format("claude-opus-5-0[1M]", &models, format)
                    .unwrap()
                    .id,
                "claude-opus-4.6-1m",
                "{format:?}"
            );
            models[1].supported_endpoints.clear();
            assert_eq!(
                resolve_model_with_format("claude-opus-5-0[1M]", &models, format)
                    .unwrap()
                    .id,
                "claude-opus-4.7",
                "{format:?}"
            );
        }

        // 普通请求保留自然大上下文 base（不跳到显式 -1m）；带 [1m] 则选 -1m 或仅有的自然 1M base
        let mut natural = model_with_endpoints("claude-opus-4.7", &endpoints);
        natural.context_window = Some(1_048_576);
        let mut explicit = model_with_endpoints("claude-opus-4.8-1m", &endpoints);
        explicit.context_window = Some(1_048_576);
        let models = [natural.clone(), explicit];

        assert_eq!(
            resolve_against_models("claude-opus-4-5", &models).as_deref(),
            Some("claude-opus-4.7")
        );
        assert_eq!(
            resolve_model("claude-opus-4-5", &models).unwrap().id,
            "claude-opus-4.7"
        );
        assert_eq!(
            resolve_model("claude-opus-4-5[1m]", &models).unwrap().id,
            "claude-opus-4.8-1m"
        );
        assert_eq!(
            resolve_model("claude-opus-4-5[1M]", std::slice::from_ref(&natural))
                .unwrap()
                .id,
            "claude-opus-4.7"
        );
        for format in formats {
            for (requested, expected) in [
                ("claude-opus-4-5", "claude-opus-4.7"),
                ("claude-opus-4.7", "claude-opus-4.7"),
                ("claude-opus-4-5[1m]", "claude-opus-4.8-1m"),
            ] {
                assert_eq!(
                    resolve_model_with_format(requested, &models, format)
                        .unwrap()
                        .id,
                    expected,
                    "{requested}: {format:?}"
                );
            }
            assert_eq!(
                resolve_model_with_format(
                    "claude-opus-4-5[1M]",
                    std::slice::from_ref(&natural),
                    format
                )
                .unwrap()
                .id,
                "claude-opus-4.7",
                "{format:?}"
            );
        }
    }

    #[test]
    fn copilot_endpoint_contract_matches_frontend_cases() {
        #[derive(serde::Deserialize)]
        struct Case {
            name: String,
            endpoints: Vec<String>,
            formats: Vec<CodexCopilotApiFormat>,
        }
        let cases: Vec<Case> = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/copilot-endpoint-cases.json"
        ))
        .unwrap();
        for case in cases {
            let mut candidate = model("gpt-contract");
            candidate.supported_endpoints = case.endpoints;
            for format in [
                CodexCopilotApiFormat::Auto,
                CodexCopilotApiFormat::OpenaiResponses,
                CodexCopilotApiFormat::OpenaiChat,
            ] {
                let resolved = resolve_model_with_format(
                    "gpt-contract",
                    std::slice::from_ref(&candidate),
                    format,
                )
                .unwrap();
                assert_eq!(
                    resolved.transport.is_some(),
                    case.formats.contains(&format),
                    "{}: {format:?}",
                    case.name
                );
            }
        }
    }
}
