//! OpenAI chat-message adapter for `tiktoken-rs`.

use serde::Deserialize;
use tiktoken_rs::{ChatCompletionRequestMessage, FunctionCall, num_tokens_from_messages};

use crate::{CountTokenRequest, TokenizerError};

/// Counts provider-agnostic messages by adapting them into OpenAI messages.
///
/// This is where the OpenAI provider owns its own conversion: the neutral
/// [`CountTokenRequest`] value is deserialized into the OpenAI request shape here,
/// so the crate-level entry point stays free of `tiktoken-rs`-specific types.
pub(crate) fn count(model: &str, messages: CountTokenRequest) -> Result<usize, TokenizerError> {
    let messages = serde_json::from_value::<Vec<RawChatMessage>>(messages.into_value())?
        .into_iter()
        .map(ChatCompletionRequestMessage::try_from)
        .collect::<Result<Vec<_>, _>>()?;

    num_tokens_from_messages(model, &messages)
        .map_err(|error| TokenizerError::Tiktoken(error.to_string()))
}

#[derive(Deserialize)]
struct RawChatMessage {
    role: String,
    #[serde(default)]
    content: Option<RawContent>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    function_call: Option<RawFunctionCall>,
    #[serde(default)]
    tool_calls: Option<RawToolCalls>,
    #[serde(default)]
    refusal: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawContent {
    Text(String),
    Parts(Vec<RawContentPart>),
}

#[derive(Deserialize)]
struct RawContentPart {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    refusal: Option<String>,
}

#[derive(Deserialize)]
struct RawFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawToolCalls {
    Calls(Vec<RawToolCall>),
    Json(String),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawToolCall {
    Wrapped { function: RawFunctionCall },
    Direct(RawFunctionCall),
}

impl RawContent {
    fn into_text_and_refusal(self) -> (Option<String>, Option<String>) {
        match self {
            Self::Text(text) => (Some(text), None),
            Self::Parts(parts) => {
                let mut text = String::new();
                let mut refusal = String::new();

                for part in parts {
                    if let Some(part_text) = part.text {
                        text.push_str(&part_text);
                    }
                    if let Some(part_refusal) = part.refusal {
                        refusal.push_str(&part_refusal);
                    }
                }

                (
                    Some(text).filter(|text| !text.is_empty()),
                    Some(refusal).filter(|refusal| !refusal.is_empty()),
                )
            }
        }
    }
}

impl From<RawFunctionCall> for FunctionCall {
    fn from(raw: RawFunctionCall) -> Self {
        Self {
            name: raw.name,
            arguments: raw.arguments,
        }
    }
}

impl From<RawToolCall> for FunctionCall {
    fn from(raw: RawToolCall) -> Self {
        match raw {
            RawToolCall::Wrapped { function } | RawToolCall::Direct(function) => function.into(),
        }
    }
}

impl RawToolCalls {
    fn into_function_calls(self) -> Result<Vec<FunctionCall>, TokenizerError> {
        match self {
            Self::Calls(calls) => Ok(calls.into_iter().map(FunctionCall::from).collect()),
            Self::Json(json) => serde_json::from_str::<Vec<RawToolCall>>(&json)
                .map_err(|error| {
                    TokenizerError::InvalidMessage(format!(
                        "message.tool_calls string must contain valid JSON: {error}"
                    ))
                })
                .map(|calls| calls.into_iter().map(FunctionCall::from).collect()),
        }
    }
}

impl TryFrom<RawChatMessage> for ChatCompletionRequestMessage {
    type Error = TokenizerError;

    fn try_from(raw: RawChatMessage) -> Result<Self, Self::Error> {
        let (content, content_refusal) = raw
            .content
            .map(RawContent::into_text_and_refusal)
            .unwrap_or_default();

        Ok(Self {
            role: raw.role,
            content,
            name: raw.name,
            function_call: raw.function_call.map(FunctionCall::from),
            tool_calls: raw
                .tool_calls
                .map(RawToolCalls::into_function_calls)
                .transpose()?
                .unwrap_or_default(),
            refusal: raw.refusal.or(content_refusal),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn messages(json: &str) -> CountTokenRequest {
        serde_json::from_str(json).expect("test body should be valid JSON")
    }

    #[test]
    fn invalid_message_shapes_are_an_error() {
        assert!(count("gpt-4o", messages(r#"{"role":"user"}"#)).is_err());
        assert!(count("gpt-4o", messages(r#"[{"content":"hi"}]"#)).is_err());
    }
}
