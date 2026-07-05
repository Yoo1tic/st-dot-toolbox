//! OpenAI chat-message adapter for `tiktoken-rs`.

use std::borrow::Cow;

use serde::Deserialize;
use tiktoken_rs::{ChatCompletionRequestMessage, FunctionCall, num_tokens_from_messages};

use crate::TokenizerError;

pub(crate) fn count(model: &str, body_json: &str) -> Result<usize, TokenizerError> {
    let raw_messages: Vec<RawChatMessage<'_>> = serde_json::from_str(body_json)?;
    let messages = raw_messages
        .into_iter()
        .map(ChatCompletionRequestMessage::try_from)
        .collect::<Result<Vec<_>, _>>()?;

    num_tokens_from_messages(model, &messages)
        .map_err(|error| TokenizerError::Tiktoken(error.to_string()))
}

#[derive(Deserialize)]
struct RawChatMessage<'a> {
    #[serde(borrow)]
    role: Cow<'a, str>,
    #[serde(default, borrow)]
    content: Option<RawContent<'a>>,
    #[serde(default, borrow)]
    name: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    function_call: Option<RawFunctionCall<'a>>,
    #[serde(default, borrow)]
    tool_calls: Option<RawToolCalls<'a>>,
    #[serde(default, borrow)]
    refusal: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawContent<'a> {
    Text(#[serde(borrow)] Cow<'a, str>),
    Parts(Vec<RawContentPart<'a>>),
}

#[derive(Deserialize)]
struct RawContentPart<'a> {
    #[serde(default, borrow)]
    text: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    refusal: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct RawFunctionCall<'a> {
    #[serde(borrow)]
    name: Cow<'a, str>,
    #[serde(borrow)]
    arguments: Cow<'a, str>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawToolCalls<'a> {
    Calls(Vec<RawToolCall<'a>>),
    Json(#[serde(borrow)] Cow<'a, str>),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawToolCall<'a> {
    Wrapped {
        #[serde(borrow)]
        function: RawFunctionCall<'a>,
    },
    Direct(RawFunctionCall<'a>),
}

impl RawContent<'_> {
    fn into_text_and_refusal(self) -> (Option<String>, Option<String>) {
        match self {
            Self::Text(text) => (Some(text.into_owned()), None),
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

impl From<RawFunctionCall<'_>> for FunctionCall {
    fn from(raw: RawFunctionCall<'_>) -> Self {
        Self {
            name: raw.name.into_owned(),
            arguments: raw.arguments.into_owned(),
        }
    }
}

impl From<RawToolCall<'_>> for FunctionCall {
    fn from(raw: RawToolCall<'_>) -> Self {
        match raw {
            RawToolCall::Wrapped { function } | RawToolCall::Direct(function) => function.into(),
        }
    }
}

impl RawToolCalls<'_> {
    fn into_function_calls(self) -> Result<Vec<FunctionCall>, TokenizerError> {
        match self {
            Self::Calls(calls) => Ok(calls.into_iter().map(FunctionCall::from).collect()),
            Self::Json(json) => serde_json::from_str::<Vec<RawToolCall<'_>>>(&json)
                .map_err(|error| {
                    TokenizerError::InvalidMessage(format!(
                        "message.tool_calls string must contain valid JSON: {error}"
                    ))
                })
                .map(|calls| calls.into_iter().map(FunctionCall::from).collect()),
        }
    }
}

impl TryFrom<RawChatMessage<'_>> for ChatCompletionRequestMessage {
    type Error = TokenizerError;

    fn try_from(raw: RawChatMessage<'_>) -> Result<Self, Self::Error> {
        let (content, content_refusal) = raw
            .content
            .map(RawContent::into_text_and_refusal)
            .unwrap_or_default();

        Ok(Self {
            role: raw.role.into_owned(),
            content,
            name: raw.name.map(Cow::into_owned),
            function_call: raw.function_call.map(FunctionCall::from),
            tool_calls: raw
                .tool_calls
                .map(RawToolCalls::into_function_calls)
                .transpose()?
                .unwrap_or_default(),
            refusal: raw.refusal.map(Cow::into_owned).or(content_refusal),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_body_is_an_error() {
        assert!(count("gpt-4o", "not json").is_err());
        assert!(count("gpt-4o", r#"{"role":"user"}"#).is_err());
        assert!(count("gpt-4o", r#"[{"content":"hi"}]"#).is_err());
    }
}
