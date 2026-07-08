//! Local tokenization logic for SillyTavern-compatible tokenizer requests.
//!
//! The crate exposes pure Rust tokenizer operations and, on wasm targets, the
//! JavaScript-callable `wasm-bindgen` boundary used by the extension.

mod error;
mod google;
mod openai;
mod router;
#[cfg(target_arch = "wasm32")]
mod wasm;

pub use google::gemma::GemmaTokenizer;
pub use openai::{OpenAiTokenizer, encode::EncodeResult};
pub use router::{Tokenizer, TokenizerProvider};
#[cfg(target_arch = "wasm32")]
pub use wasm::{
    st_dot_count_messages_json_wasm, st_dot_encode_text_wasm, st_dot_get_text_tokens_wasm,
    st_dot_get_token_count_async_wasm, st_dot_init_tokenizer_provider_wasm,
    st_dot_token_handler_count_async_wasm, start,
};

pub use error::TokenizerError;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Stable provider tag stamped onto every tokenizer result for logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderLabel {
    /// OpenAI-compatible tokenizer served by `tiktoken-rs`.
    OpenAi,
    /// Gemma/Gemini-family tokenizer served by Hugging Face `tokenizers`.
    Gemma,
}

impl ProviderLabel {
    /// Stable lowercase identifier used in JavaScript payloads and provider folders.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Gemma => "gemma",
        }
    }
}

/// Token count returned to JavaScript.
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct CountResult {
    /// Number of tokens in the request.
    pub token_count: usize,
    /// Model name the count was produced for.
    pub model_name: ModelName,
    /// Provider that produced the count.
    pub label: ProviderLabel,
}

/// Decoded text returned to JavaScript.
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct DecodeResult {
    /// Text decoded from token ids.
    pub text: String,
}

/// Model name received from the JavaScript boundary.
///
/// A newtype over the owned name. Results routinely outlive the request and carry
/// the name inside them, so a borrow would be promoted to owned on essentially
/// every path anyway — this owns up front and keeps the API lifetime-free.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(transparent)]
pub struct ModelName(String);

impl ModelName {
    /// Wraps a raw model name passed in by JavaScript.
    pub fn from_js(value: &str) -> Self {
        Self(value.to_string())
    }

    /// Returns the raw model name string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Deserialize)]
struct EncodeRequest {
    #[serde(default)]
    text: String,
}

/// A token-count request from JavaScript: the chat messages to be counted.
///
/// Holds the raw JSON array of messages so the boundary can deserialize live
/// message objects once, without a `JSON.stringify` round-trip. The entry point
/// stays free of any provider's message shape; each provider adapts this neutral
/// value into its own representation via [`Tokenizer::count`].
#[derive(Debug, Clone, Deserialize)]
#[serde(transparent)]
pub struct CountTokenRequest(serde_json::Value);

impl CountTokenRequest {
    /// Consumes the wrapper, yielding the raw JSON for a provider adapter.
    pub(crate) fn into_value(self) -> serde_json::Value {
        self.0
    }
}

/// Counts a JSON-string chat-message body locally.
///
/// `body_json` must be a JSON array of chat messages. It is parsed into the
/// neutral [`CountTokenRequest`] and routed through [`try_count_chat_messages`]:
/// models with an exact local tokenizer are counted precisely, and every other
/// model returns a structured error so JavaScript can use SillyTavern's original
/// request path.
pub fn try_count_messages(
    model: ModelName,
    body_json: &str,
) -> Result<CountResult, TokenizerError> {
    let messages: CountTokenRequest = serde_json::from_str(body_json)?;
    try_count_chat_messages(model, messages)
}

/// Counts already-parsed chat messages, routing on the model's provider.
///
/// This is the sole routing entry point: it selects the provider for `model` and
/// hands the neutral [`CountTokenRequest`] to that provider's [`Tokenizer::count`]. The
/// JavaScript boundary deserializes live message objects straight into
/// [`CountTokenRequest`], avoiding a `JSON.stringify`/`serde_json` round-trip on the
/// hot prompt-construction path.
///
/// Unknown models and provider-level failures are returned as structured errors
/// so JavaScript can use SillyTavern's native request path for that call.
pub fn try_count_chat_messages(
    model: ModelName,
    messages: CountTokenRequest,
) -> Result<CountResult, TokenizerError> {
    let Some(provider) = TokenizerProvider::from_model_name(&model) else {
        return Err(TokenizerError::Unsupported(format!(
            "model `{}` is not handled by the local counter",
            model.as_str()
        )));
    };
    provider.count(model, messages)
}

/// Encodes text with the tokenizer associated with `model`.
///
/// Returns token ids, token count, and per-token UTF-8 chunks.
pub fn try_encode_text(model: ModelName, text: &str) -> Result<EncodeResult, TokenizerError> {
    let Some(provider) = TokenizerProvider::from_model_name(&model) else {
        return Err(TokenizerError::Unsupported(format!(
            "model `{}` is not handled by the local encoder",
            model.as_str()
        )));
    };
    provider.encode(model, text)
}

/// Encodes the JSON body sent to SillyTavern's encode endpoint.
///
/// `body_json` must be an object with an optional string `text` field. Missing
/// text is treated as an empty string, matching SillyTavern's permissive route.
pub fn try_encode_request(
    model: ModelName,
    body_json: &str,
) -> Result<EncodeResult, TokenizerError> {
    let request: EncodeRequest = serde_json::from_str(body_json)?;
    try_encode_text(model, &request.text)
}

/// Initializes the tokenizer data for a provider requested by Rust.
pub fn init_tokenizer_provider(provider: &str, tokenizer_json: &str) -> Result<(), TokenizerError> {
    if provider == <GemmaTokenizer as Tokenizer>::LABEL.as_str() {
        return google::gemma::init_tokenizer(tokenizer_json);
    }

    Err(TokenizerError::Unsupported(format!(
        "unknown tokenizer provider `{provider}`"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiktoken_rs::{ChatCompletionRequestMessage, FunctionCall, num_tokens_from_messages};

    fn model(value: &str) -> ModelName {
        ModelName::from_js(value)
    }

    fn count(model: &str, body_json: &str) -> Result<usize, TokenizerError> {
        openai::chat::count(model, serde_json::from_str(body_json)?)
    }

    #[test]
    fn unsupported_chat_models_return_errors_for_js_fallback() -> Result<(), String> {
        assert!(matches!(
            try_count_messages(model("claude"), r#"[{"role":"user","content":"hi"}]"#),
            Err(TokenizerError::Unsupported(_))
        ));

        let error = try_count_messages(
            model("text-davinci-003"),
            r#"[{"role":"user","content":"hi"}]"#,
        )
        .expect_err("non-chat OpenAI models should defer to JavaScript fallback");
        assert!(matches!(error, TokenizerError::Tiktoken(_)));
        Ok(())
    }

    #[test]
    fn derives_tokenizer_provider_from_model_name() {
        assert!(matches!(
            TokenizerProvider::from_model_name(&model("gemini-2.5-pro")),
            Some(TokenizerProvider::Gemma(_))
        ));
        assert!(matches!(
            TokenizerProvider::from_model_name(&model("gpt-4o")),
            Some(TokenizerProvider::OpenAi(_))
        ));
        assert!(matches!(
            TokenizerProvider::from_model_name(&model("text-davinci-003")),
            Some(TokenizerProvider::OpenAi(_))
        ));
        assert!(TokenizerProvider::from_model_name(&model("claude")).is_none());
    }

    #[test]
    fn tokenizer_decode_is_reserved() -> Result<(), String> {
        let openai = OpenAiTokenizer::from_model_name("gpt-4o")
            .ok_or_else(|| "gpt-4o should resolve".to_string())?;
        assert!(matches!(
            openai.decode(model("gpt-4o"), &[1, 2, 3]),
            Err(TokenizerError::Unsupported(_))
        ));
        let gemma = GemmaTokenizer::from_model_name("gemini")
            .ok_or_else(|| "gemini should resolve".to_string())?;
        assert!(matches!(
            gemma.decode(model("gemini"), &[1, 2, 3]),
            Err(TokenizerError::Unsupported(_))
        ));
        Ok(())
    }

    #[test]
    fn counts_chat_message_arrays() -> Result<(), String> {
        let body = r#"[{"role":"user","content":"hello world"}]"#;
        let result =
            try_count_messages(model("gpt-4o"), body).map_err(|error| error.to_string())?;
        assert_eq!(result.label, ProviderLabel::OpenAi);
        assert!(result.token_count > 0);
        Ok(())
    }

    #[test]
    fn parsed_message_path_matches_string_path() -> Result<(), String> {
        let body = r#"[{"role":"user","content":"hello world","name":"bob"}]"#;
        let from_string =
            try_count_messages(model("gpt-4o"), body).map_err(|error| error.to_string())?;

        let messages: CountTokenRequest =
            serde_json::from_str(body).map_err(|error| error.to_string())?;
        let from_parsed = try_count_chat_messages(model("gpt-4o"), messages)
            .map_err(|error| error.to_string())?;

        assert_eq!(from_string.token_count, from_parsed.token_count);
        Ok(())
    }

    #[test]
    fn parsed_message_path_reports_unsupported_models() -> Result<(), String> {
        let messages: CountTokenRequest =
            serde_json::from_str(r#"[{"role":"user","content":"hi"}]"#)
                .map_err(|error| error.to_string())?;
        assert!(matches!(
            try_count_chat_messages(model("claude"), messages),
            Err(TokenizerError::Unsupported(_))
        ));
        Ok(())
    }

    #[test]
    fn parsed_message_path_defers_uninitialized_providers() -> Result<(), String> {
        let messages: CountTokenRequest =
            serde_json::from_str(r#"[{"role":"user","content":"hi"}]"#)
                .map_err(|error| error.to_string())?;
        let error = try_count_chat_messages(model("gemini-2.5-pro"), messages)
            .expect_err("uninitialized Gemma provider should be deferred");
        assert!(matches!(error, TokenizerError::UnInitialized { .. }));

        let body = error.body();
        assert_eq!(body.error, "UnInitialized");
        assert_eq!(body.model_name.as_str(), "gemini-2.5-pro");
        assert_eq!(body.provider, "gemma");
        Ok(())
    }

    #[test]
    fn encodes_request_bodies() -> Result<(), String> {
        let result = try_encode_request(model("gpt-4o"), r#"{"text":"hello world"}"#)
            .map_err(|error| error.to_string())?;
        assert_eq!(result.count, result.ids.len());
        Ok(())
    }

    #[test]
    fn rejects_malformed_encode_bodies() {
        assert!(try_encode_request(model("gpt-4o"), r#""hello""#).is_err());
    }

    #[test]
    fn counts_with_tiktoken_rs_chat_counter() -> Result<(), String> {
        let body = r#"[{"role":"user","content":"hello world"}]"#;
        let expected = num_tokens_from_messages(
            "gpt-4o",
            &[ChatCompletionRequestMessage {
                role: "user".to_string(),
                content: Some("hello world".to_string()),
                ..Default::default()
            }],
        )
        .map_err(|error| error.to_string())?;

        assert_eq!(
            count("gpt-4o", body).map_err(|error| error.to_string())?,
            expected
        );
        Ok(())
    }

    #[test]
    fn counts_name_with_tiktoken_rs_rules() -> Result<(), String> {
        let body = r#"[{"role":"user","content":"hi","name":"bob"}]"#;
        let expected = num_tokens_from_messages(
            "gpt-4o",
            &[ChatCompletionRequestMessage {
                role: "user".to_string(),
                content: Some("hi".to_string()),
                name: Some("bob".to_string()),
                ..Default::default()
            }],
        )
        .map_err(|error| error.to_string())?;

        assert_eq!(
            count("gpt-4o", body).map_err(|error| error.to_string())?,
            expected
        );
        Ok(())
    }

    #[test]
    fn uses_tiktoken_rs_gpt35_0301_rules() -> Result<(), String> {
        let body = r#"[{"role":"user","content":"hi","name":"bob"}]"#;
        let expected = num_tokens_from_messages(
            "gpt-3.5-turbo-0301",
            &[ChatCompletionRequestMessage {
                role: "user".to_string(),
                content: Some("hi".to_string()),
                name: Some("bob".to_string()),
                ..Default::default()
            }],
        )
        .map_err(|error| error.to_string())?;

        assert_eq!(
            count("gpt-3.5-turbo-0301", body).map_err(|error| error.to_string())?,
            expected
        );
        Ok(())
    }

    #[test]
    fn joins_multimodal_text_parts() -> Result<(), String> {
        let body = serde_json::json!([{
            "role": "user",
            "content": [
                { "type": "text", "text": "hello" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } },
                { "type": "text", "text": " world" }
            ]
        }])
        .to_string();
        let expected = num_tokens_from_messages(
            "gpt-4o",
            &[ChatCompletionRequestMessage {
                role: "user".to_string(),
                content: Some("hello world".to_string()),
                ..Default::default()
            }],
        )
        .map_err(|error| error.to_string())?;

        assert_eq!(
            count("gpt-4o", &body).map_err(|error| error.to_string())?,
            expected
        );
        Ok(())
    }

    #[test]
    fn maps_tool_calls_to_tiktoken_rs_function_calls() -> Result<(), String> {
        let arguments = r#"{"q":"hi"}"#;
        let body = serde_json::json!([{
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": arguments
                }
            }]
        }])
        .to_string();
        let expected = num_tokens_from_messages(
            "gpt-4o",
            &[ChatCompletionRequestMessage {
                role: "assistant".to_string(),
                tool_calls: vec![FunctionCall {
                    name: "search".to_string(),
                    arguments: arguments.to_string(),
                }],
                ..Default::default()
            }],
        )
        .map_err(|error| error.to_string())?;

        assert_eq!(
            count("gpt-4o", &body).map_err(|error| error.to_string())?,
            expected
        );
        Ok(())
    }

    #[test]
    fn parses_stringified_tool_calls() -> Result<(), String> {
        let tool_calls = serde_json::json!([{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": "{\"q\":\"hi\"}"
            }
        }]);
        let body = serde_json::json!([{
            "role": "assistant",
            "content": null,
            "tool_calls": tool_calls.to_string()
        }])
        .to_string();

        assert!(count("gpt-4o", &body).map_err(|error| error.to_string())? > 0);
        Ok(())
    }
}
