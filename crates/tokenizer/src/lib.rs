//! Local tokenization logic for SillyTavern-compatible tokenizer requests.
//!
//! The crate deliberately has no `wasm-bindgen` dependency. It exposes pure
//! Rust operations that the root WASM facade can adapt for JavaScript.

mod error;
mod fallback;
mod google;
mod openai;
mod router;

pub use fallback::FallbackTokenizer;
pub use google::gemma::GemmaTokenizer;
pub use openai::{OpenAiTokenizer, encode::EncodeResult};
pub use router::{Tokenizer, TokenizerAsset, TokenizerProvider, TokenizerProviderForModel};

pub use error::TokenizerError;
use serde::{Deserialize, Serialize};

/// Stable provider tag stamped onto every tokenizer result for logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderLabel {
    /// OpenAI-compatible tokenizer served by `tiktoken-rs`.
    OpenAi,
    /// Gemma/Gemini-family tokenizer served by Hugging Face `tokenizers`.
    Gemma,
    /// Heuristic character-ratio estimate for models with no exact tokenizer.
    Fallback,
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
#[derive(Debug, Deserialize)]
#[serde(transparent)]
pub struct CountTokenRequest(serde_json::Value);

impl CountTokenRequest {
    /// Consumes the wrapper, yielding the raw JSON for a provider adapter.
    pub(crate) fn into_value(self) -> serde_json::Value {
        self.0
    }
}

/// Counts a JSON-string chat-message body, always returning a local count.
///
/// `body_json` must be a JSON array of chat messages. It is parsed into the
/// neutral [`CountTokenRequest`] and routed through [`try_count_chat_messages`]:
/// models with an exact tokenizer are counted precisely, and every other model
/// falls back to the [`FallbackTokenizer`] heuristic, so callers no longer need
/// SillyTavern's original tokenizer path for counting.
pub fn try_count_messages(
    model: ModelName,
    body_json: &str,
) -> Result<CountResult, TokenizerError> {
    let messages: CountTokenRequest = serde_json::from_str(body_json)?;
    match try_count_chat_messages(model.clone(), messages)? {
        Some(result) => Ok(result),
        None => Ok(FallbackTokenizer.estimate(model, body_json)),
    }
}

/// Counts already-parsed chat messages, routing on the model's provider.
///
/// This is the sole routing entry point: it selects the provider for `model` and
/// hands the neutral [`CountTokenRequest`] to that provider's [`Tokenizer::count`]. The
/// JavaScript boundary deserializes live message objects straight into
/// [`CountTokenRequest`], avoiding a `JSON.stringify`/`serde_json` round-trip on the
/// hot prompt-construction path.
///
/// Yields `Ok(None)` when the model has no local provider, or when the selected
/// provider recognizes the model but cannot count it right now — its asset has not
/// loaded, or it has no exact chat tokenizer ([`TokenizerError::is_fallback`]) — so
/// callers can keep SillyTavern's fallback path.
pub fn try_count_chat_messages(
    model: ModelName,
    messages: CountTokenRequest,
) -> Result<Option<CountResult>, TokenizerError> {
    let Some(provider) = model.tokenizer_provider() else {
        return Ok(None);
    };
    match provider.count(model, messages) {
        Ok(result) => Ok(Some(result)),
        Err(error) if error.is_fallback() => Ok(None),
        Err(error) => Err(error),
    }
}

/// Encodes text with the tokenizer associated with `model`.
///
/// Returns token ids, token count, and per-token UTF-8 chunks. `Ok(None)` means
/// there is no usable local tokenizer — the model is unknown, or its asset has not
/// loaded ([`TokenizerError::is_fallback`]) — and callers should fall back to their
/// original tokenizer path.
pub fn try_encode_text(
    model: ModelName,
    text: &str,
) -> Result<Option<EncodeResult>, TokenizerError> {
    let Some(provider) = model.tokenizer_provider() else {
        return Ok(None);
    };
    match provider.encode(model, text) {
        Ok(result) => Ok(result),
        Err(error) if error.is_fallback() => Ok(None),
        Err(error) => Err(error),
    }
}

/// Encodes the JSON body sent to SillyTavern's encode endpoint.
///
/// `body_json` must be an object with an optional string `text` field. Missing
/// text is treated as an empty string, matching SillyTavern's permissive route.
pub fn try_encode_request(
    model: ModelName,
    body_json: &str,
) -> Result<Option<EncodeResult>, TokenizerError> {
    let request: EncodeRequest = serde_json::from_str(body_json)?;
    try_encode_text(model, &request.text)
}

/// Returns the tokenizer asset a `model` needs to be served locally, if any.
///
/// A pure predicate on the model name: it names *which* asset the model requires,
/// independent of load state. Loading each asset at most once is the caller's
/// concern — the JavaScript loader already dedupes by asset id — so this stays a
/// stateless model-to-asset map.
pub fn required_tokenizer_asset(model: ModelName) -> Option<TokenizerAsset> {
    GemmaTokenizer::supports_model(model.as_str()).then_some(TokenizerAsset::Gemma)
}

/// Initializes a tokenizer asset previously requested by Rust.
pub fn init_tokenizer_asset(asset_id: &str, tokenizer_json: &str) -> Result<(), TokenizerError> {
    TokenizerAsset::from_id(asset_id)?.init(tokenizer_json)
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
    fn estimates_unsupported_chat_models_with_fallback() -> Result<(), String> {
        for model_name in ["claude", "text-davinci-003"] {
            let result =
                try_count_messages(model(model_name), r#"[{"role":"user","content":"hi"}]"#)
                    .map_err(|error| error.to_string())?;
            assert_eq!(result.label, ProviderLabel::Fallback);
            assert!(result.token_count > 0);
        }
        Ok(())
    }

    #[test]
    fn derives_tokenizer_provider_from_model_name() {
        assert!(matches!(
            model("gemini-2.5-pro").tokenizer_provider(),
            Some(TokenizerProvider::Gemma(_))
        ));
        assert!(matches!(
            model("gpt-4o").tokenizer_provider(),
            Some(TokenizerProvider::OpenAi(_))
        ));
        assert!(matches!(
            model("text-davinci-003").tokenizer_provider(),
            Some(TokenizerProvider::OpenAi(_))
        ));
        assert!(model("claude").tokenizer_provider().is_none());
    }

    #[test]
    fn tokenizer_decode_is_reserved() -> Result<(), String> {
        let openai = OpenAiTokenizer::from_model_name("gpt-4o")
            .ok_or_else(|| "gpt-4o should resolve".to_string())?;
        assert_eq!(
            openai
                .decode(model("gpt-4o"), &[1, 2, 3])
                .map_err(|error| error.to_string())?,
            None
        );
        let gemma = GemmaTokenizer::from_model_name("gemini")
            .ok_or_else(|| "gemini should resolve".to_string())?;
        assert_eq!(
            gemma
                .decode(model("gemini"), &[1, 2, 3])
                .map_err(|error| error.to_string())?,
            None
        );
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
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "parsed path should count".to_string())?;

        assert_eq!(from_string.token_count, from_parsed.token_count);
        Ok(())
    }

    #[test]
    fn parsed_message_path_skips_unsupported_models() -> Result<(), String> {
        let messages: CountTokenRequest =
            serde_json::from_str(r#"[{"role":"user","content":"hi"}]"#)
                .map_err(|error| error.to_string())?;
        assert_eq!(
            try_count_chat_messages(model("claude"), messages).map_err(|error| error.to_string())?,
            None
        );
        Ok(())
    }

    #[test]
    fn encodes_request_bodies() -> Result<(), String> {
        let result = try_encode_request(model("gpt-4o"), r#"{"text":"hello world"}"#)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "gpt-4o should encode locally".to_string())?;
        assert_eq!(result.count, result.ids.len());
        Ok(())
    }

    #[test]
    fn rejects_malformed_encode_bodies() {
        assert!(try_encode_request(model("gpt-4o"), r#""hello""#).is_err());
    }

    #[test]
    fn selects_gemma_asset_for_gemini_family() {
        assert_eq!(
            required_tokenizer_asset(model("gemini-2.5-pro")),
            Some(TokenizerAsset::Gemma)
        );
        assert_eq!(required_tokenizer_asset(model("gpt-4o")), None);
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
