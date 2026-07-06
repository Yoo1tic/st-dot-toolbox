//! Local tokenization logic for SillyTavern-compatible OpenAI requests.
//!
//! The crate deliberately has no `wasm-bindgen` dependency. It exposes pure
//! Rust operations that the root WASM facade can adapt for JavaScript.

mod chat;
mod encode;
mod error;
mod gemma;

pub use encode::EncodeResult;
pub use error::TokenizerError;

/// Counts OpenAI-compatible chat messages with `tiktoken-rs`.
///
/// `body_json` must be a JSON array of OpenAI-style chat messages. The adapter
/// accepts SillyTavern's stringified `tool_calls` quirk as well as normal array
/// tool calls. `Ok(None)` means the model is unsupported locally and callers
/// should fall back to SillyTavern's original tokenizer path.
pub fn try_count_messages(model: &str, body_json: &str) -> Result<Option<usize>, TokenizerError> {
    if gemma::supports_model(model) {
        return Ok(None);
    }

    match chat::count(model, body_json) {
        Ok(count) => Ok(Some(count)),
        Err(error) if error.is_unsupported_model() => Ok(None),
        Err(error) => Err(error),
    }
}

/// Encodes text with the tokenizer associated with `model`.
///
/// Returns token ids, token count, and per-token UTF-8 chunks. `Ok(None)` means
/// the model is unknown to `tiktoken-rs` and callers should fall back to their
/// original tokenizer path.
pub fn try_encode_text(model: &str, text: &str) -> Result<Option<EncodeResult>, TokenizerError> {
    if let Some(result) = gemma::encode_text(model, text)? {
        return Ok(Some(result));
    }

    encode::encode_text(model, text)
}

/// Initializes the lazy Gemma/Gemini-family tokenizer from `tokenizer.json`.
pub fn init_gemma_tokenizer(tokenizer_json: &str) -> Result<(), TokenizerError> {
    gemma::init_tokenizer(tokenizer_json)
}

/// Returns whether the Gemma/Gemini-family tokenizer is ready for local use.
pub fn is_gemma_tokenizer_initialized() -> bool {
    gemma::is_initialized()
}

/// Counts plain text with the Gemma/Gemini-family tokenizer when available.
pub fn try_count_text(model: &str, text: &str) -> Result<Option<usize>, TokenizerError> {
    gemma::count_text(model, text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiktoken_rs::{ChatCompletionRequestMessage, FunctionCall, num_tokens_from_messages};

    fn count(model: &str, body_json: &str) -> Result<usize, TokenizerError> {
        chat::count(model, body_json)
    }

    #[test]
    fn unsupported_chat_models_return_none() -> Result<(), String> {
        for model in ["claude", "text-davinci-003"] {
            assert_eq!(
                try_count_messages(model, r#"[{"role":"user","content":"hi"}]"#)
                    .map_err(|error| error.to_string())?,
                None
            );
        }
        Ok(())
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
