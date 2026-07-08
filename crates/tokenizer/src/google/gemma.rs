//! Gemma/Gemini-family tokenization through Hugging Face `tokenizers`.
//!
//! The tokenizer is supplied by JavaScript as a `tokenizer.json` payload. Because a
//! [`GemmaTokenizer`] handle is a transient value derived from a model name — and
//! must exist *before* the provider data loads, so routing can request it — the parsed
//! tokenizer lives in a process-wide [`OnceLock`]: a load-once, immutable-after,
//! shared singleton. Each handle borrows that singleton, holding `None` until it
//! is present. This keeps the WASM binary small and avoids reparsing a large
//! tokenizer on every request.

use serde::Deserialize;
use std::fmt;
use std::sync::OnceLock;
use tokenizers::Tokenizer as HuggingFaceTokenizer;

use crate::{
    CountResult, CountTokenRequest, DecodeResult, EncodeResult, ModelName, ProviderLabel,
    Tokenizer, TokenizerError,
};

/// Process-wide storage for the parsed tokenizer. A runtime-initialized
/// `&'static` needs a `'static` home; `OnceLock` is that home and enforces the
/// write-once, read-many invariant for free.
static GEMMA_TOKENIZER: OnceLock<HuggingFaceTokenizer> = OnceLock::new();

/// Gemma-family tokenizer handle backed by a Hugging Face `tokenizer.json`.
///
/// Holds a borrow of the shared singleton, or `None` before the provider data has loaded.
/// The `None` state is what lets routing recognize a Gemma model yet still report
/// itself uninitialized, so callers can defer to the original request until the provider
/// arrives.
#[derive(Clone, Copy)]
pub struct GemmaTokenizer {
    tokenizer: Option<&'static HuggingFaceTokenizer>,
}

impl GemmaTokenizer {
    /// Returns a handle for `model`, borrowing the tokenizer if it has loaded, or
    /// `None` if `model` is not a Gemma-family model.
    ///
    /// This is the sole constructor: it captures the singleton's current state so
    /// [`count`](Tokenizer::count) and [`encode`](Tokenizer::encode) can read the
    /// borrow without touching the global again.
    pub(crate) fn from_model_name(model: &str) -> Option<Self> {
        Self::supports_model(model).then(|| Self {
            tokenizer: GEMMA_TOKENIZER.get(),
        })
    }

    pub(crate) fn get_tokenizer(self) -> Option<&'static HuggingFaceTokenizer> {
        self.tokenizer
    }
}

impl fmt::Debug for GemmaTokenizer {
    /// Reports only load state; the tokenizer's internals are large and opaque.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GemmaTokenizer")
            .field("initialized", &self.tokenizer.is_some())
            .finish()
    }
}

impl Tokenizer for GemmaTokenizer {
    const LABEL: ProviderLabel = ProviderLabel::Gemma;

    fn supports_model(model: &str) -> bool {
        let model = model.to_ascii_lowercase();
        model == "gemma"
            || model == "gemini"
            || model.contains("gemma")
            || model.contains("gemini")
            || model.contains("learnlm")
    }

    /// Counts messages by flattening their text and encoding it with the loaded
    /// Hugging Face tokenizer.
    ///
    /// Returns [`TokenizerError::UnInitialized`] when the provider data has not loaded, so
    /// JavaScript can defer to the original request path. The count is not
    /// chat-template exact — Gemma's template is not applied — but it uses the
    /// model's real vocabulary.
    fn count(
        &self,
        model: ModelName,
        messages: CountTokenRequest,
    ) -> Result<CountResult, TokenizerError> {
        let Some(tokenizer) = self.get_tokenizer() else {
            return Err(TokenizerError::UnInitialized {
                model_name: model,
                provider: Self::LABEL.as_str(),
            });
        };

        let text = flatten_messages(messages)?;
        let encoding = tokenizer.encode_fast(text, false)?;

        Ok(CountResult {
            token_count: encoding.len(),
            model_name: model,
            label: Self::LABEL,
        })
    }

    fn encode(&self, model: ModelName, text: &str) -> Result<EncodeResult, TokenizerError> {
        let Some(tokenizer) = self.get_tokenizer() else {
            return Err(TokenizerError::UnInitialized {
                model_name: model,
                provider: Self::LABEL.as_str(),
            });
        };

        let encoding = tokenizer.encode_fast(text, false)?;
        let ids = encoding.get_ids().to_vec();
        let chunks = encoding.get_tokens().to_vec();

        Ok(EncodeResult {
            count: ids.len(),
            ids,
            chunks,
            model_name: model,
            label: Self::LABEL,
        })
    }

    fn decode(&self, model: ModelName, _ids: &[u32]) -> Result<DecodeResult, TokenizerError> {
        Err(TokenizerError::Unsupported(format!(
            "model `{}` is not handled by the local decoder",
            model.as_str()
        )))
    }
}

/// Initializes the Gemma-family tokenizer from a Hugging Face `tokenizer.json`.
///
/// Calling this more than once is harmless; the first successfully parsed
/// tokenizer remains active for the page lifetime.
pub fn init_tokenizer(tokenizer_json: &str) -> Result<(), TokenizerError> {
    if GEMMA_TOKENIZER.get().is_some() {
        return Ok(());
    }

    let tokenizer = HuggingFaceTokenizer::from_bytes(tokenizer_json.as_bytes())?;
    let _ = GEMMA_TOKENIZER.set(tokenizer);
    Ok(())
}

/// Flattens neutral chat messages into a single string for the Gemma tokenizer.
///
/// Each message contributes its `role` marker and textual `content`, one message
/// per line. Non-text content parts (for example image URLs) are dropped, since
/// the local tokenizer only measures text. This keeps message-shape knowledge
/// inside the Gemma provider, mirroring the OpenAI adapter.
fn flatten_messages(messages: CountTokenRequest) -> Result<String, TokenizerError> {
    let raw = serde_json::from_value::<Vec<RawMessage>>(messages.into_value())?;

    let mut text = String::new();
    for message in raw {
        if let Some(role) = message.role {
            text.push_str(&role);
            text.push('\n');
        }
        if let Some(content) = message.content {
            content.push_text_into(&mut text);
        }
        text.push('\n');
    }

    Ok(text)
}

/// Minimal view of a chat message: only the fields that carry text to tokenize.
#[derive(Deserialize)]
struct RawMessage {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<RawContent>,
}

/// Message content as either a plain string or an array of typed parts.
#[derive(Deserialize)]
#[serde(untagged)]
enum RawContent {
    Text(String),
    Parts(Vec<RawContentPart>),
}

/// One content part; only its optional `text` is relevant to tokenization.
#[derive(Deserialize)]
struct RawContentPart {
    #[serde(default)]
    text: Option<String>,
}

impl RawContent {
    /// Appends this content's text to `out`, joining multipart text in order.
    fn push_text_into(self, out: &mut String) {
        match self {
            Self::Text(text) => out.push_str(&text),
            Self::Parts(parts) => {
                for part in parts {
                    if let Some(part_text) = part.text {
                        out.push_str(&part_text);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_gemma_family_models() {
        for model in [
            "gemma",
            "gemini",
            "gemini-2.5-pro",
            "learnlm-2.0-flash-experimental",
        ] {
            assert!(GemmaTokenizer::supports_model(model));
        }
        assert!(!GemmaTokenizer::supports_model("gpt-4o"));
    }

    #[test]
    fn uninitialized_handle_reports_provider_context() {
        let handle =
            GemmaTokenizer::from_model_name("gemini").expect("gemini is a Gemma-family model");
        assert!(handle.get_tokenizer().is_none());

        let error = handle
            .encode(ModelName::from_js("gemini"), "hello")
            .expect_err("an uninitialized handle cannot encode");
        assert!(matches!(
            error,
            TokenizerError::UnInitialized {
                model_name,
                provider: "gemma",
            } if model_name.as_str() == "gemini"
        ));
    }

    fn messages(json: &str) -> CountTokenRequest {
        serde_json::from_str(json).expect("test body should be valid JSON")
    }

    #[test]
    fn flatten_messages_joins_roles_and_text() -> Result<(), String> {
        let flattened = flatten_messages(messages(r#"[{"role":"user","content":"hello world"}]"#))
            .map_err(|error| error.to_string())?;
        assert_eq!(flattened, "user\nhello world\n");
        Ok(())
    }

    #[test]
    fn flatten_messages_concatenates_content_parts() -> Result<(), String> {
        let body = r#"[{"role":"user","content":[
            {"type":"text","text":"hello"},
            {"type":"image_url","image_url":{"url":"data:..."}},
            {"type":"text","text":" world"}
        ]}]"#;
        let flattened = flatten_messages(messages(body)).map_err(|error| error.to_string())?;
        assert_eq!(flattened, "user\nhello world\n");
        Ok(())
    }
}
