//! Gemma/Gemini-family tokenization through Hugging Face `tokenizers`.
//!
//! The tokenizer is supplied by JavaScript as a `tokenizer.json` asset and kept
//! as a process-local singleton. This keeps the WASM binary small and avoids
//! reparsing a large tokenizer for every count request.

use std::sync::OnceLock;

use tokenizers::Tokenizer;

use crate::{EncodeResult, TokenizerError};

static GEMMA_TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

/// Returns true when `model` should use the Gemma-family tokenizer.
pub fn supports_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model == "gemma"
        || model == "gemini"
        || model.contains("gemma")
        || model.contains("gemini")
        || model.contains("learnlm")
}

/// Initializes the Gemma-family tokenizer from a Hugging Face `tokenizer.json`.
///
/// Calling this more than once is harmless; the first successfully parsed
/// tokenizer remains active for the page lifetime.
pub fn init_tokenizer(tokenizer_json: &str) -> Result<(), TokenizerError> {
    if GEMMA_TOKENIZER.get().is_some() {
        return Ok(());
    }

    let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes())
        .map_err(|error| TokenizerError::Tokenizer(error.to_string()))?;
    let _ = GEMMA_TOKENIZER.set(tokenizer);
    Ok(())
}

/// Returns whether the Gemma-family tokenizer has been initialized.
pub fn is_initialized() -> bool {
    GEMMA_TOKENIZER.get().is_some()
}

/// Encodes text with the initialized Gemma-family tokenizer.
pub fn encode_text(model: &str, text: &str) -> Result<Option<EncodeResult>, TokenizerError> {
    if !supports_model(model) {
        return Ok(None);
    }

    let Some(tokenizer) = GEMMA_TOKENIZER.get() else {
        return Ok(None);
    };

    let encoding = tokenizer
        .encode(text, false)
        .map_err(|error| TokenizerError::Tokenizer(error.to_string()))?;
    let ids = encoding.get_ids().to_vec();
    let chunks = encoding.get_tokens().to_vec();

    Ok(Some(EncodeResult {
        count: ids.len(),
        ids,
        chunks,
    }))
}

/// Counts plain text with the initialized Gemma-family tokenizer.
pub fn count_text(model: &str, text: &str) -> Result<Option<usize>, TokenizerError> {
    Ok(encode_text(model, text)?.map(|result| result.count))
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
            assert!(supports_model(model));
        }
        assert!(!supports_model("gpt-4o"));
    }

    #[test]
    fn uninitialized_tokenizer_returns_none() -> Result<(), String> {
        assert_eq!(
            encode_text("gemma", "hello").map_err(|error| error.to_string())?,
            None
        );
        assert_eq!(
            count_text("gemma", "hello").map_err(|error| error.to_string())?,
            None
        );
        Ok(())
    }
}
