//! Text encoding through `tiktoken-rs`.

use serde::{Deserialize, Serialize};
use tiktoken_rs::{CoreBPE, Rank};

use super::OpenAiTokenizer;
use crate::{ModelName, ProviderLabel, TokenizerError};

/// Tokenized text returned to JavaScript.
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct EncodeResult {
    /// Token ids produced by the tokenizer.
    pub ids: Vec<Rank>,
    /// Number of token ids.
    pub count: usize,
    /// Lossy UTF-8 text chunk for each token id.
    pub chunks: Vec<String>,
    /// Model name the encoding was produced for.
    pub model_name: ModelName,
    /// Provider that produced the encoding.
    pub label: ProviderLabel,
}

/// Encodes text with a resolved OpenAI-compatible tokenizer.
pub fn encode_text(
    tokenizer: &OpenAiTokenizer,
    model: ModelName,
    label: ProviderLabel,
    text: &str,
) -> Result<EncodeResult, TokenizerError> {
    Ok(encode_impl(tokenizer.get_tokenizer(), model, label, text))
}

fn encode_impl(bpe: &CoreBPE, model: ModelName, label: ProviderLabel, text: &str) -> EncodeResult {
    let ids = bpe.encode_with_special_tokens(text);
    let chunks = ids
        .iter()
        .map(|&id| {
            let bytes = bpe.decode_bytes(&[id]).unwrap_or_default();
            String::from_utf8(bytes)
                .unwrap_or_else(|e| String::from_utf8_lossy(&e.into_bytes()).into_owned())
        })
        .collect();

    EncodeResult {
        count: ids.len(),
        ids,
        chunks,
        model_name: model,
        label,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_chunks_reassemble_the_text() -> Result<(), String> {
        let text = "hello world, 你好世界";
        let tokenizer = OpenAiTokenizer::from_model_name("gpt-4o")
            .ok_or_else(|| "gpt-4o should resolve".to_string())?;
        let result = encode_text(
            &tokenizer,
            ModelName::from_js("gpt-4o"),
            ProviderLabel::OpenAi,
            text,
        )
        .map_err(|error| error.to_string())?;

        assert_eq!(result.count, result.ids.len());
        assert_eq!(result.chunks.concat(), text);
        assert_eq!(result.model_name.as_str(), "gpt-4o");
        assert_eq!(result.label, ProviderLabel::OpenAi);
        Ok(())
    }

    #[test]
    fn unsupported_models_return_none() {
        assert!(OpenAiTokenizer::from_model_name("qwen2").is_none());
    }
}
