//! Text encoding through `tiktoken-rs`.

use serde::Serialize;
use tiktoken_rs::{CoreBPE, Rank, bpe_for_tokenizer, tokenizer};

use crate::TokenizerError;

/// Tokenized text returned to JavaScript.
#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct EncodeResult {
    /// Token ids produced by the tokenizer.
    pub ids: Vec<Rank>,
    /// Number of token ids.
    pub count: usize,
    /// Lossy UTF-8 text chunk for each token id.
    pub chunks: Vec<String>,
}

/// Encodes text with the tokenizer resolved from `model`.
pub fn encode_text(model: &str, text: &str) -> Result<Option<EncodeResult>, TokenizerError> {
    let Some(tokenizer) = tokenizer::get_tokenizer(model) else {
        return Ok(None);
    };
    let bpe = bpe_for_tokenizer(tokenizer)
        .map_err(|error| TokenizerError::Tiktoken(error.to_string()))?;

    Ok(Some(encode_impl(bpe, text)))
}

fn encode_impl(bpe: &CoreBPE, text: &str) -> EncodeResult {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_chunks_reassemble_the_text() -> Result<(), String> {
        let text = "hello world, 你好世界";
        let result = encode_text("gpt-4o", text)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "gpt-4o should resolve".to_string())?;

        assert_eq!(result.count, result.ids.len());
        assert_eq!(result.chunks.concat(), text);
        Ok(())
    }

    #[test]
    fn unsupported_models_return_none() -> Result<(), String> {
        assert_eq!(
            encode_text("qwen2", "hello").map_err(|error| error.to_string())?,
            None
        );
        Ok(())
    }
}
