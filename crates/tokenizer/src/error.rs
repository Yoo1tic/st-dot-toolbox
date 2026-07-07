//! Error types for local tokenization.

use thiserror::Error;

use crate::ProviderLabel;

/// Errors returned by local tokenizer operations.
#[derive(Debug, Error)]
pub enum TokenizerError {
    /// The incoming JSON request body could not be deserialized.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// The request shape is valid JSON but cannot be mapped to tokenizer input.
    #[error("{0}")]
    InvalidMessage(String),
    /// The tokenizer library rejected the requested model or input.
    #[error("{0}")]
    Tiktoken(String),
    /// The Hugging Face tokenizer rejected the supplied tokenizer or input.
    #[error("{0}")]
    Tokenizer(String),

    #[error("Tokenizer for provider {0:?} is not initialized")]
    UnInitialized(ProviderLabel),
}

impl TokenizerError {
    /// Returns true when the error means SillyTavern should use its fallback path.
    ///
    /// Covers both "no exact tokenizer exists for this model" and "the provider's
    /// asset has not loaded yet": in either case there is no usable local count, so
    /// the caller falls back rather than surfacing a hard error.
    pub fn is_fallback(&self) -> bool {
        match self {
            Self::Tiktoken(message) | Self::InvalidMessage(message) => {
                message.starts_with("No tokenizer found for model ")
                    || message.starts_with("Chat token counting is not supported for model ")
            }
            Self::UnInitialized(_) => true,
            Self::Json(_) | Self::Tokenizer(_) => false,
        }
    }
}
