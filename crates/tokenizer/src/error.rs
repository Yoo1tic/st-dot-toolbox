//! Error types for local tokenization.

use thiserror::Error;

use crate::ProviderLabel;
use serde::Serialize;

/// Structured tokenizer error payload returned across the WASM boundary.
#[derive(Debug, Serialize)]
pub struct TokenizerErrorBody {
    pub error_type: &'static str,
    pub message: String,
}

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
    HuggingfaceTokenizer(String),

    #[error("Tokenizer for provider {0:?} is not initialized")]
    UnInitialized(ProviderLabel),

    #[error("{0}")]
    Unsupported(String),
}

impl TokenizerError {
    /// Converts this error into the stable object shape JavaScript consumes.
    pub fn body(&self) -> TokenizerErrorBody {
        TokenizerErrorBody {
            error_type: match self {
                Self::Json(_) => "Json",
                Self::InvalidMessage(_) => "InvalidMessage",
                Self::Tiktoken(_) => "Tiktoken",
                Self::HuggingfaceTokenizer(_) => "HuggingfaceTokenizer",
                Self::UnInitialized(_) => "UnInitialized",
                Self::Unsupported(_) => "Unsupported",
            },
            message: self.to_string(),
        }
    }

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
            Self::Unsupported(_) => true,
            Self::Json(_) | Self::HuggingfaceTokenizer(_) => false,
        }
    }

    /// Returns true when local count should use the heuristic estimator.
    ///
    /// Uninitialized providers are deliberately excluded: those mean JavaScript
    /// should load the missing tokenizer asset and let the native ajax path serve
    /// that one request.
    pub fn should_estimate_count(&self) -> bool {
        !matches!(self, Self::UnInitialized(_)) && self.is_fallback()
    }
}
