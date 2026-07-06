//! Error types for local tokenization.

use std::{error, fmt};

/// Errors returned by local tokenizer operations.
#[derive(Debug)]
pub enum TokenizerError {
    /// The incoming JSON request body could not be deserialized.
    Json(serde_json::Error),
    /// The request shape is valid JSON but cannot be mapped to tokenizer input.
    InvalidMessage(String),
    /// The tokenizer library rejected the requested model or input.
    Tiktoken(String),
    /// The Hugging Face tokenizer rejected the supplied tokenizer or input.
    Tokenizer(String),
}

impl TokenizerError {
    /// Returns true when the error means SillyTavern should use its fallback path.
    pub fn is_unsupported_model(&self) -> bool {
        match self {
            Self::Tiktoken(message) | Self::InvalidMessage(message) => {
                message.starts_with("No tokenizer found for model ")
                    || message.starts_with("Chat token counting is not supported for model ")
            }
            Self::Json(_) | Self::Tokenizer(_) => false,
        }
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(f, "{error}"),
            Self::InvalidMessage(message) | Self::Tiktoken(message) | Self::Tokenizer(message) => {
                f.write_str(message)
            }
        }
    }
}

impl error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::InvalidMessage(_) | Self::Tiktoken(_) | Self::Tokenizer(_) => None,
        }
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
