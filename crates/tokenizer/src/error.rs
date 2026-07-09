//! Error types for local tokenization.

use thiserror::Error;

use crate::ModelName;
use serde::Serialize;

/// Structured tokenizer error payload returned across the WASM boundary.
#[derive(Debug, Serialize)]
pub struct TokenizerErrorBody {
    pub error: &'static str,
    pub message: String,
    pub model_name: ModelName,
    pub provider: &'static str,
}

/// Errors returned by local tokenizer operations.
#[derive(Debug, Error)]
pub enum TokenizerError {
    /// The incoming JSON request body could not be deserialized.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Provider bundle bytes could not be read or decompressed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Provider bundle text files were not valid UTF-8.
    #[error(transparent)]
    Utf8(#[from] std::string::FromUtf8Error),
    /// The request shape is valid JSON but cannot be mapped to tokenizer input.
    #[error("{0}")]
    InvalidMessage(String),
    /// The tokenizer provider bundle was missing required files or malformed.
    #[error("{0}")]
    InvalidProviderBundle(String),
    /// The tokenizer library rejected the requested model or input.
    #[error(transparent)]
    Tiktoken(#[from] anyhow::Error),
    /// The Hugging Face tokenizer rejected the supplied tokenizer or input.
    #[error(transparent)]
    HuggingfaceTokenizer(#[from] tokenizers::Error),
    /// The Gemma chat template could not be compiled or rendered.
    #[error(transparent)]
    MiniJinja(#[from] minijinja::Error),

    #[error("Tokenizer provider `{provider}` is not initialized for model `{model_name}`")]
    UnInitialized {
        model_name: ModelName,
        provider: &'static str,
    },

    #[error("Unsupported local tokenizer request: {0}")]
    Unsupported(String),
}

impl TokenizerError {
    fn error_name(&self) -> &'static str {
        match self {
            Self::Json(_) => "Json",
            Self::Io(_) => "Io",
            Self::Utf8(_) => "Utf8",
            Self::InvalidMessage(_) => "InvalidMessage",
            Self::InvalidProviderBundle(_) => "InvalidProviderBundle",
            Self::Tiktoken(_) => "Tiktoken",
            Self::HuggingfaceTokenizer(_) => "HuggingfaceTokenizer",
            Self::MiniJinja(_) => "MiniJinja",
            Self::UnInitialized { .. } => "UnInitialized",
            Self::Unsupported(_) => "Unsupported",
        }
    }

    /// Converts this error into the stable object shape JavaScript consumes.
    pub fn body(&self) -> TokenizerErrorBody {
        self.body_for_model(ModelName::from_js(""))
    }

    /// Converts this error into the stable object shape using request context.
    pub fn body_for_model(&self, model_name: ModelName) -> TokenizerErrorBody {
        let (model_name, provider) = match self {
            Self::UnInitialized {
                model_name,
                provider,
            } => (model_name.clone(), *provider),
            _ => (model_name, ""),
        };

        TokenizerErrorBody {
            error: self.error_name(),
            message: self.to_string(),
            model_name,
            provider,
        }
    }
}
