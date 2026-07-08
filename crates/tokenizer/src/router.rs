//! Model-to-tokenizer routing.
//!
//! This module owns the decision of which tokenizer serves a given model or
//! request. It defines the provider-agnostic [`Tokenizer`] contract and the
//! [`TokenizerProvider`] dispatcher that selects a concrete implementation. The
//! crate root keeps request data types and the public request API.

use crate::{
    CountResult, CountTokenRequest, DecodeResult, EncodeResult, GemmaTokenizer, ModelName,
    OpenAiTokenizer, ProviderLabel, TokenizerError,
};

/// Tokenizer implementation selected from a model name.
#[derive(Debug, Clone)]
pub enum TokenizerProvider {
    /// OpenAI-compatible tokenizer served by `tiktoken-rs`.
    OpenAi(OpenAiTokenizer),
    /// Gemma/Gemini-family tokenizer served by Hugging Face `tokenizers`.
    Gemma(GemmaTokenizer),
}

impl TokenizerProvider {
    /// Derives a tokenizer provider directly from a model name.
    pub fn from_model_name(model: &ModelName) -> Option<Self> {
        let model = model.as_str();
        if let Some(tokenizer) = GemmaTokenizer::from_model_name(model) {
            return Some(Self::Gemma(tokenizer));
        }

        OpenAiTokenizer::from_model_name(model).map(Self::OpenAi)
    }

    pub(crate) fn count(
        &self,
        model: ModelName,
        messages: CountTokenRequest,
    ) -> Result<CountResult, TokenizerError> {
        match self {
            Self::OpenAi(tokenizer) => tokenizer.count(model, messages),
            Self::Gemma(tokenizer) => tokenizer.count(model, messages),
        }
    }

    pub(crate) fn encode(
        &self,
        model: ModelName,
        text: &str,
    ) -> Result<EncodeResult, TokenizerError> {
        match self {
            Self::OpenAi(tokenizer) => tokenizer.encode(model, text),
            Self::Gemma(tokenizer) => tokenizer.encode(model, text),
        }
    }

    /// Decodes token ids with the selected provider tokenizer.
    pub fn decode(&self, model: ModelName, ids: &[u32]) -> Result<DecodeResult, TokenizerError> {
        match self {
            Self::OpenAi(tokenizer) => tokenizer.decode(model, ids),
            Self::Gemma(tokenizer) => tokenizer.decode(model, ids),
        }
    }
}

/// Tokenizer operations supported by provider-specific tokenizer handles.
///
/// Each method receives the [`ModelName`] it is serving so the returned result
/// can carry the model name and provider [`LABEL`](Tokenizer::LABEL) for logging.
///
/// The router consults [`supports_model`](Tokenizer::supports_model) to pick a
/// provider, then dispatches [`count`](Tokenizer::count) unconditionally. A
/// provider whose backing resources have not loaded reports this by returning
/// [`TokenizerError::UnInitialized`], so readiness is a per-provider concern
/// expressed through the error channel, not a universal trait obligation.
pub trait Tokenizer {
    /// Stable provider tag stamped onto results produced by this tokenizer.
    const LABEL: ProviderLabel;

    /// Returns whether this provider claims `model`.
    ///
    /// This is a pure predicate over the model name — it does not consult any
    /// loaded resources — so the router can decide provider ownership before a
    /// provider-backed tokenizer has been initialized.
    fn supports_model(model: &str) -> bool
    where
        Self: Sized;

    /// Counts tokens for provider-agnostic parsed chat messages.
    ///
    /// The caller has already gated on [`supports_model`](Tokenizer::supports_model),
    /// so this returns a definitive count — or [`TokenizerError::UnInitialized`] if
    /// the provider recognizes the model but its tokenizer data has not loaded yet. The
    /// implementer adapts the neutral [`CountTokenRequest`] into its own message
    /// representation, keeping message-shape knowledge inside each provider.
    fn count(
        &self,
        model: ModelName,
        messages: CountTokenRequest,
    ) -> Result<CountResult, TokenizerError>;

    /// Encodes text into token ids and token chunks.
    fn encode(&self, model: ModelName, text: &str) -> Result<EncodeResult, TokenizerError>;

    /// Decodes token ids into text. Reserved for a future decode endpoint.
    fn decode(&self, model: ModelName, ids: &[u32]) -> Result<DecodeResult, TokenizerError>;
}
