//! Model-to-tokenizer routing.
//!
//! This module owns the decision of which tokenizer serves a given model or
//! external asset. It defines the provider-agnostic [`Tokenizer`] contract, the
//! [`TokenizerProvider`] dispatcher that selects a concrete implementation, and
//! the [`TokenizerAsset`] table that maps loadable assets back to a provider.
//! The crate root keeps request data types and the public request API.

use crate::{
    CountResult, CountTokenRequest, DecodeResult, EncodeResult, FallbackTokenizer, GemmaTokenizer,
    ModelName, OpenAiTokenizer, ProviderLabel, TokenizerError,
};

/// Tokenizer implementation selected from a model name.
#[derive(Debug, Clone)]
pub enum TokenizerProvider {
    /// OpenAI-compatible tokenizer served by `tiktoken-rs`.
    OpenAi(OpenAiTokenizer),
    /// Gemma/Gemini-family tokenizer served by Hugging Face `tokenizers`.
    Gemma(GemmaTokenizer),
    /// Heuristic character-ratio estimator for models with no exact tokenizer.
    Fallback(FallbackTokenizer),
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

    /// Derives a provider from a model name, falling back to the heuristic
    /// estimator when no exact local tokenizer claims the model.
    pub(crate) fn from_model_name_or_fallback(model: &ModelName) -> Self {
        Self::from_model_name(model).unwrap_or(Self::Fallback(FallbackTokenizer))
    }

    pub(crate) fn count(
        &self,
        model: ModelName,
        messages: CountTokenRequest,
    ) -> Result<CountResult, TokenizerError> {
        match self {
            Self::OpenAi(tokenizer) => tokenizer_countor(tokenizer, model, messages),
            Self::Gemma(tokenizer) => tokenizer_countor(tokenizer, model, messages),
            Self::Fallback(tokenizer) => tokenizer_countor(tokenizer, model, messages),
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
            Self::Fallback(tokenizer) => tokenizer.encode(model, text),
        }
    }

    /// Decodes token ids with the selected provider tokenizer.
    pub fn decode(&self, model: ModelName, ids: &[u32]) -> Result<DecodeResult, TokenizerError> {
        match self {
            Self::OpenAi(tokenizer) => tokenizer.decode(model, ids),
            Self::Gemma(tokenizer) => tokenizer.decode(model, ids),
            Self::Fallback(tokenizer) => tokenizer.decode(model, ids),
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
/// [`TokenizerError::UnInitialized`], which the router folds into its fallback
/// path — so readiness is a per-provider concern expressed through the error
/// channel, not a universal trait obligation.
pub trait Tokenizer {
    /// Stable provider tag stamped onto results produced by this tokenizer.
    const LABEL: ProviderLabel;

    /// Returns whether this provider claims `model`.
    ///
    /// This is a pure predicate over the model name — it does not consult any
    /// loaded resources — so the router can decide provider ownership before an
    /// asset-backed tokenizer has been initialized.
    fn supports_model(model: &str) -> bool
    where
        Self: Sized;

    /// Counts tokens for provider-agnostic parsed chat messages.
    ///
    /// The caller has already gated on [`supports_model`](Tokenizer::supports_model),
    /// so this returns a definitive count — or [`TokenizerError::UnInitialized`] if
    /// the provider recognizes the model but its asset has not loaded yet. The
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

pub(crate) fn tokenizer_countor<T>(
    tokenizer: &T,
    model: ModelName,
    text: CountTokenRequest,
) -> Result<CountResult, TokenizerError>
where
    T: Tokenizer + ?Sized,
{
    let fallback_model = model.clone();
    let fallback_text = text.clone();

    tokenizer.count(model, text).or_else(|error| {
        if error.should_estimate_count() {
            FallbackTokenizer.count(fallback_model, fallback_text)
        } else {
            Err(error)
        }
    })
}

/// Derives a tokenizer provider from a model-name value.
pub trait TokenizerProviderForModel {
    /// Returns the provider that should serve this model, or `None` if unknown.
    fn tokenizer_provider(&self) -> Option<TokenizerProvider>;
}

impl TokenizerProviderForModel for ModelName {
    fn tokenizer_provider(&self) -> Option<TokenizerProvider> {
        TokenizerProvider::from_model_name(self)
    }
}

/// External tokenizer assets that JavaScript can fetch and pass back to Rust.
///
/// An asset names *what to load*, independent of load state, so it carries no
/// tokenizer handle — the handle is derived from the singleton once the asset
/// has been installed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerAsset {
    /// Hugging Face tokenizer JSON for Gemma/Gemini-family models.
    Gemma,
}

impl TokenizerAsset {
    /// Stable identifier understood by the JavaScript asset loader.
    pub fn id(self) -> &'static str {
        match self {
            Self::Gemma => "gemma",
        }
    }

    /// Resolves an asset from the id JavaScript passes back with its bytes.
    pub(crate) fn from_id(asset_id: &str) -> Result<Self, TokenizerError> {
        match asset_id {
            "gemma" => Ok(Self::Gemma),
            other => Err(TokenizerError::Unsupported(format!(
                "unknown tokenizer asset `{other}`"
            ))),
        }
    }

    /// Installs this asset's tokenizer from raw `tokenizer.json` bytes.
    ///
    /// Initialization is an asset concern, not a provider one: it targets the
    /// module that owns the singleton directly, so no provider handle is built.
    pub(crate) fn init(self, tokenizer_json: &str) -> Result<(), TokenizerError> {
        match self {
            Self::Gemma => crate::google::gemma::init_tokenizer(tokenizer_json),
        }
    }
}
