//! OpenAI-compatible tokenization through `tiktoken-rs`.

use crate::{
    CountResult, CountTokenRequest, DecodeResult, EncodeResult, ModelName, ProviderLabel,
    Tokenizer, TokenizerError,
};
use tiktoken_rs::tokenizer::{self, Tokenizer as TiktokenTokenizer};

pub(crate) mod chat;
pub(crate) mod encode;

/// OpenAI-compatible tokenizer selected from `tiktoken-rs`.
///
/// The model name is supplied per request via [`ModelName`](crate::ModelName),
/// so this handle only carries the resolved `tiktoken-rs` tokenizer kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiTokenizer {
    kind: TiktokenTokenizer,
}

impl OpenAiTokenizer {
    pub(crate) fn from_model_name(model: &str) -> Option<Self> {
        tokenizer::get_tokenizer(model).map(|kind| Self { kind })
    }

    pub(crate) fn kind(&self) -> TiktokenTokenizer {
        self.kind
    }
}

impl Tokenizer for OpenAiTokenizer {
    const LABEL: ProviderLabel = ProviderLabel::OpenAi;

    fn supports_model(model: &str) -> bool {
        tokenizer::get_tokenizer(model).is_some()
    }

    /// Counts chat messages with `tiktoken-rs`.
    ///
    /// A model can resolve a tokenizer yet still lack chat-counting support (for
    /// example the non-chat `text-davinci` models). Those surface as a
    /// [`is_fallback`](TokenizerError::is_fallback) error, which the router folds
    /// into its fallback path rather than a hard failure.
    fn count(
        &self,
        model: ModelName,
        messages: CountTokenRequest,
    ) -> Result<CountResult, TokenizerError> {
        let token_count = chat::count(model.as_str(), messages)?;
        Ok(CountResult {
            token_count,
            model_name: model,
            label: Self::LABEL,
        })
    }

    fn encode(&self, model: ModelName, text: &str) -> Result<Option<EncodeResult>, TokenizerError> {
        encode::encode_text(self, model, Self::LABEL, text).map(Some)
    }

    fn decode(
        &self,
        _model: ModelName,
        _ids: &[u32],
    ) -> Result<Option<DecodeResult>, TokenizerError> {
        Ok(None)
    }
}
