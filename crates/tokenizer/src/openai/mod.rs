//! OpenAI-compatible tokenization through `tiktoken-rs`.

use crate::{
    CountResult, CountTokenRequest, DecodeResult, EncodeResult, ModelName, ProviderLabel,
    Tokenizer, TokenizerError,
};
use std::fmt;
use tiktoken_rs::tokenizer::{self, Tokenizer as TiktokenTokenizer};
use tiktoken_rs::{CoreBPE, bpe_for_tokenizer};

pub(crate) mod chat;
pub(crate) mod encode;

/// OpenAI-compatible tokenizer handle backed by `tiktoken-rs`.
///
/// Holds the resolved tokenizer kind and a borrow of `tiktoken-rs`'s shared BPE
/// singleton for that kind. The model name is still supplied per request via
/// [`ModelName`](crate::ModelName), so one handle can serve any model name that
/// maps to the same tokenizer.
#[derive(Clone, Copy)]
pub struct OpenAiTokenizer {
    kind: TiktokenTokenizer,
    tokenizer: &'static CoreBPE,
}

impl fmt::Debug for OpenAiTokenizer {
    /// Reports the compact tokenizer kind; the BPE internals are large.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiTokenizer")
            .field("kind", &self.kind)
            .finish()
    }
}

impl PartialEq for OpenAiTokenizer {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl Eq for OpenAiTokenizer {}

impl OpenAiTokenizer {
    pub(crate) fn from_model_name(model: &str) -> Option<Self> {
        let kind = tokenizer::get_tokenizer(model)?;
        let tokenizer = bpe_for_tokenizer(kind).ok()?;
        Some(Self { kind, tokenizer })
    }

    pub(crate) fn get_tokenizer(self) -> &'static CoreBPE {
        self.tokenizer
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
    /// example the non-chat `text-davinci` models). Those errors are returned to
    /// JavaScript so the original request path can handle the call.
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

    fn encode(&self, model: ModelName, text: &str) -> Result<EncodeResult, TokenizerError> {
        encode::encode_text(self, model, Self::LABEL, text)
    }

    fn decode(&self, model: ModelName, _ids: &[u32]) -> Result<DecodeResult, TokenizerError> {
        Err(TokenizerError::Unsupported(format!(
            "model `{}` is not handled by the local decoder",
            model.as_str()
        )))
    }
}
