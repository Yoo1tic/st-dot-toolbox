//! Heuristic fallback tokenizer for models with no exact local tokenizer.
//!
//! When a model matches neither the OpenAI nor the Gemma provider, callers still
//! want *some* local count instead of round-tripping to SillyTavern. This module
//! provides the crudest possible estimate — a fixed characters-per-token ratio —
//! so [`try_count_messages`](crate::try_count_messages) always yields a result.

use crate::{
    CountResult, CountTokenRequest, DecodeResult, EncodeResult, ModelName, ProviderLabel,
    Tokenizer, TokenizerError,
};

/// Average characters per token assumed by the heuristic estimator.
///
/// Four characters per token is the common rule of thumb for English-heavy
/// prompts. The estimate over-counts on JSON structure, which keeps it a
/// conservative upper bound for context-budget purposes.
const CHARS_PER_TOKEN: usize = 4;

/// Last-resort token estimator for models without an exact local tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FallbackTokenizer;

impl FallbackTokenizer {
    /// Estimates the token count of a raw request body.
    ///
    /// The estimate is infallible: it never parses the body, so malformed JSON
    /// still yields a number rather than an error.
    pub(crate) fn estimate(self, model: ModelName, body_json: &str) -> CountResult {
        CountResult {
            token_count: estimate_tokens(body_json),
            model_name: model,
            label: Self::LABEL,
        }
    }
}

impl Tokenizer for FallbackTokenizer {
    const LABEL: ProviderLabel = ProviderLabel::Fallback;

    /// The fallback estimator is the last resort, so it claims every model.
    fn supports_model(_model: &str) -> bool {
        true
    }

    fn count(
        &self,
        model: ModelName,
        messages: CountTokenRequest,
    ) -> Result<CountResult, TokenizerError> {
        Ok(self.estimate(model, &messages.into_value().to_string()))
    }

    fn encode(
        &self,
        _model: ModelName,
        _text: &str,
    ) -> Result<Option<EncodeResult>, TokenizerError> {
        Ok(None)
    }

    fn decode(
        &self,
        _model: ModelName,
        _ids: &[u32],
    ) -> Result<Option<DecodeResult>, TokenizerError> {
        Ok(None)
    }
}

/// Estimates tokens as `ceil(chars / CHARS_PER_TOKEN)`.
fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(CHARS_PER_TOKEN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_one_token_per_four_characters() {
        assert_eq!(estimate_tokens("12345678"), 2);
    }

    #[test]
    fn rounds_partial_tokens_up() {
        assert_eq!(estimate_tokens("hello"), 2);
    }

    #[test]
    fn empty_body_estimates_zero_tokens() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn stamps_the_fallback_label() {
        let result = FallbackTokenizer.estimate(ModelName::from_js("claude"), "hello world");
        assert_eq!(result.label, ProviderLabel::Fallback);
        assert_eq!(result.model_name.as_str(), "claude");
        assert!(result.token_count > 0);
    }
}
