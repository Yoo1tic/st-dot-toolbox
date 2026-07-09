//! Gemma/Gemini-family tokenization through Hugging Face `tokenizers`.
//!
//! The tokenizer is supplied by JavaScript as a `tokenizer.json` payload. Because a
//! [`GemmaTokenizer`] handle is a transient value derived from a model name — and
//! must exist *before* the provider data loads, so routing can request it — the parsed
//! tokenizer and chat template live in a process-wide [`OnceLock<GemmaTokenizer>`]:
//! a load-once, immutable-after, shared singleton.

use minijinja::Environment;
use minijinja_contrib::pycompat::unknown_method_callback;
use serde::Serialize;
use std::fmt;
use std::io::Read;
use std::sync::OnceLock;
use tokenizers::Tokenizer as HuggingFaceTokenizer;

use crate::{
    CountResult, CountTokenRequest, DecodeResult, EncodeResult, ModelName, ProviderLabel,
    Tokenizer, TokenizerError,
};

const CHAT_TEMPLATE_NAME: &str = "chat_template.jinja";
const TOKENIZER_JSON_NAME: &str = "tokenizer.json";
const BOS_TOKEN: &str = "<bos>";

/// Process-wide storage for the parsed tokenizer and compiled chat template. A runtime-initialized
/// `&'static` needs a `'static` home; `OnceLock` is that home and enforces the
/// write-once, read-many invariant for free.
static GEMMA_TOKENIZER: OnceLock<GemmaTokenizer> = OnceLock::new();

/// Gemma-family tokenizer handle and initialized state.
pub struct GemmaTokenizer {
    tokenizer: HuggingFaceTokenizer,
    chat_templates: Environment<'static>,
}

impl GemmaTokenizer {
    /// Returns the initialized tokenizer, or `None` before provider assets load.
    pub(crate) fn from_model_name(model: &str) -> Option<&'static Self> {
        if Self::supports_model(model) {
            GEMMA_TOKENIZER.get()
        } else {
            None
        }
    }

    pub(crate) fn get_tokenizer(&self) -> &HuggingFaceTokenizer {
        &self.tokenizer
    }

    fn render_chat_template(&self, messages: CountTokenRequest) -> Result<String, TokenizerError> {
        let template = self.chat_templates.get_template(CHAT_TEMPLATE_NAME)?;

        Ok(template.render(ChatTemplateContext {
            messages: messages.into_value(),
            tools: Vec::new(),
            bos_token: BOS_TOKEN,
            add_generation_prompt: false,
            enable_thinking: false,
        })?)
    }
}

impl fmt::Debug for GemmaTokenizer {
    /// Reports only load state; the tokenizer's internals are large and opaque.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GemmaTokenizer")
            .field("initialized", &true)
            .finish()
    }
}

impl Tokenizer for GemmaTokenizer {
    const LABEL: ProviderLabel = ProviderLabel::Gemma;

    fn supports_model(model: &str) -> bool {
        let model = model.to_ascii_lowercase();
        model == "gemma"
            || model == "gemini"
            || model.contains("gemma")
            || model.contains("gemini")
            || model.contains("learnlm")
    }

    /// Counts messages by rendering Gemma's chat template and encoding the result
    /// with the loaded Hugging Face tokenizer.
    ///
    /// Returns [`TokenizerError::UnInitialized`] when the provider data has not loaded, so
    /// JavaScript can defer to the original request path.
    fn count(
        &self,
        model: ModelName,
        messages: CountTokenRequest,
    ) -> Result<CountResult, TokenizerError> {
        let text = self.render_chat_template(messages)?;
        let encoding = self.get_tokenizer().encode_fast(text, false)?;

        Ok(CountResult {
            token_count: encoding.len(),
            model_name: model,
            label: Self::LABEL,
        })
    }

    fn encode(&self, model: ModelName, text: &str) -> Result<EncodeResult, TokenizerError> {
        let encoding = self.get_tokenizer().encode_fast(text, false)?;
        let ids = encoding.get_ids().to_vec();
        let chunks = encoding.get_tokens().to_vec();

        Ok(EncodeResult {
            count: ids.len(),
            ids,
            chunks,
            model_name: model,
            label: Self::LABEL,
        })
    }

    fn decode(&self, model: ModelName, _ids: &[u32]) -> Result<DecodeResult, TokenizerError> {
        Err(TokenizerError::Unsupported(format!(
            "model `{}` is not handled by the local decoder",
            model.as_str()
        )))
    }
}

/// Initializes the Gemma-family tokenizer from a gzipped tar provider bundle.
///
/// The bundle must contain `tokenizer.json` and `chat_template.jinja`.
/// Calling this more than once is harmless; the first successfully parsed
/// tokenizer remains active for the page lifetime.
pub(crate) fn init_tokenizer_bundle(bundle_tar_gz: &[u8]) -> Result<(), TokenizerError> {
    if GEMMA_TOKENIZER.get().is_some() {
        return Ok(());
    }

    let (tokenizer_json, chat_template) = unpack_tokenizer_bundle(bundle_tar_gz)?;
    let chat_template = String::from_utf8(chat_template)?;

    let tokenizer = HuggingFaceTokenizer::from_bytes(&tokenizer_json)?;
    let mut chat_templates = Environment::new();
    chat_templates.set_unknown_method_callback(unknown_method_callback);
    chat_templates.add_template_owned(CHAT_TEMPLATE_NAME.to_string(), chat_template)?;

    let _ = GEMMA_TOKENIZER.set(GemmaTokenizer {
        tokenizer,
        chat_templates,
    });
    Ok(())
}

fn unpack_tokenizer_bundle(bundle_tar_gz: &[u8]) -> Result<(Vec<u8>, Vec<u8>), TokenizerError> {
    let decoder = flate2::read::GzDecoder::new(bundle_tar_gz);
    let mut archive = tar::Archive::new(decoder);
    let mut tokenizer_json = None;
    let mut chat_template = None;

    for entry in archive.entries()? {
        let mut entry = entry?;
        match read_bundle_entry(&mut entry)? {
            Some(TokenizerBundleEntry::TokenizerJson(content)) => tokenizer_json = Some(content),
            Some(TokenizerBundleEntry::ChatTemplate(content)) => chat_template = Some(content),
            None => {}
        };
    }

    Ok((
        required_bundle_file(tokenizer_json, TOKENIZER_JSON_NAME)?,
        required_bundle_file(chat_template, CHAT_TEMPLATE_NAME)?,
    ))
}

enum TokenizerBundleEntry {
    TokenizerJson(Vec<u8>),
    ChatTemplate(Vec<u8>),
}

fn read_bundle_entry<R: Read>(
    entry: &mut tar::Entry<'_, R>,
) -> Result<Option<TokenizerBundleEntry>, TokenizerError> {
    if !entry.header().entry_type().is_file() {
        return Ok(None);
    }

    let file_name = {
        let path = entry.path()?;
        match path.file_name().and_then(|name| name.to_str()) {
            Some(TOKENIZER_JSON_NAME) => TOKENIZER_JSON_NAME,
            Some(CHAT_TEMPLATE_NAME) => CHAT_TEMPLATE_NAME,
            _ => return Ok(None),
        }
    };

    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;

    Ok(Some(match file_name {
        TOKENIZER_JSON_NAME => TokenizerBundleEntry::TokenizerJson(bytes),
        CHAT_TEMPLATE_NAME => TokenizerBundleEntry::ChatTemplate(bytes),
        _ => unreachable!("bundle entry names are filtered above"),
    }))
}

fn required_bundle_file(value: Option<Vec<u8>>, name: &str) -> Result<Vec<u8>, TokenizerError> {
    value.ok_or_else(|| {
        TokenizerError::InvalidProviderBundle(format!("Gemma provider bundle is missing `{name}`"))
    })
}

#[derive(Serialize)]
struct ChatTemplateContext {
    messages: serde_json::Value,
    tools: Vec<serde_json::Value>,
    bos_token: &'static str,
    add_generation_prompt: bool,
    enable_thinking: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tokenizers::models::wordlevel::WordLevel;

    #[test]
    fn recognizes_gemma_family_models() {
        for model in [
            "gemma",
            "gemini",
            "gemini-2.5-pro",
            "learnlm-2.0-flash-experimental",
        ] {
            assert!(GemmaTokenizer::supports_model(model));
        }
        assert!(!GemmaTokenizer::supports_model("gpt-4o"));
    }

    #[test]
    fn uninitialized_handle_reports_provider_context() {
        let provider = crate::TokenizerProvider::from_model_name(&ModelName::from_js("gemini"))
            .expect("gemini is a Gemma-family model");
        assert!(matches!(provider, crate::TokenizerProvider::Gemma(None)));

        let error = provider
            .encode(ModelName::from_js("gemini"), "hello")
            .expect_err("an uninitialized handle cannot encode");
        assert!(matches!(
            error,
            TokenizerError::UnInitialized {
                model_name,
                provider: "gemma",
            } if model_name.as_str() == "gemini"
        ));
    }

    fn messages(json: &str) -> CountTokenRequest {
        serde_json::from_str(json).expect("test body should be valid JSON")
    }

    fn tokenizer_with_template(chat_template: &str) -> Result<GemmaTokenizer, TokenizerError> {
        let tokenizer = HuggingFaceTokenizer::new(WordLevel::default());
        let mut chat_templates = Environment::new();
        chat_templates.set_unknown_method_callback(unknown_method_callback);
        chat_templates
            .add_template_owned(CHAT_TEMPLATE_NAME.to_string(), chat_template.to_owned())?;

        Ok(GemmaTokenizer {
            tokenizer,
            chat_templates,
        })
    }

    fn bundled_chat_template() -> Result<String, TokenizerError> {
        let (_, chat_template) = unpack_tokenizer_bundle(include_bytes!(
            "../../../../assets/gemma/gemma.tar.gz"
        ))?;
        Ok(String::from_utf8(chat_template)?)
    }

    fn test_bundle(
        tokenizer_json: &str,
        chat_template: &str,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut tar_bytes = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut tar_bytes);
            let mut header = tar::Header::new_gnu();
            header.set_size(tokenizer_json.len() as u64);
            header.set_cksum();
            builder.append_data(&mut header, TOKENIZER_JSON_NAME, tokenizer_json.as_bytes())?;

            let mut header = tar::Header::new_gnu();
            header.set_size(chat_template.len() as u64);
            header.set_cksum();
            builder.append_data(&mut header, CHAT_TEMPLATE_NAME, chat_template.as_bytes())?;
            builder.finish()?;
        }

        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&tar_bytes)?;
        Ok(encoder.finish()?)
    }

    #[test]
    fn unpacks_tokenizer_bundle() -> Result<(), String> {
        let tokenizer_json = r#"{"version":"1.0"}"#;
        let chat_template = "{{ bos_token }}";
        let bundle =
            test_bundle(tokenizer_json, chat_template).map_err(|error| error.to_string())?;
        let (actual_tokenizer_json, actual_chat_template) =
            unpack_tokenizer_bundle(&bundle).map_err(|error| error.to_string())?;

        assert_eq!(actual_tokenizer_json, tokenizer_json.as_bytes());
        assert_eq!(actual_chat_template, chat_template.as_bytes());
        Ok(())
    }

    #[test]
    fn chat_template_context_renders_messages() -> Result<(), String> {
        let tokenizer = tokenizer_with_template(
            "{{ bos_token }}{% for message in messages %}{{ message['role'] }}={{ message.get('content') }};{% endfor %}",
        )
        .map_err(|error| error.to_string())?;
        let rendered = tokenizer
            .render_chat_template(messages(r#"[{"role":"user","content":"hello world"}]"#))
            .map_err(|error| error.to_string())?;
        assert_eq!(rendered, "<bos>user=hello world;");
        Ok(())
    }

    #[test]
    fn bundled_chat_template_renders_user_message() -> Result<(), String> {
        let chat_template = bundled_chat_template().map_err(|error| error.to_string())?;
        let tokenizer = tokenizer_with_template(&chat_template).map_err(|error| error.to_string())?;
        let rendered = tokenizer
            .render_chat_template(messages(r#"[{"role":"user","content":"hello world"}]"#))
            .map_err(|error| error.to_string())?;
        assert!(rendered.starts_with("<bos>"));
        assert!(rendered.contains("<|turn>user\nhello world<turn|>"));
        Ok(())
    }

    #[test]
    fn bundled_chat_template_handles_leading_system_message() -> Result<(), String> {
        let chat_template = bundled_chat_template().map_err(|error| error.to_string())?;
        let tokenizer = tokenizer_with_template(&chat_template).map_err(|error| error.to_string())?;
        let rendered = tokenizer
            .render_chat_template(messages(
                r#"[
                    {"role":"system","content":"be concise"},
                    {"role":"user","content":"hello world"}
                ]"#,
            ))
            .map_err(|error| error.to_string())?;
        assert!(rendered.contains("<|turn>system\nbe concise<turn|>"));
        assert!(rendered.contains("<|turn>user\nhello world<turn|>"));
        Ok(())
    }
}
