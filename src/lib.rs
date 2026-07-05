//! Local WASM replacement for SillyTavern's `/api/tokenizers/openai/*` endpoints.
//!
//! Two exports mirror the two request shapes ST sends: [`count_messages`] for
//! `/count` (an array of chat messages) and [`encode_text`] for `/encode` (a
//! single string). Both resolve the encoding from the `model` query param, so a
//! `deepseek-*` model is tokenized with the real `deepseek_v3` BPE rather than
//! ST's cl100k fallback — the whole reason for running a real tokenizer locally.
//!
//! The `*_impl` functions hold the pure logic and are exercised by native
//! `cargo test`; the `#[wasm_bindgen]` wrappers only translate errors to JS.

use std::borrow::Cow;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use tiktoken::CoreBpe;
use wasm_bindgen::prelude::*;

/// Per-message chat-format overhead, straight from OpenAI's token-counting
/// cookbook (also what ST's `/count` adds): every message costs a fixed prelude,
/// a `name` field costs one extra token, and the whole request is padded once.
const TOKENS_PER_MESSAGE: usize = 3;
const TOKENS_PER_NAME: usize = 1;
const TOKENS_PADDING: usize = 3;

/// Resolve a `model` query value to its encoding, preferring the model's real
/// tokenizer (`gpt-4o` → o200k, `deepseek-*` → deepseek_v3, …) and falling back
/// to cl100k_base only for models the crate doesn't recognize.
fn resolve_encoding(model: &str) -> &'static CoreBpe {
    // cl100k_base is always compiled in, so the fallback expect never fires.
    tiktoken::encoding_for_model(model)
        .or_else(|| tiktoken::get_encoding("cl100k_base"))
        .expect("cl100k_base encoding must be available")
}

/// A message field name. `#[serde(borrow)]` keeps it a slice of the request
/// body unless the key contains escape sequences.
#[derive(Deserialize, PartialEq, Eq, Hash)]
struct FieldName<'a>(#[serde(borrow)] Cow<'a, str>);

/// A chat message, deserialized without building a DOM: values stay raw JSON
/// slices, so huge `content` strings aren't copied and non-string fields
/// (e.g. multimodal content arrays) are skipped over, never parsed.
type Message<'a> = HashMap<FieldName<'a>, &'a RawValue>;

/// Token count of a raw JSON value if it is a string, else 0 (only string
/// fields count, matching ST). The common escape-free string is tokenized
/// straight from the borrowed slice; only escaped strings are unescaped
/// into a fresh buffer.
fn count_json_string(enc: &CoreBpe, raw: &RawValue) -> usize {
    let json = raw.get();
    if !json.starts_with('"') {
        return 0;
    }
    match serde_json::from_str::<&str>(json) {
        Ok(s) => enc.count(s),
        Err(_) => {
            let s: String = serde_json::from_str(json)
                .expect("a RawValue starting with '\"' is a valid JSON string");
            enc.count(&s)
        }
    }
}

fn count_messages_impl(enc: &CoreBpe, body_json: &str) -> serde_json::Result<usize> {
    let messages: Vec<Message> = serde_json::from_str(body_json)?;

    // Every string field's value is tokenized (role, content, name…);
    // a `name` key adds one token on top.
    let fields: usize = messages
        .iter()
        .flat_map(|msg| msg.iter())
        .map(|(key, value)| {
            let name = if key.0 == "name" { TOKENS_PER_NAME } else { 0 };
            count_json_string(enc, value) + name
        })
        .sum();

    Ok(TOKENS_PADDING + messages.len() * TOKENS_PER_MESSAGE + fields)
}

#[derive(Serialize)]
struct EncodeResult {
    ids: Vec<u32>,
    count: usize,
    chunks: Vec<String>,
}

fn encode_impl(enc: &CoreBpe, text: &str) -> EncodeResult {
    let ids = enc.encode(text);
    let chunks = ids
        .iter()
        .map(|&id| {
            let bytes = enc.decode(std::slice::from_ref(&id));
            // Move the decode buffer into the String when it's valid UTF-8; a
            // token splitting a multibyte char decodes lossily (as in ST).
            String::from_utf8(bytes)
                .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
        })
        .collect();

    EncodeResult {
        count: ids.len(),
        ids,
        chunks,
    }
}

/// Replaces `POST /api/tokenizers/openai/count?model=<model>`.
///
/// `body_json` is the raw request body — a JSON array of chat messages such as
/// `[{"role":"user","content":"hi"}, …]`. Returns the `token_count` including
/// the per-message and padding overhead ST accounts for.
#[wasm_bindgen]
pub fn count_messages(model: &str, body_json: &str) -> Result<u32, JsError> {
    let total = count_messages_impl(resolve_encoding(model), body_json)?;
    Ok(total as u32)
}

/// Replaces `POST /api/tokenizers/openai/encode?model=<model>`.
///
/// Returns `{ ids, count, chunks }` like ST, so the token highlighter keeps
/// working. `chunks` are per-token decoded strings.
#[wasm_bindgen]
pub fn encode_text(model: &str, text: &str) -> Result<JsValue, JsError> {
    let result = encode_impl(resolve_encoding(model), text);
    Ok(serde_wasm_bindgen::to_value(&result)?)
}

/// Runs on module init for readable panic messages in the browser console.
#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc() -> &'static CoreBpe {
        resolve_encoding("gpt-4o")
    }

    #[test]
    fn count_includes_message_and_padding_overhead() {
        let body = r#"[{"role":"user","content":"hello world"}]"#;
        let expected =
            TOKENS_PADDING + TOKENS_PER_MESSAGE + enc().count("user") + enc().count("hello world");
        assert_eq!(count_messages_impl(enc(), body).unwrap(), expected);
    }

    #[test]
    fn name_field_costs_its_tokens_plus_one() {
        let plain = r#"[{"role":"user","content":"hi"}]"#;
        let named = r#"[{"role":"user","content":"hi","name":"bob"}]"#;
        let diff =
            count_messages_impl(enc(), named).unwrap() - count_messages_impl(enc(), plain).unwrap();
        assert_eq!(diff, enc().count("bob") + TOKENS_PER_NAME);
    }

    #[test]
    fn non_string_fields_are_ignored() {
        let multimodal = r#"[{"role":"user","content":[{"type":"text","text":"hi"}]}]"#;
        let no_content = r#"[{"role":"user"}]"#;
        assert_eq!(
            count_messages_impl(enc(), multimodal).unwrap(),
            count_messages_impl(enc(), no_content).unwrap(),
        );
    }

    #[test]
    fn escaped_strings_are_unescaped_before_counting() {
        let escaped = r#"[{"role":"user","content":"a\nb"}]"#;
        let expected =
            TOKENS_PADDING + TOKENS_PER_MESSAGE + enc().count("user") + enc().count("a\nb");
        assert_eq!(count_messages_impl(enc(), escaped).unwrap(), expected);
    }

    #[test]
    fn invalid_body_is_an_error() {
        assert!(count_messages_impl(enc(), "not json").is_err());
        assert!(count_messages_impl(enc(), r#"{"role":"user"}"#).is_err());
    }

    #[test]
    fn encode_chunks_reassemble_the_text() {
        let text = "hello world, 你好世界";
        let result = encode_impl(enc(), text);
        assert_eq!(result.count, result.ids.len());
        assert_eq!(result.ids, enc().encode(text));
        assert_eq!(result.chunks.concat(), text);
    }
}
