//! WASM boundary for the SillyTavern tokenizer extension.
//!
//! The surrounding crate owns the tokenizer logic. This module only exposes
//! stable JavaScript-callable functions and converts Rust results into JS values.

use crate::{ModelName, TokenizerError};
use wasm_bindgen::prelude::*;

fn tokenizer_error(model_name: &str, error: TokenizerError) -> Result<JsValue, JsError> {
    Ok(serde_wasm_bindgen::to_value(
        &error.body_for_model(ModelName::from_js(model_name)),
    )?)
}

fn count_chat_messages(model_name: &str, messages: JsValue) -> Result<JsValue, JsError> {
    let messages = serde_wasm_bindgen::from_value(messages)?;
    match crate::try_count_chat_messages(ModelName::from_js(model_name), messages) {
        Ok(result) => Ok(serde_wasm_bindgen::to_value(&result)?),
        Err(error) => tokenizer_error(model_name, error),
    }
}

/// Try to count messages locally from a JSON string body.
///
/// Always returns a `{ token_count }` object: models with an exact tokenizer are
/// counted precisely, and every other model returns a structured error.
/// Tokenizer failures are returned as `{ error, message, model_name, provider }`.
#[wasm_bindgen(js_name = st_dot_count_messages_json)]
pub fn st_dot_count_messages_json_wasm(
    model_name: &str,
    body_json: &str,
) -> Result<JsValue, JsError> {
    match crate::try_count_messages(ModelName::from_js(model_name), body_json) {
        Ok(result) => Ok(serde_wasm_bindgen::to_value(&result)?),
        Err(error) => tokenizer_error(model_name, error),
    }
}

/// Local replacement for SillyTavern's `getTokenCountAsync` request path.
///
/// Deserializes `messages` (a JavaScript array of message objects) directly into
/// Rust, so the hot prompt-construction path avoids a `JSON.stringify` on the JS
/// side and a re-parse on the Rust side. Returns `{ token_count }` on success or
/// a structured error object when the local tokenizer cannot serve the request.
#[wasm_bindgen(js_name = st_dot_get_token_count_async)]
pub fn st_dot_get_token_count_async_wasm(
    model_name: &str,
    messages: JsValue,
) -> Result<JsValue, JsError> {
    count_chat_messages(model_name, messages)
}

/// Local replacement for `TokenHandler.prototype.countAsync`.
#[wasm_bindgen(js_name = st_dot_token_handler_count_async)]
pub fn st_dot_token_handler_count_async_wasm(
    model_name: &str,
    messages: JsValue,
) -> Result<JsValue, JsError> {
    count_chat_messages(model_name, messages)
}

/// Initialize tokenizer data for a provider previously requested by Rust.
#[wasm_bindgen(js_name = st_dot_init_tokenizer_provider)]
pub fn st_dot_init_tokenizer_provider_wasm(
    provider: &str,
    tokenizer_json: &str,
) -> Result<(), JsError> {
    crate::init_tokenizer_provider(provider, tokenizer_json)
        .map_err(|error| JsError::new(&error.to_string()))
}

/// Local replacement for SillyTavern's `getTextTokens` request path.
///
/// Returns `{ ids, count, chunks }` on success or a structured error object when
/// the local tokenizer cannot serve the request.
#[wasm_bindgen(js_name = st_dot_get_text_tokens)]
pub fn st_dot_get_text_tokens_wasm(model_name: &str, body_json: &str) -> Result<JsValue, JsError> {
    match crate::try_encode_request(ModelName::from_js(model_name), body_json) {
        Ok(result) => Ok(serde_wasm_bindgen::to_value(&result)?),
        Err(error) => tokenizer_error(model_name, error),
    }
}

/// Try to encode text locally.
///
/// Returns `{ ids, count, chunks }` on success or a structured error object when
/// the local tokenizer cannot serve the request.
#[wasm_bindgen(js_name = st_dot_encode_text)]
pub fn st_dot_encode_text_wasm(model_name: &str, text: &str) -> Result<JsValue, JsError> {
    match crate::try_encode_text(ModelName::from_js(model_name), text) {
        Ok(result) => Ok(serde_wasm_bindgen::to_value(&result)?),
        Err(error) => tokenizer_error(model_name, error),
    }
}

/// Runs on module init for readable panic messages in the browser console.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}
