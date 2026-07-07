//! WASM boundary for the SillyTavern toolbox extension.
//!
//! Feature crates own the actual logic. This crate only exposes stable
//! JavaScript-callable functions and converts Rust results into JS values.

use st_dot_toolbox_tokenizer::{self as tokenizer, ModelName, TokenizerError};
use wasm_bindgen::prelude::*;

fn tokenizer_error(error: TokenizerError) -> Result<JsValue, JsError> {
    Ok(serde_wasm_bindgen::to_value(&error.body())?)
}

/// Try to count messages locally from a JSON string body.
///
/// Always returns a `{ token_count }` object: models with an exact tokenizer are
/// counted precisely, and every other model falls back to a heuristic estimate.
/// Tokenizer failures are returned as `{ error_type, message }`.
#[wasm_bindgen]
pub fn try_count_messages(model_name: &str, body_json: &str) -> Result<JsValue, JsError> {
    match tokenizer::try_count_messages(ModelName::from_js(model_name), body_json) {
        Ok(result) => Ok(serde_wasm_bindgen::to_value(&result)?),
        Err(error) => tokenizer_error(error),
    }
}

/// Try to count a live array of chat-message objects locally.
///
/// Deserializes `messages` (a JavaScript array of message objects) directly into
/// Rust, so the hot prompt-construction path avoids a `JSON.stringify` on the JS
/// side and a re-parse on the Rust side. Returns `{ token_count }` on success or
/// `{ error_type, message }` when the local tokenizer cannot serve the request.
#[wasm_bindgen]
pub fn try_count_chat_messages(model_name: &str, messages: JsValue) -> Result<JsValue, JsError> {
    let messages = serde_wasm_bindgen::from_value(messages)?;
    match tokenizer::try_count_chat_messages(ModelName::from_js(model_name), messages) {
        Ok(result) => Ok(serde_wasm_bindgen::to_value(&result)?),
        Err(error) => tokenizer_error(error),
    }
}

/// Return the tokenizer asset id Rust needs for `model`, or an empty string if none.
#[wasm_bindgen]
pub fn tokenizer_asset_for_model(model_name: &str) -> JsValue {
    JsValue::from_str(
        tokenizer::required_tokenizer_asset(ModelName::from_js(model_name))
            .map(|asset| asset.id())
            .unwrap_or(""),
    )
}

/// Initialize a tokenizer asset previously requested by Rust.
#[wasm_bindgen]
pub fn init_tokenizer_asset(asset_id: &str, tokenizer_json: &str) -> Result<(), JsError> {
    tokenizer::init_tokenizer_asset(asset_id, tokenizer_json)
        .map_err(|error| JsError::new(&error.to_string()))
}

/// Try to encode an endpoint request body locally.
///
/// Returns `{ ids, count, chunks }` on success or `{ error_type, message }` when
/// the local tokenizer cannot serve the request.
#[wasm_bindgen]
pub fn try_encode_request(model_name: &str, body_json: &str) -> Result<JsValue, JsError> {
    match tokenizer::try_encode_request(ModelName::from_js(model_name), body_json) {
        Ok(result) => Ok(serde_wasm_bindgen::to_value(&result)?),
        Err(error) => tokenizer_error(error),
    }
}

/// Try to encode text locally.
///
/// Returns `{ ids, count, chunks }` on success or `{ error_type, message }` when
/// the local tokenizer cannot serve the request.
#[wasm_bindgen]
pub fn try_encode_text(model_name: &str, text: &str) -> Result<JsValue, JsError> {
    match tokenizer::try_encode_text(ModelName::from_js(model_name), text) {
        Ok(result) => Ok(serde_wasm_bindgen::to_value(&result)?),
        Err(error) => tokenizer_error(error),
    }
}

/// Runs on module init for readable panic messages in the browser console.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}
