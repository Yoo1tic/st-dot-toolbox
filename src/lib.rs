//! WASM boundary for the SillyTavern toolbox extension.
//!
//! Feature crates own the actual logic. This crate only exposes stable
//! JavaScript-callable functions and converts Rust results into JS values.

use st_dot_toolbox_tokenizer as tokenizer;
use wasm_bindgen::prelude::*;

/// Try to count messages locally.
///
/// Returns a token count on success, `undefined` for unsupported models, and
/// throws only for real errors such as invalid JSON.
#[wasm_bindgen]
pub fn try_count_messages(model: &str, body_json: &str) -> Result<Option<usize>, JsError> {
    tokenizer::try_count_messages(model, body_json)
        .map_err(|error| JsError::new(&error.to_string()))
}

/// Try to encode text locally.
///
/// Returns `{ ids, count, chunks }` on success, `null` for unsupported models,
/// and throws only for real tokenizer/serialization errors.
#[wasm_bindgen]
pub fn try_encode_text(model: &str, text: &str) -> Result<JsValue, JsError> {
    match tokenizer::try_encode_text(model, text)
        .map_err(|error| JsError::new(&error.to_string()))?
    {
        Some(result) => Ok(serde_wasm_bindgen::to_value(&result)?),
        None => Ok(JsValue::NULL),
    }
}

/// Runs on module init for readable panic messages in the browser console.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}
