//! Smoke tests for the wasm-bindgen exports (run with `wasm-pack test --headless`).
//! The tokenization logic itself is covered by native `cargo test` in src/lib.rs;
//! these only verify the JS boundary: JsValue results and JsError conversion.

#![cfg(target_arch = "wasm32")]

use st_dot_toolbox::{count_messages, encode_text};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn count_messages_returns_a_count() {
    let n = count_messages("gpt-4o", r#"[{"role":"user","content":"hi"}]"#).unwrap();
    assert!(n > 0);
}

#[wasm_bindgen_test]
fn count_messages_rejects_invalid_json() {
    assert!(count_messages("gpt-4o", "not json").is_err());
}

#[wasm_bindgen_test]
fn encode_text_returns_a_js_object() {
    let value = encode_text("gpt-4o", "hello world").unwrap();
    assert!(value.is_object());
}
