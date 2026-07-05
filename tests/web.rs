//! Smoke tests for the wasm-bindgen exports (run with `wasm-pack test --headless`).
//! The tokenization logic itself is covered by native `cargo test` in src/lib.rs;
//! these only verify the JS boundary: JsValue results and JsError conversion.

#![cfg(target_arch = "wasm32")]

use st_dot_toolbox::{try_count_messages, try_encode_text};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn count_messages_returns_a_count() {
    match try_count_messages("gpt-4o", r#"[{"role":"user","content":"hi"}]"#) {
        Ok(Some(count)) => assert!(count > 0),
        Ok(None) => panic!("gpt-4o should support local chat counting"),
        Err(error) => panic!("gpt-4o count should succeed: {error:?}"),
    }
}

#[wasm_bindgen_test]
fn count_messages_rejects_invalid_json() {
    assert!(try_count_messages("gpt-4o", "not json").is_err());
}

#[wasm_bindgen_test]
fn unsupported_models_are_reported() {
    assert!(matches!(
        try_count_messages("claude", r#"[{"role":"user","content":"hi"}]"#),
        Ok(None)
    ));
    assert!(matches!(
        try_encode_text("qwen2", "hello"),
        Ok(value) if value.is_null()
    ));
}

#[wasm_bindgen_test]
fn encode_text_returns_a_js_object() {
    match try_encode_text("gpt-4o", "hello world") {
        Ok(value) => assert!(value.is_object()),
        Err(error) => panic!("gpt-4o encode should succeed: {error:?}"),
    }
}
