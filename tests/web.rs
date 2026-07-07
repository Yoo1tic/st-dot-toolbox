//! Smoke tests for the wasm-bindgen exports (run with `wasm-pack test --headless`).
//! The tokenization logic itself is covered by native `cargo test` in src/lib.rs;
//! these only verify the JS boundary: JsValue results and JsError conversion.

#![cfg(target_arch = "wasm32")]

use st_dot_toolbox::{
    tokenizer_asset_for_model, try_count_chat_messages, try_count_messages, try_encode_request,
    try_encode_text,
};
use st_dot_toolbox_tokenizer::CountResult;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn count_messages_returns_a_count() {
    match try_count_messages("gpt-4o", r#"[{"role":"user","content":"hi"}]"#) {
        Ok(value) if value.is_object() => {
            let result: CountResult =
                serde_wasm_bindgen::from_value(value).expect("count result should deserialize");
            assert!(result.token_count > 0);
        }
        Ok(value) => panic!("gpt-4o should return a count object, got {value:?}"),
        Err(error) => panic!("gpt-4o count should succeed: {error:?}"),
    }
}

#[wasm_bindgen_test]
fn count_chat_messages_counts_a_live_array() {
    let messages = serde_wasm_bindgen::to_value(&serde_json::json!([
        { "role": "user", "content": "hi" }
    ]))
    .expect("messages should serialize to a JS array");

    match try_count_chat_messages("gpt-4o", messages) {
        Ok(value) if value.is_object() => {
            let result: CountResult =
                serde_wasm_bindgen::from_value(value).expect("count result should deserialize");
            assert!(result.token_count > 0);
        }
        Ok(value) => panic!("gpt-4o should return a count object, got {value:?}"),
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
        try_encode_text("qwen2", "hello"),
        Ok(value) if value.is_null()
    ));
}

#[wasm_bindgen_test]
fn unsupported_models_fall_back_to_an_estimate() {
    match try_count_messages("claude", r#"[{"role":"user","content":"hi"}]"#) {
        Ok(value) if value.is_object() => {
            let result: CountResult =
                serde_wasm_bindgen::from_value(value).expect("count result should deserialize");
            assert!(result.token_count > 0);
        }
        Ok(value) => panic!("claude should fall back to an estimate object, got {value:?}"),
        Err(error) => panic!("claude count should succeed via fallback: {error:?}"),
    }
}

#[wasm_bindgen_test]
fn encode_request_returns_a_js_object() {
    match try_encode_request("gpt-4o", r#"{"text":"hello world"}"#) {
        Ok(value) => assert!(value.is_object()),
        Err(error) => panic!("gpt-4o encode should succeed: {error:?}"),
    }
}

#[wasm_bindgen_test]
fn encode_text_returns_a_js_object() {
    match try_encode_text("gpt-4o", "hello world") {
        Ok(value) => assert!(value.is_object()),
        Err(error) => panic!("gpt-4o encode should succeed: {error:?}"),
    }
}

#[wasm_bindgen_test]
fn tokenizer_asset_selection_lives_in_rust() {
    assert_eq!(
        tokenizer_asset_for_model("gemini-2.5-pro")
            .as_string()
            .as_deref(),
        Some("gemma")
    );
    assert!(tokenizer_asset_for_model("gpt-4o").is_null());
}
