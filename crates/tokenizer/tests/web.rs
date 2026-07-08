//! Smoke tests for the wasm-bindgen exports (run with `wasm-pack test --headless`).
//! The tokenization logic itself is covered by native `cargo test`; these only
//! verify the JS boundary: JsValue results and JsError conversion.

#![cfg(target_arch = "wasm32")]

use st_dot_toolbox_tokenizer::{
    CountResult, st_dot_count_messages_json_wasm, st_dot_encode_text_wasm,
    st_dot_get_text_tokens_wasm, st_dot_get_token_count_async_wasm,
    st_dot_token_handler_count_async_wasm,
};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn count_messages_returns_a_count() {
    match st_dot_count_messages_json_wasm("gpt-4o", r#"[{"role":"user","content":"hi"}]"#) {
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

    match st_dot_get_token_count_async_wasm("gpt-4o", messages) {
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
fn token_handler_count_async_counts_a_live_array() {
    let messages = serde_wasm_bindgen::to_value(&serde_json::json!([
        { "role": "user", "content": "hi" }
    ]))
    .expect("messages should serialize to a JS array");

    match st_dot_token_handler_count_async_wasm("gpt-4o", messages) {
        Ok(value) if value.is_object() => {
            let result: CountResult =
                serde_wasm_bindgen::from_value(value).expect("count result should deserialize");
            assert!(result.token_count > 0);
        }
        Ok(value) => panic!("TokenHandler countAsync should return a count object, got {value:?}"),
        Err(error) => panic!("TokenHandler countAsync should succeed: {error:?}"),
    }
}

#[wasm_bindgen_test]
fn count_messages_rejects_invalid_json() {
    match st_dot_count_messages_json_wasm("gpt-4o", "not json") {
        Ok(value) if value.is_object() => {
            let error: serde_json::Value =
                serde_wasm_bindgen::from_value(value).expect("error should deserialize");
            assert_eq!(error["error"], "Json");
            assert_eq!(error["model_name"], "gpt-4o");
            assert_eq!(error["provider"], "");
        }
        Ok(value) => panic!("invalid JSON should return an error object, got {value:?}"),
        Err(error) => panic!("invalid JSON should not throw: {error:?}"),
    }
}

#[wasm_bindgen_test]
fn unsupported_models_are_reported() {
    match st_dot_encode_text_wasm("qwen2", "hello") {
        Ok(value) if value.is_object() => {
            let error: serde_json::Value =
                serde_wasm_bindgen::from_value(value).expect("error should deserialize");
            assert_eq!(error["error"], "Unsupported");
            assert_eq!(error["model_name"], "qwen2");
            assert_eq!(error["provider"], "");
        }
        Ok(value) => panic!("unsupported model should return an error object, got {value:?}"),
        Err(error) => panic!("unsupported model should not throw: {error:?}"),
    }
}

#[wasm_bindgen_test]
fn unsupported_count_models_are_reported() {
    match st_dot_count_messages_json_wasm("claude", r#"[{"role":"user","content":"hi"}]"#) {
        Ok(value) if value.is_object() => {
            let error: serde_json::Value =
                serde_wasm_bindgen::from_value(value).expect("error should deserialize");
            assert_eq!(error["error"], "Unsupported");
            assert_eq!(error["model_name"], "claude");
            assert_eq!(error["provider"], "");
        }
        Ok(value) => panic!("unsupported count model should return an error object, got {value:?}"),
        Err(error) => panic!("unsupported count model should not throw: {error:?}"),
    }
}

#[wasm_bindgen_test]
fn uninitialized_provider_errors_include_model_and_provider() {
    let messages = serde_wasm_bindgen::to_value(&serde_json::json!([
        { "role": "user", "content": "hi" }
    ]))
    .expect("messages should serialize to a JS array");

    match st_dot_get_token_count_async_wasm("gemini-2.5-pro", messages) {
        Ok(value) if value.is_object() => {
            let error: serde_json::Value =
                serde_wasm_bindgen::from_value(value).expect("error should deserialize");
            assert_eq!(error["error"], "UnInitialized");
            assert_eq!(error["model_name"], "gemini-2.5-pro");
            assert_eq!(error["provider"], "gemma");
        }
        Ok(value) => panic!("uninitialized Gemma should return an error object, got {value:?}"),
        Err(error) => panic!("uninitialized Gemma should not throw: {error:?}"),
    }
}

#[wasm_bindgen_test]
fn encode_request_returns_a_js_object() {
    match st_dot_get_text_tokens_wasm("gpt-4o", r#"{"text":"hello world"}"#) {
        Ok(value) => assert!(value.is_object()),
        Err(error) => panic!("gpt-4o encode should succeed: {error:?}"),
    }
}

#[wasm_bindgen_test]
fn encode_text_returns_a_js_object() {
    match st_dot_encode_text_wasm("gpt-4o", "hello world") {
        Ok(value) => assert!(value.is_object()),
        Err(error) => panic!("gpt-4o encode should succeed: {error:?}"),
    }
}
