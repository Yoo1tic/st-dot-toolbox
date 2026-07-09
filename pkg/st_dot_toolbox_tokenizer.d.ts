/* tslint:disable */
/* eslint-disable */

/**
 * Try to count messages locally from a JSON string body.
 *
 * Always returns a `{ token_count }` object: models with an exact tokenizer are
 * counted precisely, and every other model returns a structured error.
 * Tokenizer failures are returned as `{ error, message, model_name, provider }`.
 */
export function st_dot_count_messages_json(model_name: string, body_json: string): any;

/**
 * Try to encode text locally.
 *
 * Returns `{ ids, count, chunks }` on success or a structured error object when
 * the local tokenizer cannot serve the request.
 */
export function st_dot_encode_text(model_name: string, text: string): any;

/**
 * Local replacement for SillyTavern's `getTextTokens` request path.
 *
 * Returns `{ ids, count, chunks }` on success or a structured error object when
 * the local tokenizer cannot serve the request.
 */
export function st_dot_get_text_tokens(model_name: string, body_json: string): any;

/**
 * Local replacement for SillyTavern's `getTokenCountAsync` request path.
 *
 * Deserializes `messages` (a JavaScript array of message objects) directly into
 * Rust, so the hot prompt-construction path avoids a `JSON.stringify` on the JS
 * side and a re-parse on the Rust side. Returns `{ token_count }` on success or
 * a structured error object when the local tokenizer cannot serve the request.
 */
export function st_dot_get_token_count_async(model_name: string, messages: any): any;

/**
 * Initialize tokenizer data for a provider previously requested by Rust.
 */
export function st_dot_init_tokenizer_provider(provider: string, bundle_tar_gz: Uint8Array): void;

/**
 * Local replacement for `TokenHandler.prototype.countAsync`.
 */
export function st_dot_token_handler_count_async(model_name: string, messages: any): any;

/**
 * Runs on module init for readable panic messages in the browser console.
 */
export function start(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly st_dot_count_messages_json: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly st_dot_encode_text: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly st_dot_get_text_tokens: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly st_dot_get_token_count_async: (a: number, b: number, c: any) => [number, number, number];
    readonly st_dot_init_tokenizer_provider: (a: number, b: number, c: number, d: number) => [number, number];
    readonly start: () => void;
    readonly st_dot_token_handler_count_async: (a: number, b: number, c: any) => [number, number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
