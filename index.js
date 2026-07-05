// SillyTavern third-party extension entry point.
//
// Replaces the server-side `/api/tokenizers/openai/{count,encode}` round-trips
// with a local WASM BPE tokenizer, so token counts update instantly without
// hitting the backend. Each model is tokenized with its real encoding
// (deepseek → deepseek_v3, gpt-4o → o200k, …); see src/lib.rs.

import init, { count_messages, encode_text } from './pkg/st_dot_toolbox.js';

const LOG = '[st-dot-toolbox]';

// Matches e.g. /api/tokenizers/openai/count?model=deepseek-v4-pro
const TOKENIZER_RE = /\/api\/tokenizers\/openai\/(count|encode)\b/;

/** Parse a request URL into `{ op, model }`, or `null` if we don't handle it. */
function parseTarget(rawUrl) {
    let url;
    try {
        // rawUrl may be relative ("/api/...") — resolve against the page origin.
        url = new URL(rawUrl, window.location.origin);
    } catch {
        return null;
    }
    const match = TOKENIZER_RE.exec(url.pathname);
    return match ? { op: match[1], model: url.searchParams.get('model') ?? '' } : null;
}

/** Compute the exact JSON payload ST's endpoint would return. */
function computeResponse({ op, model }, bodyText) {
    if (op === 'count') {
        return { token_count: count_messages(model, bodyText) };
    }
    const { text = '' } = JSON.parse(bodyText); // /encode body is { text }
    return encode_text(model, text); // { ids, count, chunks }
}

function patchFetch() {
    const originalFetch = window.fetch.bind(window);

    window.fetch = async function (input, options) {
        try {
            const rawUrl = typeof input === 'string' ? input : input?.url;
            const target = rawUrl && parseTarget(rawUrl);
            if (target) {
                // Body lives on options, or on a Request passed as input.
                const body = options?.body ?? (input instanceof Request ? await input.clone().text() : '');
                // ST sends JSON strings; anything else (Blob, stream…) goes to the backend untouched.
                if (typeof body === 'string') {
                    return Response.json(computeResponse(target, body));
                }
            }
        } catch (e) {
            console.error(LOG, 'fetch intercept failed, falling back to backend:', e);
        }
        return originalFetch(input, options);
    };
}

function patchJQuery($) {
    const originalAjax = $.ajax;

    $.ajax = function (options) {
        try {
            const target = options?.url && parseTarget(options.url);
            if (target) {
                const bodyText =
                    typeof options.data === 'string' ? options.data : JSON.stringify(options.data ?? '');

                const payload = computeResponse(target, bodyText);
                options.success?.(payload);
                return $.Deferred().resolve(payload).promise();
            }
        } catch (e) {
            console.error(LOG, '$.ajax intercept failed, falling back to backend:', e);
        }
        return originalAjax.apply(this, arguments);
    };
}

(async function boot() {
    try {
        await init(); // loads pkg/st_dot_toolbox_bg.wasm
        patchFetch();
        if (window.jQuery) patchJQuery(window.jQuery);
        console.log(`${LOG} local WASM tokenizer active (fetch${window.jQuery ? ' + $.ajax' : ''}).`);
    } catch (e) {
        console.error(LOG, 'init failed, leaving backend tokenizer in place:', e);
        window.toastr?.error('WASM tokenizer 初始化失败', 'st-dot-toolbox');
    }
})();
