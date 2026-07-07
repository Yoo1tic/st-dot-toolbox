// Local WASM tokenizer feature for SillyTavern OpenAI/tiktoken models.

import init, {
	init_tokenizer_asset,
	tokenizer_asset_for_model,
	try_count_chat_messages,
	try_encode_request,
} from "../pkg/st_dot_toolbox.js";
import { createLogger, setLogLevel } from "./logger.js";
import { patchOnce } from "./patching.js";

const logger = createLogger("tokenizer");

/** Logs, at debug, a locally served request with its model and token count. */
function logSuccess(op, model, tokenCount) {
	logger.debug(`${op} ok: "${model || "unknown"}" → ${tokenCount} tokens.`);
}

/**
 * Logs, at debug, that a request was delegated to SillyTavern's backend.
 *
 * Unsupported models fall through to the original path by design; this makes the
 * reason visible when debug is enabled, without affecting the info stream.
 */
function logFallback(op, model) {
	logger.debug(
		`"${model || "unknown"}" not handled locally (${op}); using backend.`,
	);
}

// Counting is served by directly replacing `TokenHandler.prototype.countAsync`
// (see installTokenHandlerFastPath), so no count request ever reaches the wire.
// Encoding goes through `getTextTokens`, an ES-module export we cannot replace,
// so its `POST /api/tokenizers/openai/encode` is intercepted at `$.ajax` instead.
const ENCODE_ENDPOINT = "/api/tokenizers/openai/encode";
const TOKENIZER_METHOD = "POST";

let installPromise = null;

/**
 * Installs the tokenizer feature once.
 *
 * Supported OpenAI/tiktoken models are handled locally through WASM. Unsupported
 * tokenizer families fall back to SillyTavern's original handlers.
 */
export function installTokenizer() {
	if (!installPromise) {
		installPromise = installTokenizerInner().catch((error) => {
			installPromise = null;
			throw error;
		});
	}

	return installPromise;
}

async function installTokenizerInner() {
	await init();
	installEncodeInterceptor();
	await installTokenHandlerFastPath();
	// Live verbosity toggle for field debugging, e.g. __stDotToolbox.setLogLevel("debug").
	window.__stDotToolbox = { setLogLevel };
	logger.info("local WASM tokenizer active.");
}

function normalizePathname(pathname) {
	return pathname.length > 1 && pathname.endsWith("/")
		? pathname.slice(0, -1)
		: pathname;
}

function normalizeMethod(method) {
	return String(method ?? "GET").toUpperCase();
}

/**
 * Returns the `?model=` for a local encode request, or `null` when the request
 * is not the OpenAI encode endpoint we serve.
 */
function parseEncodeModel(rawUrl, method) {
	if (normalizeMethod(method) !== TOKENIZER_METHOD) {
		return null;
	}

	let url;
	try {
		// rawUrl may be relative ("/api/...") — resolve against the page origin.
		url = new URL(rawUrl, window.location.origin);
	} catch {
		return null;
	}

	if (url.origin !== window.location.origin) {
		return null;
	}

	if (normalizePathname(url.pathname) !== ENCODE_ENDPOINT) {
		return null;
	}

	return url.searchParams.get("model") ?? "";
}

const TOKENIZER_ASSET_URLS = Object.freeze({
	gemma: new URL("../assets/gemma/tokenizer.json.gz", import.meta.url),
});
const tokenizerAssetPromises = new Map();

function tokenizerAssetUrl(assetId) {
	const url = TOKENIZER_ASSET_URLS[assetId];
	if (!url) throw new Error(`unknown tokenizer asset: ${assetId}`);
	return url;
}

async function fetchCompressedTokenizerJson(assetId) {
	const response = await fetch(tokenizerAssetUrl(assetId), {
		cache: "force-cache",
	});
	if (!response.ok) {
		throw new Error(
			`failed to fetch compressed tokenizer asset ${assetId}: ${response.status} ${response.statusText}`,
		);
	}
	if (!response.body || typeof DecompressionStream !== "function") {
		throw new Error("gzip decompression is not supported by this browser");
	}

	const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
	return new Response(stream).text();
}

function ensureTokenizerAsset(model) {
	const assetId = tokenizer_asset_for_model(model);
	if (!assetId) return false;

	if (!tokenizerAssetPromises.has(assetId)) {
		const promise = fetchCompressedTokenizerJson(assetId)
			.then((tokenizerJson) => {
				init_tokenizer_asset(assetId, tokenizerJson);
				logger.info(`tokenizer asset initialized: ${assetId}.`);
				return true;
			})
			.catch((error) => {
				tokenizerAssetPromises.delete(assetId);
				throw error;
			});
		tokenizerAssetPromises.set(assetId, promise);
	}

	return tokenizerAssetPromises.get(assetId);
}

function warmTokenizerAsset(model) {
	if (tokenizer_asset_for_model(model)) {
		void ensureTokenizerAsset(model).catch((error) => {
			logger.warn("tokenizer asset preload failed:", error);
		});
	}
}

/**
 * Encode a `{ text }` body locally. The Gemma asset is warmed but cannot be
 * awaited inside jQuery's blocking `$.ajax`; an unwarmed model returns `null`
 * for one call and falls back to the backend until the asset finishes loading.
 */
function encodeLocally(model, bodyText) {
	warmTokenizerAsset(model);
	return try_encode_request(model, bodyText);
}

function getAjaxRequest(options, settings) {
	return typeof options === "string"
		? { ...(settings ?? {}), url: options }
		: options;
}

function readAjaxBody(request) {
	return typeof request.data === "string"
		? request.data
		: JSON.stringify(request.data ?? "");
}

function resolveAjaxLocally($, request, payload) {
	const deferred = $.Deferred();
	const jqXHR = deferred.promise();
	const context = request.context ?? request;

	request.success?.call(context, payload, "success", jqXHR);
	deferred.resolve(payload, "success", jqXHR);
	request.complete?.call(context, jqXHR, "success");

	return jqXHR;
}

function installEncodeInterceptor() {
	// `getTextTokens` reaches the server exclusively through `jQuery.ajax`
	// (getTextTokensFromServer), so intercepting `$.ajax` covers every encode.
	if (!window.jQuery) {
		logger.warn("jQuery unavailable; encode requests will use the backend.");
		return;
	}

	const $ = window.jQuery;
	const installed = patchOnce($, "tokenizer.jquery-ajax", () => {
		const originalAjax = $.ajax;

		$.ajax = function stDotToolboxAjax(options, settings) {
			try {
				const request = getAjaxRequest(options, settings);
				const model = request?.url
					? parseEncodeModel(request.url, request.type ?? request.method)
					: null;
				if (model !== null) {
					const payload = encodeLocally(model, readAjaxBody(request));
					if (payload) {
						logSuccess("encode", model, payload.count);
						return resolveAjaxLocally($, request, payload);
					}
					logFallback("encode", model);
				}
			} catch (error) {
				logger.error(
					"$.ajax intercept failed, falling back to backend:",
					error,
				);
			}

			return originalAjax.apply(this, arguments);
		};

		return { originalAjax };
	});

	if (installed) logger.debug("jQuery encode interceptor installed.");
}

async function installTokenHandlerFastPath() {
	try {
		const { TokenHandler } = await import("/scripts/openai.js");
		const { getTokenizerModel } = await import("/scripts/tokenizers.js");

		const installed = patchOnce(
			TokenHandler.prototype,
			"tokenizer.token-handler-count-async",
			() => {
				const originalCountAsync = TokenHandler.prototype.countAsync;

				// Direct hot path for prompt construction. The WASM branch must also
				// update `this.counts[type]`; fallback delegates to the original method,
				// which performs its own counting and bookkeeping.
				TokenHandler.prototype.countAsync =
					async function stDotToolboxCountAsync(messages, full, type) {
						try {
							const model = getTokenizerModel();
							const messagesArray = Array.isArray(messages)
								? messages
								: [messages];

							// Pass live message objects straight to Rust; the WASM
							// boundary deserializes them without a JSON string hop.
							const payload = try_count_chat_messages(model, messagesArray);
							if (!payload) {
								// A miss on a Gemma model means its asset has not loaded
								// yet; warm it (a no-op for models needing no asset) so the
								// next count is exact. This call falls back meanwhile.
								warmTokenizerAsset(model);
								logFallback("count", model);
								return originalCountAsync.call(this, messages, full, type);
							}

							const { token_count } = payload;
							logSuccess("count", model, token_count);

							// ST often calls countAsync without a bucket type. The original method
							// creates a NaN `undefined` bucket; skip that bookkeeping noise here.
							if (type) {
								this.counts[type] += token_count;
							}

							return token_count;
						} catch (error) {
							logger.error(
								"TokenHandler.prototype.countAsync failed, falling back:",
								error,
							);
							return originalCountAsync.call(this, messages, full, type);
						}
					};

				return { originalCountAsync };
			},
		);

		if (installed) {
			logger.info(
				"TokenHandler.prototype.countAsync patched successfully (OAI WASM fast path).",
			);
		}
	} catch (error) {
		logger.warn(
			"failed to patch TokenHandler prototype, relying on request fallback:",
			error,
		);
	}
}
