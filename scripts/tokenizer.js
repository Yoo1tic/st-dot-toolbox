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
function logFallback(op, model, error = null) {
	const reason = error?.message ? `: ${error.message}` : "";
	logger.debug(
		`"${model || "unknown"}" not handled locally (${op}); using backend${reason}.`,
	);
}

function makeTokenizerError(errorType, message) {
	return { error_type: errorType, message };
}

function isTokenizerError(value) {
	return Boolean(
		value &&
			typeof value === "object" &&
			typeof value.error_type === "string" &&
			typeof value.message === "string",
	);
}

function isUninitializedTokenizerError(value) {
	return isTokenizerError(value) && value.error_type === "UnInitialized";
}

// SillyTavern reaches the tokenizer server through jQuery.ajax on two endpoints:
// `getTextTokens` (an ES-module export we cannot replace) encodes via
// `POST /api/tokenizers/openai/encode`, and `getTokenCountAsync` counts via
// `POST /api/tokenizers/openai/count`. Both are intercepted at `$.ajax`.
//
// The `TokenHandler.prototype.countAsync` fast path (see installTokenHandlerFastPath)
// additionally short-circuits prompt-construction counting before a request is
// ever built; but that path is not the only counter — standalone UI counts (chat
// messages, character cards, world info budgets, …) still hit the count endpoint,
// so the `$.ajax` net is what actually keeps every count off the wire.
const ENCODE_ENDPOINT = "/api/tokenizers/openai/encode";
const COUNT_ENDPOINT = "/api/tokenizers/openai/count";
const TOKENIZER_METHOD = "POST";

/** Maps a served endpoint pathname to the tokenizer op it performs. */
const TOKENIZER_ENDPOINTS = Object.freeze({
	[ENCODE_ENDPOINT]: "encode",
	[COUNT_ENDPOINT]: "count",
});

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
	installAjaxInterceptor();
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
 * Classifies a jQuery ajax request as a tokenizer op we serve locally.
 *
 * Returns `{ op, model }` for a POST to one of the OpenAI tokenizer endpoints we
 * handle, or `null` for any other request so the caller leaves it untouched.
 */
function parseTokenizerRequest(rawUrl, method) {
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

	const op = TOKENIZER_ENDPOINTS[normalizePathname(url.pathname)];
	if (!op) {
		return null;
	}

	return { op, model: url.searchParams.get("model") ?? "" };
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
 * awaited inside jQuery's blocking `$.ajax`; an unwarmed model returns a
 * structured `UnInitialized` error for one call and falls back to the backend
 * until the asset finishes loading.
 */
function encodeLocally(model, bodyText) {
	warmTokenizerAsset(model);
	return try_encode_request(model, bodyText);
}

/**
 * Count a `[{ role, content }]` chat-message body locally.
 *
 * Rust returns an exact count when a provider is ready, or a heuristic fallback
 * estimate for unsupported models. A structured `UnInitialized` error is
 * reserved for a recognized asset tokenizer that has not loaded yet; in that one
 * case this caller falls back to SillyTavern's native ajax path while the asset
 * preload continues.
 */
function countLocally(model, bodyText) {
	warmTokenizerAsset(model);

	let messages;
	try {
		messages = JSON.parse(bodyText);
	} catch (error) {
		return makeTokenizerError("Json", error.message);
	}
	if (!Array.isArray(messages)) {
		messages = [messages];
	}

	return try_count_chat_messages(model, messages);
}

/**
 * Serves a classified tokenizer request locally, returning either a success
 * payload or a structured tokenizer error.
 *
 * Count only returns an error while an asset-backed tokenizer is still loading.
 * Encode can also error when no local encoder can produce ids/chunks for a model.
 */
function serveTokenizerLocally({ op, model }, bodyText) {
	if (op === "encode") {
		const payload = encodeLocally(model, bodyText);
		if (!isTokenizerError(payload)) logSuccess("encode", model, payload.count);
		return payload;
	}

	const payload = countLocally(model, bodyText);
	if (!isTokenizerError(payload)) logSuccess("count", model, payload.token_count);
	return payload;
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

function installAjaxInterceptor() {
	// Both `getTextTokens` (encode) and `getTokenCountAsync` (count) reach the
	// server exclusively through `jQuery.ajax`, so one `$.ajax` patch covers every
	// encode and count request regardless of which SillyTavern caller issued it.
	if (!window.jQuery) {
		logger.warn("jQuery unavailable; tokenizer requests will use the backend.");
		return;
	}

	const $ = window.jQuery;
	const installed = patchOnce($, "tokenizer.jquery-ajax", () => {
		const originalAjax = $.ajax;

		$.ajax = function stDotToolboxAjax(options, settings) {
			try {
				const request = getAjaxRequest(options, settings);
				const parsed = request?.url
					? parseTokenizerRequest(request.url, request.type ?? request.method)
					: null;
				if (parsed) {
					const payload = serveTokenizerLocally(parsed, readAjaxBody(request));
					if (isTokenizerError(payload)) {
						if (isUninitializedTokenizerError(payload)) {
							warmTokenizerAsset(parsed.model);
						}
						logFallback(parsed.op, parsed.model, payload);
					} else {
						return resolveAjaxLocally($, request, payload);
					}
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

	if (installed) logger.debug("jQuery tokenizer interceptor installed.");
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
							if (isTokenizerError(payload)) {
								if (isUninitializedTokenizerError(payload)) {
									warmTokenizerAsset(model);
								}
								logFallback("count", model, payload);
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
