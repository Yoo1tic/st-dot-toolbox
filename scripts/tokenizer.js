// Local WASM tokenizer feature for SillyTavern OpenAI/tiktoken models.

import init, {
	st_dot_get_text_tokens,
	st_dot_get_token_count_async,
	st_dot_init_tokenizer_provider,
	st_dot_token_handler_count_async,
} from "../pkg/st_dot_toolbox_tokenizer.js";
import { createLogger, setLogLevel } from "./logger.js";
import { patchOnce } from "./patching.js";

const logger = createLogger("tokenizer");
const UNKNOWN_LOG_VALUE = "unknown";

function logString(value) {
	return JSON.stringify(String(value || UNKNOWN_LOG_VALUE));
}

function stringField(value, field) {
	return value &&
		typeof value === "object" &&
		typeof value[field] === "string"
		? value[field]
		: "";
}

function logModelName(model, payload = null) {
	return stringField(payload, "model_name") || model || UNKNOWN_LOG_VALUE;
}

function logProvider(payload = null, fallbackProvider = "") {
	return (
		stringField(payload, "label") ||
		stringField(payload, "provider") ||
		fallbackProvider ||
		UNKNOWN_LOG_VALUE
	);
}

function logTokenResult(op, payload) {
	const tokenCount =
		typeof payload === "number"
			? payload
			: op === "encode"
				? payload?.count
				: payload?.token_count;

	return Number.isFinite(tokenCount) ? `${tokenCount} tokens` : UNKNOWN_LOG_VALUE;
}

/** Logs a completed local tokenizer request. */
function logSuccess(path, op, model, payload) {
	logger.info(
		`${op} success: path=${path}; model=${logString(logModelName(model, payload))}; provider=${logProvider(payload)}; result=${logTokenResult(op, payload)}.`,
	);
}

/** Logs why the local tokenizer could not handle a request. */
function logFailure(path, op, model, error = null, reasonOverride = "", level = "info") {
	const reason = reasonOverride || error?.message || "";
	const detail = reason ? `; reason=${logString(reason)}` : "";
	logger[level](
		`${op} failure: path=${path}; model=${logString(logModelName(model, error))}; provider=${logProvider(error)}${detail}.`,
	);
}

/** Logs a completed backend fallback request. */
function logFallback(path, op, model, payload, context = null) {
	const provider = logProvider(payload, logProvider(context, ""));
	logger.info(
		`${op} fallback: path=${path}; model=${logString(logModelName(model, payload))}; provider=${provider}; result=${logTokenResult(op, payload)}.`,
	);
}

function observeAjaxFallback(jqXHR, { op, model, error }) {
	if (!jqXHR || typeof jqXHR.done !== "function") {
		return;
	}

	jqXHR.done((payload) => {
		logFallback("ajax", op, model, payload, error);
	});

	if (typeof jqXHR.fail === "function") {
		jqXHR.fail((_jqXHR, textStatus, thrownError) => {
			logFailure(
				"ajax",
				op,
				model,
				error,
				`backend fallback failed: ${thrownError?.message || thrownError || textStatus}`,
				"warn",
			);
		});
	}
}

async function countWithBackendFallback(
	originalCountAsync,
	tokenHandler,
	messages,
	full,
	type,
	model,
	error,
) {
	try {
		const tokenCount = await originalCountAsync.call(
			tokenHandler,
			messages,
			full,
			type,
		);
		logFallback("fast-path", "count", model, tokenCount, error);
		return tokenCount;
	} catch (fallbackError) {
		logFailure(
			"fast-path",
			"count",
			model,
			error,
			`backend fallback failed: ${fallbackError?.message || fallbackError}`,
			"warn",
		);
		throw fallbackError;
	}
}

function makeTokenizerError(errorName, message, modelName = "", provider = "") {
	return { error: errorName, message, model_name: modelName, provider };
}

function isTokenizerError(value) {
	return Boolean(
		value &&
			typeof value === "object" &&
			typeof value.error === "string" &&
			typeof value.message === "string" &&
			typeof value.model_name === "string" &&
			typeof value.provider === "string",
	);
}

function isUninitializedTokenizerError(value) {
	return isTokenizerError(value) && value.error === "UnInitialized";
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

const TOKENIZER_PROVIDER_ASSET_URLS = Object.freeze({
	gemma: new URL("../assets/gemma/tokenizer.json.gz", import.meta.url),
});
const tokenizerProviderPromises = new Map();

function tokenizerProviderAssetUrl(provider) {
	const url = TOKENIZER_PROVIDER_ASSET_URLS[provider];
	if (!url) throw new Error(`unknown tokenizer provider: ${provider}`);
	return url;
}

async function fetchCompressedTokenizerJson(provider) {
	const response = await fetch(tokenizerProviderAssetUrl(provider), {
		cache: "force-cache",
	});
	if (!response.ok) {
		throw new Error(
			`failed to fetch compressed tokenizer provider ${provider}: ${response.status} ${response.statusText}`,
		);
	}
	if (!response.body || typeof DecompressionStream !== "function") {
		throw new Error("gzip decompression is not supported by this browser");
	}

	const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
	return new Response(stream).text();
}

function ensureTokenizerProvider(provider) {
	if (!provider) return false;

	if (!tokenizerProviderPromises.has(provider)) {
		const promise = fetchCompressedTokenizerJson(provider)
			.then((tokenizerJson) => {
				st_dot_init_tokenizer_provider(provider, tokenizerJson);
				logger.info(`tokenizer provider initialized: ${provider}.`);
				return true;
			})
			.catch((error) => {
				tokenizerProviderPromises.delete(provider);
				throw error;
			});
		tokenizerProviderPromises.set(provider, promise);
	}

	return tokenizerProviderPromises.get(provider);
}

function tokenizerProviderFromError(error) {
	return isUninitializedTokenizerError(error) &&
		typeof error.provider === "string" &&
		error.provider
		? error.provider
		: "";
}

function warmTokenizerProviderFromError(error) {
	const provider = tokenizerProviderFromError(error);
	if (provider) {
		void ensureTokenizerProvider(provider).catch((error) => {
			logger.warn(`tokenizer provider preload failed: ${provider}`, error);
		});
	}
}

/**
 * Encode a `{ text }` body locally. An unwarmed provider-backed model returns a
 * structured `UnInitialized` error for one call; the caller starts loading that
 * provider and falls back to the backend until it finishes.
 */
function encodeLocally(model, bodyText) {
	return st_dot_get_text_tokens(model, bodyText);
}

/**
 * Count a `[{ role, content }]` chat-message body locally.
 *
 * Rust returns an exact count when a provider is ready, or a structured error
 * when the local tokenizer cannot serve the request. The caller then falls back
 * to SillyTavern's native ajax path while any returned provider preload continues.
 */
function countLocally(model, bodyText) {
	let messages;
	try {
		messages = JSON.parse(bodyText);
	} catch (error) {
		return makeTokenizerError("Json", error.message, model);
	}
	if (!Array.isArray(messages)) {
		messages = [messages];
	}

	return st_dot_get_token_count_async(model, messages);
}

/**
 * Serves a classified tokenizer request locally, returning either a success
 * payload or a structured tokenizer error.
 *
 * Count only returns an error while a provider-backed tokenizer is still loading.
 * Encode can also error when no local encoder can produce ids/chunks for a model.
 */
function serveTokenizerLocally({ op, model }, bodyText, path) {
	if (op === "encode") {
		const payload = encodeLocally(model, bodyText);
		if (!isTokenizerError(payload)) {
			logSuccess(path, "encode", model, payload);
		}
		return payload;
	}

	const payload = countLocally(model, bodyText);
	if (!isTokenizerError(payload)) {
		logSuccess(path, "count", model, payload);
	}
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
			let fallback = null;
			try {
				const request = getAjaxRequest(options, settings);
				const parsed = request?.url
					? parseTokenizerRequest(request.url, request.type ?? request.method)
					: null;
				if (parsed) {
					const payload = serveTokenizerLocally(
						parsed,
						readAjaxBody(request),
						"ajax",
					);
					if (isTokenizerError(payload)) {
						if (isUninitializedTokenizerError(payload)) {
							warmTokenizerProviderFromError(payload);
						}
						logFailure("ajax", parsed.op, parsed.model, payload);
						fallback = { op: parsed.op, model: parsed.model, error: payload };
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

			const jqXHR = originalAjax.apply(this, arguments);
			if (fallback) observeAjaxFallback(jqXHR, fallback);
			return jqXHR;
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
						let model = "";
						try {
							model = getTokenizerModel();
							const messagesArray = Array.isArray(messages)
								? messages
								: [messages];

							// Pass live message objects straight to Rust; the WASM
							// boundary deserializes them without a JSON string hop.
							const payload = st_dot_token_handler_count_async(
								model,
								messagesArray,
							);
							if (isTokenizerError(payload)) {
								if (isUninitializedTokenizerError(payload)) {
									warmTokenizerProviderFromError(payload);
								}
								logFailure("fast-path", "count", model, payload);
								return countWithBackendFallback(
									originalCountAsync,
									this,
									messages,
									full,
									type,
									model,
									payload,
								);
							}

							const { token_count } = payload;
							logSuccess("fast-path", "count", model, payload);

							// ST often calls countAsync without a bucket type. The original method
							// creates a NaN `undefined` bucket; skip that bookkeeping noise here.
							if (type) {
								this.counts[type] += token_count;
							}

							return token_count;
						} catch (error) {
							const fallbackError = makeTokenizerError(
								"JavaScript",
								error?.message || String(error),
								model,
							);
							logFailure("fast-path", "count", model, fallbackError, "", "warn");
							return countWithBackendFallback(
								originalCountAsync,
								this,
								messages,
								full,
								type,
								model,
								fallbackError,
							);
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
