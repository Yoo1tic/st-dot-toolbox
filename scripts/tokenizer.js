// Local WASM tokenizer feature for SillyTavern OpenAI/tiktoken models.

import init, {
	init_gemma_tokenizer,
	is_gemma_tokenizer_initialized,
	try_count_messages,
	try_count_text,
	try_encode_text,
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

const LOCAL_TOKENIZER_ROUTES = Object.freeze({
	count: "/api/tokenizers/openai/count",
	encode: "/api/tokenizers/openai/encode",
	gemmaEncode: "/api/tokenizers/gemma/encode",
});
const LOCAL_TOKENIZER_OPERATION_BY_PATH = new Map(
	Object.entries(LOCAL_TOKENIZER_ROUTES).map(([operation, pathname]) => [
		pathname,
		operation,
	]),
);
const TOKENIZER_METHOD = "POST";

let installPromise = null;

/**
 * Installs the tokenizer feature once.
 *
 * Supported OpenAI/tiktoken models are handled locally through WASM. Unsupported
 * tokenizer families fall back to SillyTavern's original handlers.
 */
export async function installTokenizer() {
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
	installEndpointFallbacks();
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

/** Parse a request into `{ op, model }`, or `null` if we don't handle it. */
function parseTarget(rawUrl, method) {
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

	const op = LOCAL_TOKENIZER_OPERATION_BY_PATH.get(
		normalizePathname(url.pathname),
	);
	if (!op) return null;

	return {
		op: op === "gemmaEncode" ? "encode" : op,
		model:
			op === "gemmaEncode" ? "gemma" : (url.searchParams.get("model") ?? ""),
	};
}

function isGemmaFamilyModel(model) {
	const normalized = String(model ?? "").toLowerCase();
	return (
		normalized === "gemma" ||
		normalized === "gemini" ||
		normalized.includes("gemma") ||
		normalized.includes("gemini") ||
		normalized.includes("learnlm")
	);
}

const GEMMA_TOKENIZER_URL = new URL(
	"../assets/gemma/tokenizer.json.gz",
	import.meta.url,
);
let gemmaTokenizerPromise = null;

async function ensureGemmaTokenizer(model) {
	if (!isGemmaFamilyModel(model)) return false;
	if (is_gemma_tokenizer_initialized()) return true;

	if (!gemmaTokenizerPromise) {
		gemmaTokenizerPromise = fetch(GEMMA_TOKENIZER_URL, { cache: "force-cache" })
			.then(async (response) => {
				if (!response.ok) {
					throw new Error(
						`failed to fetch compressed Gemma tokenizer: ${response.status} ${response.statusText}`,
					);
				}
				if (!response.body || typeof DecompressionStream !== "function") {
					throw new Error(
						"gzip decompression is not supported by this browser",
					);
				}

				const stream = response.body.pipeThrough(
					new DecompressionStream("gzip"),
				);
				return new Response(stream).text();
			})
			.then((tokenizerJson) => {
				init_gemma_tokenizer(tokenizerJson);
				logger.info("Gemma/Gemini tokenizer initialized.");
				return true;
			})
			.catch((error) => {
				gemmaTokenizerPromise = null;
				throw error;
			});
	}

	return gemmaTokenizerPromise;
}

function warmGemmaTokenizer(model) {
	if (isGemmaFamilyModel(model) && !is_gemma_tokenizer_initialized()) {
		void ensureGemmaTokenizer(model).catch((error) => {
			logger.warn("Gemma/Gemini tokenizer preload failed:", error);
		});
	}
}

/**
 * Rust returns `None` for unsupported models, which wasm-bindgen exposes as
 * `undefined`; in that case keep the request on SillyTavern's original path.
 */
function tryCountMessages(model, bodyText) {
	const token_count = try_count_messages(model, bodyText);
	return token_count === undefined ? null : { token_count };
}

function readEncodeTextBody(bodyText) {
	try {
		const parsed = JSON.parse(bodyText);
		return parsed?.text ?? "";
	} catch (error) {
		logger.error("failed to parse encode body as JSON:", error);
		return "";
	}
}

function readCountTextBody(bodyText) {
	try {
		const parsed = JSON.parse(bodyText);
		return typeof parsed?.text === "string" ? parsed.text : null;
	} catch (error) {
		logger.error("failed to parse count body as JSON:", error);
		return null;
	}
}

function tryCountText(model, text) {
	const token_count = try_count_text(model, text);
	return token_count === undefined ? null : { token_count };
}

function tryBuildTokenizerResponseSync({ op, model }, bodyText) {
	if (op === "count") {
		const text = readCountTextBody(bodyText);
		return (
			(text === null ? null : tryCountText(model, text)) ??
			tryCountMessages(model, bodyText)
		);
	}

	warmGemmaTokenizer(model);
	return try_encode_text(model, readEncodeTextBody(bodyText));
}

async function tryBuildTokenizerResponse({ op, model }, bodyText) {
	if (op === "count") {
		const text = readCountTextBody(bodyText);
		if (text !== null) {
			await ensureGemmaTokenizer(model);
			return tryCountText(model, text) ?? tryCountMessages(model, bodyText);
		}
		return tryCountMessages(model, bodyText);
	}

	await ensureGemmaTokenizer(model);
	return try_encode_text(model, readEncodeTextBody(bodyText));
}

/** Token count carried by a local response, for either operation. */
function payloadTokenCount(op, payload) {
	return op === "count" ? payload.token_count : payload.count;
}

function readFetchUrl(input) {
	if (typeof input === "string") return input;
	if (input instanceof URL) return input.href;
	return input?.url;
}

function readFetchMethod(input, options) {
	return options?.method ?? (input instanceof Request ? input.method : "GET");
}

async function readFetchBody(input, options) {
	return (
		options?.body ??
		(input instanceof Request ? await input.clone().text() : "")
	);
}

function installEndpointFallbacks() {
	installFetchFallback();

	if (window.jQuery) {
		installJQueryFallback(window.jQuery);
	}
}

function installFetchFallback() {
	const installed = patchOnce(window, "tokenizer.fetch", () => {
		const originalFetch = window.fetch.bind(window);

		window.fetch = async function stDotToolboxFetch(input, options) {
			try {
				const rawUrl = readFetchUrl(input);
				const target =
					rawUrl && parseTarget(rawUrl, readFetchMethod(input, options));
				if (target) {
					const body = await readFetchBody(input, options);
					// ST sends JSON strings. Unsupported models and non-string bodies fall
					// through to the original backend request.
					if (typeof body === "string") {
						const payload = await tryBuildTokenizerResponse(target, body);
						if (payload) {
							logSuccess(
								target.op,
								target.model,
								payloadTokenCount(target.op, payload),
							);
							return Response.json(payload);
						}
					}
					logFallback(target.op, target.model);
				}
			} catch (error) {
				logger.error("fetch intercept failed, falling back to backend:", error);
			}

			return originalFetch(input, options);
		};

		return { originalFetch };
	});

	if (installed) logger.debug("fetch endpoint fallback installed.");
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

function installJQueryFallback($) {
	const installed = patchOnce($, "tokenizer.jquery-ajax", () => {
		const originalAjax = $.ajax;

		$.ajax = function stDotToolboxAjax(options, settings) {
			try {
				const request = getAjaxRequest(options, settings);
				const target =
					request?.url &&
					parseTarget(request.url, request.type ?? request.method);
				if (target) {
					const payload = tryBuildTokenizerResponseSync(
						target,
						readAjaxBody(request),
					);
					if (payload) {
						logSuccess(
							target.op,
							target.model,
							payloadTokenCount(target.op, payload),
						);
						return resolveAjaxLocally($, request, payload);
					}
					logFallback(target.op, target.model);
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

	if (installed) logger.debug("jQuery endpoint fallback installed.");
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
							const bodyText = JSON.stringify(messagesArray);

							const token_count = try_count_messages(model, bodyText);
							if (token_count === undefined) {
								logFallback("count", model);
								return originalCountAsync.call(this, messages, full, type);
							}
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
