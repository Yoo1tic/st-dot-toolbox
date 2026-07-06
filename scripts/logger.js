//! Scoped console logging with a runtime-adjustable severity threshold.
//!
//! The threshold is shared across all scoped loggers and persisted in
//! localStorage, so field debugging can raise verbosity to `debug` without a
//! redeploy and lower it back to `info` (the default) for normal use.

const ROOT_SCOPE = "st-dot-toolbox";
const LOG_LEVEL_KEY = `${ROOT_SCOPE}.logLevel`;
const DEFAULT_LEVEL = "info";

/** Severity ordering; a message is emitted when its level >= the threshold. */
const LEVELS = Object.freeze({
	debug: 10,
	info: 20,
	warn: 30,
	error: 40,
	silent: 100,
});

const consoleSink = {
	debug: console.debug.bind(console),
	info: console.log.bind(console),
	warn: console.warn.bind(console),
	error: console.error.bind(console),
};

function readStoredLevel() {
	try {
		const raw = window.localStorage?.getItem(LOG_LEVEL_KEY);
		return raw && raw in LEVELS ? raw : DEFAULT_LEVEL;
	} catch {
		// localStorage can throw in sandboxed frames; fall back to the default.
		return DEFAULT_LEVEL;
	}
}

let currentLevel = readStoredLevel();

function enabled(level) {
	return LEVELS[level] >= LEVELS[currentLevel];
}

/**
 * Raises or lowers the shared severity threshold and persists the choice.
 *
 * @param {"debug"|"info"|"warn"|"error"|"silent"} level Minimum level to emit.
 */
export function setLogLevel(level) {
	if (!(level in LEVELS)) {
		consoleSink.warn(
			`[${ROOT_SCOPE}] unknown log level "${level}"; keeping "${currentLevel}".`,
		);
		return;
	}

	currentLevel = level;
	try {
		window.localStorage?.setItem(LOG_LEVEL_KEY, level);
	} catch {
		// Persistence is best-effort; the in-memory threshold still applies.
	}
}

/**
 * Creates a small scoped console logger for extension modules.
 *
 * @param {string} [scope] Feature scope appended to the root prefix.
 * @returns {{debug: Function, info: Function, warn: Function, error: Function}}
 */
export function createLogger(scope = "") {
	const prefix = scope ? `[${ROOT_SCOPE}/${scope}]` : `[${ROOT_SCOPE}]`;
	const emit =
		(level, sink) =>
		(...args) => {
			if (enabled(level)) sink(prefix, ...args);
		};

	return {
		debug: emit("debug", consoleSink.debug),
		info: emit("info", consoleSink.info),
		warn: emit("warn", consoleSink.warn),
		error: emit("error", consoleSink.error),
	};
}
