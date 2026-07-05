const ROOT_SCOPE = "st-dot-toolbox";
const consoleSink = {
	debug: console.debug.bind(console),
	info: console.log.bind(console),
	warn: console.warn.bind(console),
	error: console.error.bind(console),
};

/**
 * Creates a small scoped console logger for extension modules.
 *
 * @param {string} [scope] Feature scope appended to the root prefix.
 * @returns {{debug: Function, info: Function, warn: Function, error: Function}}
 */
export function createLogger(scope = "") {
	const prefix = scope ? `[${ROOT_SCOPE}/${scope}]` : `[${ROOT_SCOPE}]`;

	return {
		debug: (...args) => consoleSink.debug(prefix, ...args),
		info: (...args) => consoleSink.info(prefix, ...args),
		warn: (...args) => consoleSink.warn(prefix, ...args),
		error: (...args) => consoleSink.error(prefix, ...args),
	};
}
