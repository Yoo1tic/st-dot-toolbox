/**
 * Installs a monkey patch at most once per browser page lifetime.
 *
 * The marker is stored on the patched owner object with a global symbol, so a
 * reloaded extension script does not wrap the same SillyTavern API repeatedly.
 *
 * @param {object} target Object that owns the patched behavior.
 * @param {string} key Stable patch identifier.
 * @param {() => unknown} install Patch installer.
 * @returns {boolean} Whether the patch was installed by this call.
 */
export function patchOnce(target, key, install) {
	if (!target) return false;

	const marker = Symbol.for(`st-dot-toolbox.${key}.patched`);
	if (target[marker]) return false;

	const metadata = install();
	Object.defineProperty(target, marker, {
		value: metadata ?? true,
		configurable: false,
		enumerable: false,
		writable: false,
	});

	return true;
}
