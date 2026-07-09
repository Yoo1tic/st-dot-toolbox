// SillyTavern third-party extension entry point.
//
// Keep this file as the composition root: feature modules own their setup and
// this bootstrapper decides which features are installed.

import { createLogger } from "./scripts/logger.js";
import { installTokenizer } from "./scripts/tokenizer.js";

const logger = createLogger();

const features = [["tokenizer", installTokenizer]];

async function installFeature(name, install) {
	try {
		await install();
		logger.info(`${name} installed.`);
	} catch (error) {
		logger.error(`${name} failed to install:`, error);
		window.toastr?.error(`${name} initialization failed`, "st-dot-toolbox");
	}
}

(async function boot() {
	for (const [name, install] of features) {
		await installFeature(name, install);
	}

	logger.info("loaded.");
})();
