"""Machine-specific locations, read from the environment so nothing local is hardcoded.

TINKER_COOKBOOK_DIR   checkout of the Tinker-side repository whose harness, environment and
                      reward are imported (and gated on git blob hashes).
BIRD_REFERENCE_URI    s3://<bucket>/<prefix>/gold/reference_answers.json
BIRD_FW_ENV_FILE      env file holding FIREWORKS_API_KEY (default: .env in the working directory)
"""
from __future__ import annotations

import os
from pathlib import Path


def tinker_cookbook_dir() -> Path:
    value = os.environ.get("TINKER_COOKBOOK_DIR")
    if not value:
        raise SystemExit("set TINKER_COOKBOOK_DIR to a checkout of the Tinker-side repository at the pinned commit")
    return Path(value).expanduser()


def reference_uri() -> str:
    value = os.environ.get("BIRD_REFERENCE_URI")
    if not value:
        raise SystemExit("set BIRD_REFERENCE_URI to the s3:// location of reference_answers.json")
    return value


def fireworks_env_file() -> Path:
    return Path(os.environ.get("BIRD_FW_ENV_FILE", ".env")).expanduser()
