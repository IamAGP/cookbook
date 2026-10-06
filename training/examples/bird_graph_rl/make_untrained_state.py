"""Save the training state of a freshly initialised, never-trained adapter (serverless).

For the serving-route control: the trained adapter is evaluated by loading its saved state into
a fresh session and sampling a snapshot of it, while the base model was sampled through the
session's base-only route. Loading an *untrained* adapter the same way and scoring it shows
whether the route by itself moves accuracy. No training or sampling call is made here, so no
tokens are billed.

    python bird_graph_rl/make_untrained_state.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import fireworks_env_file  # noqa: E402

SERVERLESS_URL = "https://api.fireworks.ai/training/v1/serverless"
MODEL = "accounts/fireworks/models/qwen3p8-27b"
NAME = "untrained0"


def main() -> None:
    from fireworks.training.sdk import FiretitanServiceClient

    load_dotenv(fireworks_env_file())
    service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url=SERVERLESS_URL)
    try:
        # Same adapter shape as FW-R1: rank 32, MLP and attention, no unembedding.
        training = service.create_lora_training_client(base_model=MODEL, rank=32, train_mlp=True,
                                                       train_attn=True, train_unembed=False)
        state_path = training.save_state(NAME).result().path
        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "state_path": state_path,
                  "session": service.training_session_id, "model": MODEL, "rank": 32,
                  "train_unembed": False, "training_calls": 0}
    finally:
        service.close()
    Path(sys.argv[1]).expanduser().write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
