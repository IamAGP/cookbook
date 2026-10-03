"""One-sample probe of the Fireworks base-only sampler through the peer's exact code path.

Checks, before any eval spend, the three things that would silently corrupt the comparison:
1. the base-only serverless route works for Qwen3.8-27B;
2. the sampled tokens end with the renderer's stop token (her parser needs it to see a
   complete message, and tool calls are parsed from that message);
3. sampling logprobs align one-to-one with sampled tokens.

Cost: one prompt (~2K tokens) and one completion (at most 8,192 tokens) -> under $0.05.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_eval_fireworks import MY_ENV, FireworksServiceShim, be, verify_harness_pinned  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from tinker_cookbook import model_info, tokenizer_utils  # noqa: E402
from tinker_cookbook.completers import TinkerTokenCompleter  # noqa: E402
from tinker_cookbook.renderers import get_renderer  # noqa: E402

BASE_MODEL = "Qwen/Qwen3.8-27B"
QUESTION = "How many users are in the graph?"


async def probe() -> dict:
    shim = FireworksServiceShim()
    try:
        sampler = await shim.create_sampling_client_async(base_model=BASE_MODEL)
        renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
        tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
        renderer = get_renderer(renderer_name, tokenizer)
        tool = be.CypherTool(driver=None, database="neo4j")  # spec only; never executed here
        messages = renderer.create_conversation_prefix_with_tools(
            tools=[tool.run_cypher.to_spec()], system_prompt=be.SYSTEM_PROMPT
        ) + [{"role": "user", "content": QUESTION}]
        model_input = renderer.build_generation_prompt(messages)
        stop = renderer.get_stop_sequences()
        policy = TinkerTokenCompleter(sampler, max_tokens=8192, temperature=1.0)
        out = await policy(model_input, stop)
        tokens = list(out.tokens)
        message, parse_ok = renderer.parse_response(tokens)
        return {
            "session": shim.service.training_session_id,
            "sampler_model": getattr(sampler.deployment_sampler, "model", "?"),
            "renderer": renderer_name,
            "prompt_tokens": model_input.length,
            "sampled_tokens": len(tokens),
            "stop_reason": out.stop_reason,
            "stop_ids": stop,
            "stop_strings": [tokenizer.decode([s], skip_special_tokens=False) for s in stop]
            if all(isinstance(s, int) for s in stop) else stop,
            "last_token": tokens[-1] if tokens else None,
            "last_token_is_stop": bool(tokens) and tokens[-1] in stop,
            "logprobs_aligned": len(out.logprobs) == len(tokens),
            "parse_ok": bool(parse_ok),
            "has_tool_call": bool(message.get("tool_calls")),
            "tool_calls": [str(tc)[:300] for tc in (message.get("tool_calls") or [])],
            "tail_text": tokenizer.decode(tokens[-40:], skip_special_tokens=False),
            "est_usd_upper": round((model_input.length * 1.86 + len(tokens) * 5.595) / 1e6, 5),
        }
    finally:
        shim.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv(MY_ENV)
    print(json.dumps(verify_harness_pinned(), indent=1))
    print(json.dumps(asyncio.run(probe()), indent=1, default=str))
