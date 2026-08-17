"""Local Qwen behind the FROZEN M2 contract, via MLX on Apple silicon.

Fourth engine behind `M2(state, reasoning_budget) -> M2Result`, and the Governor
has still never been changed to accommodate one. The point of this backend is
not accuracy -- a 1.7B model is weaker than the hosted ones -- it is that the
seam holds for an engine with completely different runtime characteristics: no
network, no per-day token quota, and generation speed bounded by local memory
bandwidth rather than by someone's rate limiter.

HARDWARE, MEASURED. This machine has 8 GB of unified memory and ~9 GB of free
disk, so 4B at 4-bit is not safe alongside everything else running. 1.7B-4bit is
about 1 GB of weights and fits. The size is chosen from the hardware, not from
what would look best.

`cost_units` is the completion-token count, the same quantity the hosted engines
charge, so the accounting is comparable across backends.
"""
from __future__ import annotations

import time
from typing import Any

from governor.gate.m2_interface import M2Result

DEFAULT_MODEL = "mlx-community/Qwen3-1.7B-4bit"


class QwenLocalM2:
    name = "qwen_local"

    def __init__(self, model: str = DEFAULT_MODEL, system_prompt: str | None = None,
                 enable_thinking: bool | None = None):
        self.model_id = model
        self.system_prompt = system_prompt
        # Qwen3 gates its reasoning trace on a chat-template flag. Measured:
        # with the standard terse system prompt the 1.7B emits an EMPTY
        # <think></think> and answers in 11 tokens, so a token budget cannot
        # matter to it. Recording both settings is the honest way to report
        # that, rather than picking whichever produces a curve.
        self.enable_thinking = enable_thinking
        self._model: Any = None
        self._tok: Any = None

    def load(self) -> None:
        if self._model is None:
            from mlx_lm import load
            self._model, self._tok = load(self.model_id)

    def __call__(self, state: dict, reasoning_budget: float) -> M2Result:
        t0 = time.time()
        try:
            self.load()
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
            msgs = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": self.system_prompt})
            msgs.append({"role": "user", "content": state["prompt"]})
            kw = ({} if self.enable_thinking is None
                  else {"enable_thinking": self.enable_thinking})
            try:
                prompt = self._tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False, **kw)
            except TypeError:
                prompt = self._tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False)
            n_prompt = len(self._tok.encode(prompt))
            text = generate(self._model, self._tok, prompt=prompt,
                            max_tokens=int(reasoning_budget), verbose=False,
                            sampler=make_sampler(temp=0.0))
            n_out = len(self._tok.encode(text))
            # `generate` stops at max_tokens without saying so; a completion that
            # used its whole allowance is treated as starved, exactly as a
            # provider finish_reason="length" is.
            starved = n_out >= int(reasoning_budget) - 1
            return M2Result(result=text,
                            reasoning_tokens=float(n_out),
                            total_tokens=float(n_prompt + n_out),
                            latency_s=time.time() - t0,
                            cost_units=float(n_out),
                            ok=not starved and bool(text.strip()),
                            error="" if not starved else "hit max_tokens (starved)")
        except Exception as e:                               # noqa: BLE001
            return M2Result(result=None, latency_s=time.time() - t0,
                            cost_units=0.0, ok=False, error=str(e)[:200])
