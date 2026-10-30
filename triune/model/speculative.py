"""Self-Speculative Decoding Engine using Triune's Hierarchical Early Exits.

Exploits Triune's structural multi-exit architecture (Reflex -> Limbic -> Cortex):
- Draft Phase: Reflex exit (layer reflex_exit_layer) generates K draft tokens at ~3-6x speed.
- Verification Phase: Cortex full depth (layer num_layers) evaluates all K draft tokens in a
  single parallel forward pass.
- Zero extra weights in VRAM: draft and target models are the exact same weights and embeddings.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfSpeculativeGenerator:
    """Self-speculative decoding pipeline utilizing Triune's native hierarchical exits."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        k_draft: int = 3,
        temperature: float = 0.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.k_draft = max(1, k_draft)
        self.temperature = max(0.0, temperature)

        # Resolve special tokens
        self.pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else getattr(tokenizer, "token_to_id", lambda t: None)("[PAD]")
        )
        self.eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else getattr(tokenizer, "token_to_id", lambda t: None)("[SEP]")
        )

        self.telemetry = {
            "draft_tokens_proposed": 0,
            "draft_tokens_accepted": 0,
            "verification_passes": 0,
            "total_tokens_emitted": 0,
            "wall_time_seconds": 0.0,
        }

    @property
    def acceptance_rate(self) -> float:
        proposed = self.telemetry["draft_tokens_proposed"]
        if proposed == 0:
            return 0.0
        return self.telemetry["draft_tokens_accepted"] / proposed

    def reset_telemetry(self) -> None:
        for k in self.telemetry:
            self.telemetry[k] = 0 if isinstance(self.telemetry[k], int) else 0.0

    @torch.inference_mode()
    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int = 64,
        on_token: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Generates tokens using Reflex-to-Cortex self-speculative decoding."""
        t_start = time.perf_counter()
        device = next(self.model.parameters()).device

        # Tokenize prompt if string
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt).ids
            input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        else:
            input_ids = prompt.to(device=device, dtype=torch.long)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

        initial_len = input_ids.size(1)
        curr_ids = input_ids.clone()
        generated_tokens: List[int] = []

        while len(generated_tokens) < max_new_tokens:
            curr_len = curr_ids.size(1)

            # 1. Draft Phase: Reflex exit (force_depth=0) proposes K tokens
            draft_ids: List[int] = []
            draft_ctx = curr_ids.clone()

            for _ in range(self.k_draft):
                res_draft = self.model(draft_ctx, force_depth=0)
                logits_draft = res_draft[0] if isinstance(res_draft, (tuple, list)) else res_draft
                next_draft_logits = logits_draft[0, -1]

                if self.temperature == 0.0:
                    next_tok = int(torch.argmax(next_draft_logits).item())
                else:
                    probs = F.softmax(next_draft_logits / self.temperature, dim=-1)
                    next_tok = int(torch.multinomial(probs, num_samples=1).item())

                draft_ids.append(next_tok)
                next_tok_tensor = torch.tensor([[next_tok]], device=device, dtype=torch.long)
                draft_ctx = torch.cat([draft_ctx, next_tok_tensor], dim=1)

                if self.eos_token_id is not None and next_tok == self.eos_token_id:
                    break

            K_actual = len(draft_ids)
            self.telemetry["draft_tokens_proposed"] += K_actual

            # 2. Verification Phase: Single parallel Cortex forward pass (force_depth=2)
            # Evaluates context + proposed draft tokens all at once
            self.telemetry["verification_passes"] += 1
            res_target = self.model(draft_ctx, force_depth=2)
            logits_target = res_target[0] if isinstance(res_target, (tuple, list)) else res_target

            # Verify each proposed draft token
            accepted_in_this_cycle = 0
            stopped = False

            for k_idx in range(K_actual):
                target_pos_logits = logits_target[0, curr_len - 1 + k_idx]
                if self.temperature == 0.0:
                    target_tok = int(torch.argmax(target_pos_logits).item())
                else:
                    probs = F.softmax(target_pos_logits / self.temperature, dim=-1)
                    target_tok = int(torch.multinomial(probs, num_samples=1).item())

                draft_tok = draft_ids[k_idx]

                if draft_tok == target_tok:
                    # Accepted
                    accepted_in_this_cycle += 1
                    self.telemetry["draft_tokens_accepted"] += 1
                    generated_tokens.append(draft_tok)
                    curr_ids = torch.cat([curr_ids, torch.tensor([[draft_tok]], device=device)], dim=1)
                    if on_token is not None:
                        tok_str = getattr(self.tokenizer, "decode", lambda ids: "")([draft_tok])
                        on_token(draft_tok, tok_str)

                    if self.eos_token_id is not None and draft_tok == self.eos_token_id:
                        stopped = True
                        break
                    if len(generated_tokens) >= max_new_tokens:
                        stopped = True
                        break
                else:
                    # Mismatch: reject current draft token, emit target token, discard remaining draft
                    generated_tokens.append(target_tok)
                    curr_ids = torch.cat([curr_ids, torch.tensor([[target_tok]], device=device)], dim=1)
                    if on_token is not None:
                        tok_str = getattr(self.tokenizer, "decode", lambda ids: "")([target_tok])
                        on_token(target_tok, tok_str)
                    if self.eos_token_id is not None and target_tok == self.eos_token_id:
                        stopped = True
                    break

            if stopped:
                break

            # If all K tokens were accepted and we haven't stopped, bonus token from target position K
            if accepted_in_this_cycle == K_actual and len(generated_tokens) < max_new_tokens:
                bonus_logits = logits_target[0, curr_len - 1 + K_actual]
                if self.temperature == 0.0:
                    bonus_tok = int(torch.argmax(bonus_logits).item())
                else:
                    probs = F.softmax(bonus_logits / self.temperature, dim=-1)
                    bonus_tok = int(torch.multinomial(probs, num_samples=1).item())

                generated_tokens.append(bonus_tok)
                curr_ids = torch.cat([curr_ids, torch.tensor([[bonus_tok]], device=device)], dim=1)
                if on_token is not None:
                    tok_str = getattr(self.tokenizer, "decode", lambda ids: "")([bonus_tok])
                    on_token(bonus_tok, tok_str)
                if self.eos_token_id is not None and bonus_tok == self.eos_token_id:
                    break

        t_elapsed = max(1e-6, time.perf_counter() - t_start)
        self.telemetry["total_tokens_emitted"] = len(generated_tokens)
        self.telemetry["wall_time_seconds"] = t_elapsed

        decoded_text = ""
        if hasattr(self.tokenizer, "decode"):
            decoded_text = self.tokenizer.decode(generated_tokens)

        return {
            "text": decoded_text,
            "token_ids": generated_tokens,
            "tokens_count": len(generated_tokens),
            "tokens_per_second": round(len(generated_tokens) / t_elapsed, 2),
            "wall_time_seconds": round(t_elapsed, 4),
            "acceptance_rate": round(self.acceptance_rate, 4),
            "telemetry": dict(self.telemetry),
        }
