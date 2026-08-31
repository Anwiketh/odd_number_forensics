"""Generic binary answer-slot metric shared by every environment."""
import torch
from typing import List


class SlotMetric:
    """LD(side) = logsumexp(logits over answers[side]) - logsumexp(other side).

    Requires every answer string to be a single token after the prefill
    'ANSWER: '. verify() checks this and reports how much next-token mass the
    answer set actually captures (we want ~1.0, else the metric is measuring a
    thin slice of an open distribution)."""

    def __init__(self, runner, task):
        self.R, self.T = runner, task
        # If the prefill already ends in a space the model emits the bare
        # token ('4'); if it does not, the model emits the space-prefixed one
        # (' A'). Getting this backwards silently puts ~0 probability mass on
        # the answer set, which makes CLD a ratio of two negligible numbers.
        space_prefixed = not task.prefill.endswith(" ")
        self.ids = {}
        for side, strs in task.answers.items():
            out = []
            for s in strs:
                order = ([" " + s, s] if space_prefixed else [s, " " + s])
                tok = None
                for cand in order:
                    t = runner.tok.encode(cand, add_special_tokens=False)
                    if len(t) == 1:
                        tok = t[0]
                        break
                assert tok is not None, f"{s!r} has no single-token form"
                out.append(tok)
            self.ids[side] = out

    def ld(self, logits_last: torch.Tensor, side: str) -> float:
        a = torch.stack([logits_last[i] for i in self.ids[side]])
        b = torch.stack([logits_last[i] for i in self.ids[self.T.other(side)]])
        return float(torch.logsumexp(a, 0) - torch.logsumexp(b, 0))

    def mass(self, logits_last: torch.Tensor) -> float:
        p = torch.softmax(logits_last, -1)
        return float(sum(p[i] for s in self.ids for i in self.ids[s]))

    def argmax_side(self, logits_last: torch.Tensor) -> str:
        s0 = self.T.sides[0]
        return s0 if self.ld(logits_last, s0) > 0 else self.T.other(s0)
