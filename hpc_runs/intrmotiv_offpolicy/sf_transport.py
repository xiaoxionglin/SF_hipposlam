"""Ordered physical replay ingestion from asynchronously delivered SF packets."""

from collections import defaultdict
from dataclasses import replace


class OrderedIngress:
    """SF's batcher may reorder buffer slices. Never treat a slice as a stream.

    A nonterminal action waits for the next actor feature packet, including at
    rollout boundaries. This avoids re-encoding and reset-image bootstrapping.
    Unconsumed transport tails remain outside the training-frame budget.
    """

    def __init__(self, max_pending):
        self.expected = defaultdict(int)
        self.waiting = defaultdict(dict)
        self.max_pending = max_pending
        self.received = 0
        self.emitted = 0

    @property
    def pending(self):
        return sum(len(q) for q in self.waiting.values())

    def add(self, stream, serial, row):
        queue = self.waiting[stream]
        if row.stream != stream or serial < self.expected[stream] or serial in queue:
            raise ValueError("duplicate or inconsistent SF stream packet")
        queue[serial] = row
        self.received += 1
        if self.pending > self.max_pending:
            raise ValueError("SF reorder queue exceeded transport bound")

    def drain(self):
        # Interleave streams instead of draining one physical episode at a time.
        while True:
            advanced = False
            for stream in sorted(self.waiting):
                queue = self.waiting[stream]
                serial = self.expected[stream]
                row = queue.get(serial)
                if row is None:
                    continue
                ended = row.terminated or row.truncated
                following = queue.get(serial + 1)
                if not ended and following is None:
                    continue
                if not ended and (following.episode != row.episode or following.index != row.index + 1):
                    raise ValueError("SF successor crosses a physical reset or gap")
                result = replace(
                    row,
                    successor=row.successor if ended else following.observation,
                    successor_valid=row.successor_valid if ended else True,
                )
                del queue[serial]
                self.expected[stream] += 1
                self.emitted += 1
                advanced = True
                yield result
            if not advanced:
                break


def updates_due(accepted, warmup, cadence, completed):
    """Debt is based on completed optimizer updates, never on a capped request."""
    return max(0, (accepted - warmup) // cadence - completed)
