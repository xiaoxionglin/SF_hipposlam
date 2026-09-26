"""Main update accounting is independent of auxiliary sampling and acceptance."""

from dataclasses import asdict, dataclass

import numpy as np

from hpc_runs.intrmotiv_offpolicy.sf_transport import updates_due


@dataclass
class UpdateClock:
    learning_starts: int = 16384
    cadence: int = 64
    target_interval: int = 100
    completed: int = 0
    main_positions: int = 0
    auxiliary_positions: int = 0
    target_at: int = 0

    def due(self, accepted):
        # Warm-up itself does not accumulate replay update debt.
        return updates_due(int(accepted), self.learning_starts, self.cadence, self.completed)

    def finish(self, main_positions, auxiliary_positions):
        if main_positions <= 0 or auxiliary_positions < 0:
            raise ValueError("A completed update requires main TD positions")
        self.completed += 1
        self.main_positions += main_positions
        self.auxiliary_positions += auxiliary_positions
        refresh = self.completed - self.target_at >= self.target_interval
        if refresh:
            self.target_at = self.completed
        return refresh

    def state_dict(self):
        return asdict(self)

    def load_state_dict(self, state):
        for key in asdict(self):
            setattr(self, key, int(state[key]))


def replay_rngs(seed):
    # Auxiliary sample counts never advance the main replay generator.
    main, auxiliary = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(main), np.random.default_rng(auxiliary)
