"""Keep physical final observations distinct from same-step reset outputs."""

import copy


def reset_with_final(env, observation, info):
    """Validity must be certified by the environment, never inferred from shape.

    DMLab historically returns the previous image after shutdown; that image
    can have the correct shape without representing the final physical state.
    """
    valid = bool(info.get("intrmotiv_final_observation_valid", False))
    final_observation = copy.deepcopy(observation) if valid else None
    final_info = copy.deepcopy(info)
    reset_observation, reset_info = env.reset()
    reset_info = dict(reset_info)
    reset_info.update(final_info=final_info, final_observation=final_observation, final_observation_valid=valid)
    return reset_observation, reset_info


def certified_successor(next_observation, info, ended):
    """Never accept a same-step autoreset image as a physical successor."""
    if not ended:
        return next_observation
    if info.get("final_observation_valid", False):
        return info.get("final_observation")
    return None


def vector_final_info(infos, index, num_envs):
    """Read outer autoreset fields even when the legacy selector returns final_info."""
    import numpy as np

    def select(value):
        if isinstance(value, dict):
            return {
                k: select(v)
                for k, v in value.items()
                if not k.startswith("_") and bool(np.asarray(value.get("_" + k, np.ones(num_envs, dtype=bool)))[index])
            }
        array = np.asarray(value)
        return array[index] if array.ndim and len(array) == num_envs else value

    return {
        k: select(infos[k])
        for k in ("final_observation", "final_observation_valid")
        if k in infos and bool(np.asarray(infos.get("_" + k, np.ones(num_envs, dtype=bool)))[index])
    }
