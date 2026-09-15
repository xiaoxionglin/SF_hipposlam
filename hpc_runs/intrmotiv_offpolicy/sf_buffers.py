"""Opt-in extension of SF's standard policy-output buffer schema.

SF already copies all declared outputs through inference, rollout and Batcher.
Only the shape declaration needs extending; ordinary SF configurations retain
exactly their existing fields. Install before constructing a Runner.
"""


def install():
    from sample_factory.algo.utils import shared_buffers

    if getattr(shared_buffers.policy_output_shapes, "_intrmotiv_extended", False):
        return
    original = shared_buffers.policy_output_shapes

    def shapes(cfg, *args, **kwargs):
        result = original(cfg, *args, **kwargs)
        width = getattr(cfg, "ddqn_packet_width", 0)
        if width:
            if width < 1 or any(name == "ddqn_packet" for name, _ in result):
                raise ValueError("invalid DDQN policy packet extension")
            result = result + [("ddqn_packet", [width])]
        return result

    shapes._intrmotiv_extended = True
    shared_buffers.policy_output_shapes = shapes
