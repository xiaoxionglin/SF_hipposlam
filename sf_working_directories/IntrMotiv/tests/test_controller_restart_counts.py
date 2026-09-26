from sf_working_directories.IntrMotiv.dmlab.controller_replay import PhysicalReplay


def test_multiple_restart_discard_counts_accumulate_without_inventing_interactions():
    replay = PhysicalReplay(100, 99)
    state = replay.state_dict()
    state.update(pending=3, received=13, accepted=10)
    replay.load_state_dict(state)
    assert replay.rejected["restart_pending_tail"] == 3
    state = replay.state_dict()
    state["pending"] = 2
    replay.load_state_dict(state)
    assert replay.rejected["restart_pending_tail"] == 5
    assert replay.received == 13 and replay.accepted == 10 and replay.session == 2
