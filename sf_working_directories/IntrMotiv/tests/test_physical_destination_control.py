import numpy as np
from sf_working_directories.IntrMotiv.evaluation.physical_destination_control import destination_arrival, evaluate_destinations
from sf_working_directories.IntrMotiv.evaluation.absent_goal_interventions import summarize_commands


def trial(command, path):
    return dict(start=0,command=command,eligible_count=2,eligible_commands=[0,1],
                start_position=[1050,1050,0],trajectory=path)


def test_actual_arrival_without_dg_events_and_straightness():
    mask=np.zeros((19,19),bool);mask[0,0]=True
    result=destination_arrival(trial(0,[[500,500],[150,150]]),mask,2)
    assert result['hit'] and result['latency']==2
    assert abs(result['straight_line_efficiency']-1)<1e-7
    assert result['arrival_x']==150


def test_matched_control_requires_different_physical_destinations():
    masks=np.zeros((19,19,2),bool);masks[0,0,0]=True;masks[18,18,1]=True
    rows=[trial(0,[[150,150]]),trial(1,[[1950,1950]])]
    _,_,summary=evaluate_destinations(rows,masks,[1])
    assert summary['1']['hit_difference']==1
    rows[1]['trajectory']=[[150,150]]
    _,_,summary=evaluate_destinations(rows,masks,[1])
    assert summary['1']['hit_difference']==0


def test_incomplete_commands_and_start_inside_do_not_inflate_success():
    masks=np.zeros((19,19,2),bool);masks[0,0,0]=True;masks[9,9,1]=True
    rows=[trial(0,[[150,150]]),trial(1,[[1050,1050]])]
    _,_,summary=evaluate_destinations(rows,masks,[1])
    assert summary['1']['represented_target_ids']==[0]
    assert summary['1']['all_identity_matched_hit'] is None
    assert summary['1']['all_identity_matched_hit_bounds']==[.5,1.]
    _,_,summary=evaluate_destinations(rows[:1],masks,[1])
    assert summary['1']['hit_difference'] is None
    rows[1]['trajectory']=[]
    _,_,summary=evaluate_destinations(rows,masks,[1])
    assert summary['1']['hit_difference'] is None


def test_extended_horizons_keep_late_arrivals():
    rows=[dict(start=0,command=0,complete=True,eligible_count=2,first_onset_times=[300,0]),
          dict(start=0,command=1,complete=True,eligible_count=2,first_onset_times=[0,300])]
    summary=summarize_commands(rows,2,[64,256,512,768])
    assert summary['ctc_256']==0 and summary['ctc_512']==1


def test_optional_field_arrays_preserve_evaluator_xy_orientation():
    import pandas as pd
    from sf_working_directories.IntrMotiv.evaluation.place_fields import spatial_details_for_artifact, compute_place_fields
    pose=pd.DataFrame(dict(x=[150]*24,y=[350]*24,rot_y=[0]*24))
    values=np.ones((24,2))
    occupancy,maps,_,_=compute_place_fields(pose,values,19)
    details=spatial_details_for_artifact(pose,values,19)
    np.testing.assert_array_equal(details['occupancy'],occupancy)
    np.testing.assert_allclose(details['rate_maps'][occupancy>0],maps[occupancy>0])
    assert details['occupancy'][0,2]==24 and details['occupancy'][2,0]==0


def test_existing_information_scales_with_amplitude_not_relative_selectivity():
    from sf_working_directories.IntrMotiv.evaluation.place_fields import spatial_information
    rate=np.array([[1.,0.],[0.,0.]])
    occupancy=np.ones((2,2))
    score=spatial_information(rate,occupancy)
    scaled=spatial_information(rate/400,occupancy)
    assert np.isclose(scaled,score/400)
    assert np.isclose(scaled/(rate/400).mean(),score/rate.mean())


def test_command_trajectory_effect_uses_only_within_start_pairs():
    from sf_working_directories.IntrMotiv.evaluation.physical_destination_control import trajectory_command_effect
    rows=[trial(0,[[150,150]]),trial(1,[[150,150]])]
    rows.append({**trial(0,[[1900,1900]]),'start':1})
    result=trajectory_command_effect(rows)
    assert result['paired_command_pairs']==1 and result['identical_trajectory_fraction']==1


def test_explicit_map_basis_preserves_raw_default_and_labels_sensitivity(tmp_path):
    import json
    import pandas as pd
    from sf_working_directories.IntrMotiv.evaluation.physical_destination_control import analyze
    rows=[trial(0,[[150,150]]),trial(1,[[1950,1950]])]
    for row in rows:
        for key in ('trajectory','start_position','eligible_commands'):
            row[key]=json.dumps(row[key])
    trials=tmp_path/'intervention_trials.csv';pd.DataFrame(rows).to_csv(trials,index=False)
    trials.with_name('intervention_summary.json').write_text(json.dumps(dict(
        exact_start_verified=True,policy_frozen=True,checkpoint='/tmp/checkpoint.pth',horizons=[1])))
    maps=np.zeros((19,19,2));maps[0,0,0]=1;maps[18,18,1]=1
    fields=tmp_path/'fields.npz'
    np.savez(fields,raw_dg_rate_maps=maps,rate_maps=maps[...,::-1],occupancy=np.ones((19,19)),
             checkpoint='/tmp/checkpoint.pth',observation_panel='/tmp/independent.npz',pose_alignment='observation_time')
    raw=analyze(trials,fields,tmp_path/'raw')
    effective=analyze(trials,fields,tmp_path/'effective',map_key='rate_maps')
    assert raw['peak_bin']['1']['hit_difference']==1
    assert effective['peak_bin']['1']['hit_difference']==-1
    assert raw['destination_map_key']=='raw_dg_rate_maps'
    assert effective['destination_map_key']=='rate_maps'
