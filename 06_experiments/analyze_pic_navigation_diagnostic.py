"""Thin analysis adapter for the reviewed PIC diagnostic manifests.

Run from the NEMO2 runtime with its existing field/intervention backends.
Writes lightweight tables; raw observations and trajectories stay in workspace.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sf_working_directories.IntrMotiv.evaluation.analyze_place_field_manifest import (
    artifact_for_suffix, pairwise_map_cosine, peak_statistics)
from sf_working_directories.IntrMotiv.evaluation.qualify_absent_goal_interventions import canonical_components
from sf_working_directories.IntrMotiv.evaluation.physical_destination_control import analyze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--fields-only',action='store_true')
    parser.add_argument('--fields-subdir',default='fields_r2')
    args=parser.parse_args();root=args.root;out=root/'summary';out.mkdir(exist_ok=True)
    rows=list(csv.DictReader((root/'selected_manifest.tsv').open(),delimiter='\t'))
    summaries=[];units=[];components=[];plot_rows=[];controls=[];missing=[]
    for row in rows:
        try:
            path=artifact_for_suffix(root/args.fields_subdir/'raw',row['label_suffix'])
        except (FileNotFoundError,ValueError) as error:
            missing.append(dict(label=row['label_suffix'],artifact='fields',reason=str(error)));continue
        with np.load(path,allow_pickle=False) as archive:
            data={k:archive[k] for k in archive.files}
        identity={k:row[k] for k in ('condition','seed','target_frames','checkpoint_frames')}
        occ=data['occupancy'];visited=occ>0
        for layer, mapkey, activitykey, infokey in (
            ('raw','raw_dg_rate_maps','raw_dg_active_fraction','raw_dg_spatial_information'),
            ('inhibited','rate_maps','active_fraction','spatial_information')):
            maps=data[mapkey];active=np.flatnonzero(data[activitykey]>0)
            unique,entropy,distance=peak_statistics(maps,active,require_positive=True)
            prefix='raw_dg' if layer=='raw' else 'post_inhibition'
            mono=data[prefix+'_field_mono'];eligible=data[prefix+'_field_eligible']
            fractions=data[prefix+'_field_threshold_fractions']
            counts=data[prefix+'_field_component_count']
            middle=int(np.argmin(np.abs(fractions-.5)))
            masks=canonical_components(maps,occ)
            area=masks.sum((0,1))/max(1,visited.sum())
            mean_activity=np.nansum(maps*occ[...,None],axis=(0,1))/max(1,occ.sum())
            information_per_activity=np.divide(data[infokey],mean_activity,
                out=np.full_like(mean_activity,np.nan),where=mean_activity>0)
            summaries.append(dict(**identity,layer=layer,observations=int(occ.sum()),
                visited_bins=int(visited.sum()),total_grid_bins=occ.size,
                active_units=len(active),silent_units=maps.shape[-1]-len(active),
                eligible_units=int(eligible.sum()),mono_units=int(mono.sum()),
                median_half_peak_components=float(np.median(counts[middle,eligible])) if eligible.any() else None,
                mean_spatial_information=float(np.mean(data[infokey][active])) if len(active) else None,
                mean_activity=float(np.mean(mean_activity)),
                mean_information_per_activity=float(np.nanmean(information_per_activity)) if len(active) else None,
                active_only_map_cosine=pairwise_map_cosine(maps,occ,active),
                unique_peak_bins=unique,peak_entropy=entropy,pairwise_peak_distance_bins=distance,
                median_canonical_area_fraction=float(np.median(area[active])) if len(active) else None,
                mean_activity_fraction=float(np.mean(data[activitykey])),
                observation_panel=str(data['observation_panel']),artifact=str(path)))
            for unit in range(maps.shape[-1]):
                units.append(dict(**identity,layer=layer,unit=unit,active_fraction=float(data[activitykey][unit]),
                    spatial_information=float(data[infokey][unit]),eligible=bool(eligible[unit]),
                    mean_activity=float(mean_activity[unit]),information_per_activity=float(information_per_activity[unit]),
                    mono=bool(mono[unit]),mono_score=float(data[prefix+'_field_mono_score'][unit]),
                    canonical_area_fraction=float(area[unit])))
                for level,fraction in enumerate(fractions):
                    components.append(dict(**identity,layer=layer,unit=unit,eligible=bool(eligible[unit]),
                        peak_fraction=float(fraction),component_count=int(counts[level,unit]),
                        dominant_mass_fraction=float(data[prefix+'_field_dominant_mass_fraction'][level,unit])))
            for unit in (0,4,8,12):
                peak=float(np.nanmax(maps[...,unit]))
                for x in range(maps.shape[0]):
                    for y in range(maps.shape[1]):
                        plot_rows.append(dict(**identity,layer=layer,unit=unit,x_bin=x,y_bin=y,
                            occupancy=int(occ[x,y]),rate=float(maps[x,y,unit]),unit_peak=peak))
        maps=data['pre_threshold_rate_maps'];allunits=np.arange(maps.shape[-1])
        for unit in (0,4,8,12):
            for x in range(maps.shape[0]):
                for y in range(maps.shape[1]):
                    plot_rows.append(dict(**identity,layer='pre_threshold',unit=unit,x_bin=x,y_bin=y,
                        occupancy=int(occ[x,y]),rate=float(maps[x,y,unit]),unit_peak=float(np.nanmax(maps[...,unit]))))
        unique,entropy,distance=peak_statistics(maps,allunits,require_positive=False)
        summaries.append(dict(**identity,layer='pre_threshold',observations=int(occ.sum()),
            visited_bins=int(visited.sum()),active_only_map_cosine=pairwise_map_cosine(maps,occ),
            unique_peak_bins=unique,peak_entropy=entropy,pairwise_peak_distance_bins=distance,
            observation_panel=str(data['observation_panel']),artifact=str(path)))
        if row['schedule']=='goal' and not args.fields_only:
            trials=root/'interventions/raw'/row['label_suffix']/'intervention_trials.csv'
            if not trials.exists():
                missing.append(dict(label=row['label_suffix'],artifact='interventions',reason='Not completed'));continue
            result=analyze(trials,path,out/row['label_suffix'])
            if row['condition']=='PIC_G_CONTEXT':
                # This condition has no reentry inhibition: rate_maps retain
                # effective history-gated responses, while raw maps are visual-only.
                analyze(trials,path,out/row['label_suffix']/'context_effective_destination_sensitivity',
                        map_key='rate_maps')
            for definition in ('peak_bin','canonical_component'):
                for horizon,values in result[definition].items():
                    controls.append(dict(**identity,destination=definition,horizon=int(horizon),**values))
    pd.DataFrame(summaries).to_csv(out/'field_summary.csv',index=False)
    pd.DataFrame(units).to_csv(out/'field_units.csv',index=False)
    pd.DataFrame(components).to_csv(out/'field_components.csv',index=False)
    pd.DataFrame(plot_rows).to_csv(out/'selected_map_data.csv',index=False)
    pd.DataFrame(controls).to_csv(out/'physical_control_summary.csv',index=False)
    manifest=json.loads((root/'study_manifest.json').read_text())
    manifest.update(analysis='PIC navigation diagnostic',selected_manifest=str(root/'selected_manifest.tsv'),
                    missing=missing,fields_only=args.fields_only)
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(field_rows=len(summaries),control_rows=len(controls),missing=missing),indent=2))


if __name__=='__main__':
    main()
