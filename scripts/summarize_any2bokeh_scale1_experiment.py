"""Summarize measured ATB unit-scale results without changing baseline tables."""
import argparse
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def peak_summary(summary):
    assert summary['status']=='PHASE_MEMORY_COMPLETE' and len(summary['runs'])==2
    for row in summary['runs']:
        assert row['peak_reset_events'][0].get('excluded_preexisting_peak'), 'Use the corrected run-boundary peak guard'
        assert len(row['peak_reset_events'])==18 and len(row['stages'])==15
    names={row['stage'] for run in summary['runs'] for row in run['stages']}
    stages={name:max(row['peak_allocated_GiB'] for run in summary['runs'] for row in run['stages'] if row['stage']==name) for name in names}
    return dict(stage_allocated_peaks_GiB=stages,
        full_allocated_GiB=max(run['observed_peak_allocated_GiB'] for run in summary['runs']),
        full_reserved_GiB=max(run['observed_peak_reserved_GiB'] for run in summary['runs']),
        highest_named_stage=max(stages,key=stages.get))


def rng_positions(trace):
    return [dict(kind=row['kind'],clip=row['clip'],unit=row.get('unit'),field=key,state=row[key])
            for row in trace for key in ['RNG','RNG_before','RNG_after'] if key in row]


def paired_history(output):
    base=output/'repro_004/history';variants={}
    for variant in ['native','candidate']:
        folder=base/variant
        if not (folder/'memory-history-evidence.json').exists():
            return None
        summary=read(folder/'summary.json');evidence=read(folder/'memory-history-evidence.json')
        assert evidence['target_interval_not_overwritten'] and evidence['synchronized_requested_byte_boundaries_verified']==31
        with (folder/'memory-snapshot.pickle').open('rb') as file:
            trace=pickle.load(file)['device_traces'][0]
        ranges=[(row['stage_start_event'],row['stage_end_event']) for row in evidence['peak_evidence']]
        returns=[]
        for index,event in enumerate(trace):
            if event['action']!='alloc' or not any(start<=index<end for start,end in ranges):
                continue
            frame=event['frames'][0] if event.get('frames') else {}
            is_return=(frame.get('filename','').endswith('/diffusers/models/resnet.py') and frame.get('line')==258) if variant=='native' else (frame.get('filename')=='<ATB_encoder_scale1_candidate>' and frame.get('line')==1)
            if is_return:
                returns.append(event)
        phase_rows=[]
        for row in evidence['peak_evidence']:
            event=trace[row['requested_live_peak_event']]
            top=event['frames'][0]
            phase_rows.append(dict(stage=row['stage'],native_unit=row['native_unit'],
                concurrent_phase_requests_at_least1GiB=len(row['large_phase_allocations_at_requested_peak']),
                large_request_sizes_GiB=[x['requested_GiB'] for x in row['large_phase_allocations_at_requested_peak']],
                incremental_requested_live_peak_GiB=row['incremental_requested_live_peak_above_entry_GiB'],
                entry_RGB_cache_live=row['entry_RGB_cache_all_live_at_requested_peak'],
                requested_peak_trigger_size_MiB=event['size']/2**20,
                requested_peak_trigger=dict(file=top['filename'].split('/site-packages/')[-1],line=top['line'],function=top['name'])))
        variants[variant]=dict(events=len(trace),max_entries=summary['allocation_history']['max_entries'],
            total_alloc_events=sum(x['action']=='alloc' for x in trace),encoder_return_alloc_events=len(returns),
            other_original_ResNet_return_alloc_events=sum(x['action']=='alloc' and bool(x.get('frames')) and
                x['frames'][0]['filename'].endswith('/diffusers/models/resnet.py') and x['frames'][0]['line']==258 for x in trace)
                -(len(returns) if variant=='native' else 0),
            full_resolution4_5GiB_return_alloc_events=sum(x['size']==int(4.5*2**30) for x in returns),
            cumulative_encoder_return_requested_GiB=sum(x['size'] for x in returns)/2**30,
            actual_full_allocated_peak_GiB=summary['peak_allocated_GiB'],phase_evidence=phase_rows,
            coverage=evidence,allocator_settings=summary['allocation_history']['allocator_settings'],
            snapshots=dict(initial=str((folder/'memory-initial-snapshot.pickle').relative_to(ROOT)),final=str((folder/'memory-snapshot.pickle').relative_to(ROOT))))
    assert variants['native']['allocator_settings']==variants['candidate']['allocator_settings']
    assert variants['native']['other_original_ResNet_return_alloc_events']==variants['candidate']['other_original_ResNet_return_alloc_events']
    return dict(status='PAIRED_REPRESENTATIVE_HISTORY_COMPLETE',scene='repro_004',warmups=1,observations=1,variants=variants,
        interpretation='Encoder return-path alloc events120->60, full-resolution4.5GiB return requests24->12. This matches one omitted division materialization per invoked encoder ResNet; all60 invocations use scale1. Encoder high-request intervals have5->4 simultaneous4.5GiB requests. The candidate high point moves internally to conv1, with an additional~3.375MiB request whose exact internal role is unassigned; the named whole-run peak phase remains CoC.',
        bounds='Return counts are restricted to six RGB/CoC encoder stages; decoder/UNet return allocations are excluded. Cumulative requested bytes are allocation volume, not a peak or measured traffic. Trace sizes are requested bytes, not exact allocator block decomposition. Python source/stack comparison supports materialization attribution, without a low-level C++ operator/workspace trace. Non-PyTorch memory remains outside history; history times never enter Core table.')


def summarize(output):
    protocol=read(ROOT/'configs/efficiency-track-a.json')
    execution=read(output/'execution-status.json')
    table1,table2,table3,scenes=[],[],[],{}
    for sid,K in zip(protocol['scenes'],protocol['K']):
        folder=output/sid
        cp=folder/'correctness/candidate/summary.json'
        if not cp.exists():
            table1.append(dict(scene=sid,K=K,passed=None,status='NOT_MEASURED'))
            continue
        correctness=read(cp)
        traces={variant:read(folder/f'correctness/{variant}/trace.json') for variant in ['native','candidate']}
        samples={variant:[row for row in trace if row['kind']=='posterior_sample'] for variant,trace in traces.items()}
        rng_equal=rng_positions(traces['native'])==rng_positions(traces['candidate'])
        sample_equal=samples['native']==samples['candidate']
        measured=correctness['outputs'][1]
        c_row=dict(scene=sid,K=K,sample_calls_and_order_equal=sample_equal,RNG_positions_equal=rng_equal,
            RNG_positions_compared=len(rng_positions(traces['native'])),posterior_calls_per_clip=6,
            RGB_cache_equal=correctness['cache_contents_equal'],latent_cache_equal=correctness['latent_continuity_equal'],
            frames=measured['output_frames'],shape=measured['shape'],finite=measured['finite'],
            float_max_abs_diff=measured['max_abs_diff'],float_different_elements=measured['different_float_elements'],
            float_different_bit_patterns=measured['different_float_bit_patterns'],RGB8_different_bytes=measured['RGB8_different_bytes'],
            warmup_output=correctness['outputs'][0],passed=correctness['passed'])
        table1.append(c_row)
        data=dict(correctness=c_row,correctness_runtime={v:read(folder/f'correctness/{v}/summary.json')['runtime'] for v in ['native','candidate']},
            actual_encoder_warmup_hits={v:read(folder/f'correctness/{v}/summary.json')['warmup_patch_audit'] for v in ['native','candidate']},
            artifacts=dict(correctness={v:str((folder/f'correctness/{v}/summary.json').relative_to(ROOT)) for v in ['native','candidate']}))
        phase_paths={v:folder/f'phase/{v}/summary.json' for v in ['native','candidate']}
        time_paths={v:folder/f'timing/{v}/summary.json' for v in ['native','candidate']}
        if all(path.exists() for path in phase_paths.values()):
            phase={v:peak_summary(read(path)) for v,path in phase_paths.items()}
            data['phase_memory']=phase
            data['artifacts']['phase_memory']={v:str(path.relative_to(ROOT)) for v,path in phase_paths.items()}
            pairs=[('RGB encode allocated peak','A: phase diagnostic','stage_allocated_peaks_GiB','VAE_RGB_encode_sample_cache'),
                   ('CoC encode allocated peak','A: phase diagnostic','stage_allocated_peaks_GiB','VAE_CoC_encode_sample'),
                   ('Complete allocated peak','A: phase diagnostic','full_allocated_GiB',None),
                   ('Complete reserved peak','A: phase diagnostic','full_reserved_GiB',None)]
            for label,source,key,child in pairs:
                a,b=(phase[v][key][child] if child else phase[v][key] for v in ['native','candidate'])
                table2.append(dict(scene=sid,K=K,metric=label,source=source,native_GiB=a,candidate_GiB=b,
                    candidate_minus_native_GiB=b-a,relative_change_percent=100*(b-a)/a))
        if all(path.exists() for path in time_paths.values()):
            timing={v:read(path) for v,path in time_paths.items()}
            for v,t in timing.items():
                assert t['status']=='STANDARDIZED_TIMING_COMPLETE' and len(t['rows'])==5
                assert len(read(time_paths[v].parent/'any2bokeh/warmups.json'))==1
                for row in t['rows']:
                    assert row['output_frames']==25 and [u['output_frames'] for u in row['native_units']]==[8,8,9]
                    assert row['core_region_calls']=={'ATB_video_pipeline':3}
            assert timing['native']['parameters']['active_parameters']==timing['candidate']['parameters']['active_parameters']
            data['timing']={v:{key:value for key,value in t.items() if key not in ['parameters','Table3A']} for v,t in timing.items()}
            data['participating_parameters_unchanged']=timing['native']['parameters']['active_parameters']
            data['artifacts']['timing']={v:str(path.relative_to(ROOT)) for v,path in time_paths.items()}
            for label,key in [('Complete allocated peak','peak_torch_allocated_GiB'),('Complete reserved peak','peak_torch_reserved_GiB'),('Device sampled peak','peak_device_memory_MiB_sampled')]:
                a,b=(max(row[key] for row in timing[v]['rows'])/(1024 if key.endswith('MiB_sampled') else 1) for v in ['native','candidate'])
                table2.append(dict(scene=sid,K=K,metric=label,source='B: uninstrumented Core timing',native_GiB=a,candidate_GiB=b,
                    candidate_minus_native_GiB=b-a,relative_change_percent=100*(b-a)/a))
            a,b=(timing[v]['Core_seconds_mean'] for v in ['native','candidate'])
            sa,sb=(timing[v]['Core_seconds_std'] for v in ['native','candidate'])
            table3.append(dict(scene=sid,K=K,native_Core_mean=a,native_Core_std_sample=sa,candidate_Core_mean=b,candidate_Core_std_sample=sb,
                candidate_minus_native_seconds=b-a,time_change_percent=100*(b-a)/a,native_over_candidate=a/b,
                interpretation='未观察到可靠加速' if a-b<=2*max(sa,sb) else '本轮耗时降低，均值差超过两组各自标准差的2倍；非普遍性能保证'))
        scenes[sid]=data
    aggregate=None
    if len(table3)==4:
        samples={v:[row['core_seconds'] for scene in scenes.values() for row in scene['timing'][v]['rows']] for v in ['native','candidate']}
        a,b=(statistics.mean(samples[v]) for v in ['native','candidate'])
        aggregate=dict(aggregation='Equal scene weight,5 repeats per scene; pooled20-run sample SD includes between-scene differences, not pure timing jitter',
            native_Core_mean=a,candidate_Core_mean=b,native_Core_std_sample=statistics.stdev(samples['native']),candidate_Core_std_sample=statistics.stdev(samples['candidate']),
            time_change_percent=100*(b-a)/a,native_over_candidate=a/b,
            native_full_allocated_max_GiB=max(row['native_GiB'] for row in table2 if row['source'].startswith('B:') and row['metric']=='Complete allocated peak'),
            candidate_full_allocated_max_GiB=max(row['candidate_GiB'] for row in table2 if row['source'].startswith('B:') and row['metric']=='Complete allocated peak'))
    return dict(status=execution['status'],completed_UTC=datetime.now(timezone.utc).isoformat(),
        experiment_config='configs/any2bokeh-encoder-scale1-experiment.json',base_protocol='configs/efficiency-track-a.json',
        source_diagnosis='03_实验记录/records/any2bokeh-native-module-profile-4090d-20261001.json',
        algorithm='Only this ATB VAE encoder ResNet instance forward: non-inplace add, skip divide iff exact output_scale_factor==1; original divide for all other scales.',
        position='ATB implementation efficiency experiment; original baseline retained; not an Ours innovation claim',
        correctness_boundary='Direct32MiB CPU chunk comparison of every warmup/measured RGB cache and latent view. Actual CPU/CUDA/passed-generator states compared byte-for-byte at40 positions. Float bit equality uses uint32 bit views, not numerical max-diff inference.',
        memory_boundary='A: phase synchronization/guard,1 warmup+2 repeats. B: existing uninstrumented Core worker,1 warmup+5 repeats and0.2s device sampling. Independent-run max differences are method contrasts, not same-instant memory decompositions.',
        timing_boundary='Fresh native and candidate processes for each scene/mode,seed0 then continuous warmup/repeats. No correctness copying, history or phase hooks in B. Core is actual three native8/8/9 calls for25 outputs, not extrapolation.',
        reliability_boundary='Within-scene SD uses ddof1 over5 repeats. If native-minus-candidate mean does not exceed twice the larger within-scene SD, label未观察到可靠加速. This conservative descriptive rule is not a causal or universal speed guarantee.',
        historical_uninstrumented_Core_seconds=20.112800140678882,historical_value_used_as_only_control=False,
        table1_correctness=table1,table2_memory=table2,table3_Core=table3,aggregate=aggregate,scenes=scenes,
        paired_history=paired_history(output),
        unchanged=['weights','precision/upcast','batch16','windows/overlap','posterior sampling/RNG','native cache lifetime','decoder','UNet','allocator settings','main baseline efficiency/quality tables','PPT'],
        limitations=['Only the measured fixed inputs are covered; no universal output-equivalence guarantee.', 'A full-run allocated value below32GiB does not show availability on a32GB device.', 'Do not attribute a specific historical4.5GiB allocation solely from max-memory differences.'],
        artifacts=dict(output=str(output.relative_to(ROOT)),execution=str((output/'execution-status.json').relative_to(ROOT))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--record',type=Path,required=True)
    args=parser.parse_args();result=summarize(args.output_dir.resolve())
    with args.record.open('x') as file:
        json.dump(result,file,indent=2,ensure_ascii=False);file.write('\n')
    print(result['status'],len(result['table3_Core']),'timed scenes')


if __name__=='__main__':
    main()
