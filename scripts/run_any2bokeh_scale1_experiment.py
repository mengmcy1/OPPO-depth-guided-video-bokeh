"""Correctness-first, serial native/candidate ATB unit-scale experiment."""
import argparse
import csv
from datetime import datetime, timezone
import json
import mmap
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def rng_states(torch, generator=None):
    def state(value):
        return value.numpy().tobytes().hex()
    result = dict(CPU=state(torch.get_rng_state()), CUDA0=state(torch.cuda.get_rng_state(0)))
    if generator is not None:
        assert not isinstance(generator, list)
        result['explicit_generator'] = dict(device=str(generator.device), state=state(generator.get_state()))
    return result


def release_reference_mapping(array, path):
    """Release only this experiment's CPU reference mapping/file page cache."""
    array._mmap.madvise(mmap.MADV_DONTNEED)
    array._mmap.close()
    with path.open('rb') as file:
        os.posix_fadvise(file.fileno(),0,0,os.POSIX_FADV_DONTNEED)


class CorrectnessTrace:
    """Transfer temporary CPU chunks; never retain a tensor in trace metadata."""
    def __init__(self, state, variant, output, reference):
        import torch
        self.torch, self.state, self.variant = torch, state, variant
        self.pipe, self.output, self.reference = state['pipe'], output, reference
        self.rows, self.originals = [], []
        self.clip, self.unit, self.pending, self.previous_latent = None, -1, None, None

    def tensor(self, tensor, name):
        import numpy as np
        shape, dtype = list(tensor.shape), str(tensor.dtype)
        cpu_dtype = {self.torch.float32:np.float32, self.torch.float16:np.float16}[tensor.dtype]
        path = (self.output if self.variant=='native' else self.reference)/(name+'.npy')
        if self.variant=='native':
            path.parent.mkdir(parents=True, exist_ok=True)
            array = np.lib.format.open_memmap(path, mode='w+', dtype=cpu_dtype, shape=tuple(shape))
        else:
            array = np.load(path, mmap_mode='r')
            assert list(array.shape)==shape and array.dtype==cpu_dtype
        chunk_bytes = 32*2**20
        if tensor.ndim==4:
            tile_rows=max(1,chunk_bytes//(shape[1]*shape[3]*tensor.element_size()))
            slices=[(slice(frame,frame+1),slice(None),slice(row,min(row+tile_rows,shape[2])),slice(None))
                    for frame in range(shape[0]) for row in range(0,shape[2],tile_rows)]
        else:
            assert tensor.numel()*tensor.element_size()<=chunk_bytes
            slices=[tuple(slice(None) for _ in shape)]
        different_bytes = 0
        for selection in slices:
            # Native caches can be channels-last/strided. Preserve GPU layout;
            # any contiguous conversion for byte comparison happens on CPU only.
            cpu = np.ascontiguousarray(tensor[selection].detach().cpu().numpy())
            if self.variant=='native':
                array[selection] = cpu
            else:
                host=np.ascontiguousarray(array[selection])
                different_bytes += int(np.count_nonzero(cpu.view(np.uint8)!=host.view(np.uint8)))
                del host
            del cpu
        if self.variant=='native':
            array.flush()
        release_reference_mapping(array,path)
        del array
        return dict(name=name, shape=shape, dtype=dtype, different_bytes=different_bytes,
            stride=list(tensor.stride()),
            bytes_compared=tensor.numel()*tensor.element_size())

    @staticmethod
    def descriptor(tensor):
        if tensor is None:
            return None
        storage=tensor.untyped_storage()
        return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), storage_pointer=storage.data_ptr(),
            storage_bytes=storage.nbytes(), offset=tensor.storage_offset())

    def attach(self):
        from models.vae import DiagonalGaussianDistribution
        def replace(obj, name, fn):
            original=getattr(obj,name); self.originals.append((obj,name,original));setattr(obj,name,fn(original))
        def meter_wrap(original):
            def observed(label, fn, *args, **kwargs):
                self.unit+=1
                self.rows.append(dict(kind='pipeline_entry', clip=self.clip, unit=self.unit,
                    RNG=rng_states(self.torch, kwargs.get('generator'))))
                result=original(label,fn,*args,**kwargs)
                self.rows.append(dict(kind='pipeline_exit',clip=self.clip,unit=self.unit,
                    RNG=rng_states(self.torch,kwargs.get('generator'))))
                return result
            return observed
        replace(self.state['meter'],'call',meter_wrap)
        def encode_wrap(original):
            def observed(*args,**kwargs):
                result=original(*args,**kwargs)
                self.pending=(id(result.latent_dist),'RGB' if kwargs.get('cache_flag') else 'CoC')
                if kwargs.get('cache_flag'):
                    cache=self.pipe.vae.encoder.residual_down_blocks
                    self.rows.append(dict(kind='RGB_cache',clip=self.clip,unit=self.unit,
                        levels=[self.tensor(value,f'{self.clip}/unit{self.unit}/RGB{level}') for level,value in enumerate(cache)],
                        unique_storage_bytes=sum({(str(v.device),v.untyped_storage().data_ptr()):v.untyped_storage().nbytes() for v in cache}.values())))
                    print('CACHE_CONTENT_CHECK',self.clip,self.unit,self.variant,
                        sum(v['different_bytes'] for v in self.rows[-1]['levels']),flush=True)
                return result
            return observed
        replace(self.pipe.vae,'encode',encode_wrap)
        def sample_wrap(original):
            def observed(distribution,*args,**kwargs):
                assert self.pending is not None and id(distribution)==self.pending[0], 'Unexpected posterior sample'
                role=self.pending[1];self.pending=None
                generator=kwargs.get('generator',args[0] if args else None)
                before=rng_states(self.torch,generator)
                result=original(distribution,*args,**kwargs)
                self.rows.append(dict(kind='posterior_sample',clip=self.clip,unit=self.unit,role=role,
                    shape=list(result.shape),dtype=str(result.dtype),mean_device=str(distribution.mean.device),
                    generator='global_default_on_mean_device' if generator is None else str(generator.device),
                    RNG_before=before,RNG_after=rng_states(self.torch,generator)))
                return result
            return observed
        replace(DiagonalGaussianDistribution,'sample',sample_wrap)
        def decode_wrap(original):
            def observed(*args,**kwargs):
                latent=kwargs.get('latent_cache',args[5] if len(args)>5 else None)
                before=self.descriptor(latent)
                continuity=(before is None) if self.unit==0 else before==self.previous_latent
                captures=[]
                if latent is not None:
                    captures.append(self.tensor(latent,f'{self.clip}/unit{self.unit}/latent_before'))
                result=original(*args,**kwargs)
                after=self.descriptor(result[1]);self.previous_latent=after
                captures.append(self.tensor(result[1],f'{self.clip}/unit{self.unit}/latent_after'))
                self.rows.append(dict(kind='latent_cache',clip=self.clip,unit=self.unit,
                    before=None if before is None else {k:v for k,v in before.items() if k!='storage_pointer'},
                    after={k:v for k,v in after.items() if k!='storage_pointer'},continuity=continuity,contents=captures))
                return result
            return observed
        replace(self.pipe,'decode_latents',decode_wrap)

    def begin_clip(self, name):
        self.clip,self.unit,self.previous_latent=name,-1,None
        self.rows.append(dict(kind='clip_entry',clip=name,RNG=rng_states(self.torch)))
        print('CORRECTNESS_CLIP_START',name,self.variant,flush=True)

    def end_clip(self, outputs):
        import numpy as np
        assert len(outputs)==25 and all(v.shape==(576,1024,3) and np.isfinite(v).all() for v in outputs)
        value=np.stack(outputs)
        assert value.dtype==np.float32
        rgb=np.uint8(np.round(np.clip(value,0,1)*255))
        result=dict(output_frames=25,shape=list(value.shape),finite=True)
        if self.variant=='native':
            np.save(self.output/f'{self.clip}-float.npy',value)
            np.save(self.output/f'{self.clip}-RGB8.npy',rgb)
        else:
            ref=np.load(self.reference/f'{self.clip}-float.npy',mmap_mode='r')
            ref8=np.load(self.reference/f'{self.clip}-RGB8.npy',mmap_mode='r')
            result.update(max_abs_diff=float(np.max(np.abs(value-ref))),
                different_float_elements=int(np.count_nonzero(value!=ref)),
                different_float_bit_patterns=int(np.count_nonzero(value.view(np.uint32)!=ref.view(np.uint32))),
                RGB8_different_bytes=int(np.count_nonzero(rgb!=ref8)))
            np.save(self.output/f'{self.clip}-float.npy',value)
            np.save(self.output/f'{self.clip}-RGB8.npy',rgb)
            release_reference_mapping(ref,self.reference/f'{self.clip}-float.npy')
            release_reference_mapping(ref8,self.reference/f'{self.clip}-RGB8.npy')
        self.rows.append(dict(kind='clip_exit',clip=self.clip,RNG=rng_states(self.torch),output=result))
        return result

    def detach(self):
        for obj,name,original in reversed(self.originals):
            setattr(obj,name,original)


def worker(args):
    import measure_efficiency as efficiency
    from baselines.any2bokeh_scale1 import EncoderScale1
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    common=read(ROOT/'configs/efficiency-track-a.json')
    K=common['K'][common['scenes'].index(args.scene)]
    out=args.output_dir.resolve()
    if args.mode=='timing':
        assert read(args.gate)['passed'], 'Correctness must pass first'
        original_load,original_infer=efficiency.load,efficiency.infer
        patches=[]
        def load(method, config):
            state,runtime=original_load(method,config)
            patch=EncoderScale1(state['pipe'],args.variant=='candidate');patch.observe_warmup();patches.append(patch)
            runtime['scale1_variant']=args.variant
            return state,runtime
        warming=[True]
        def infer(method,state,frames):
            result=original_infer(method,state,frames)
            if warming[0]:
                warming[0]=False
                audit=patches[0].stop_observing()
                assert len(audit['observed_modules'])==audit['modules']==10
                write(out/'warmup-patch-audit.json',audit)
            return result
        efficiency.load,efficiency.infer=load,infer
        config=dict(common,scenes=[args.scene],K=[K],table='3b',output=str(out.relative_to(ROOT)),skip_flop_diagnostic=True)
        efficiency.worker('any2bokeh',config)
        summary=read(out/'any2bokeh/summary.json')
        write(out/'summary.json',summary)
        return
    efficiency.assert_idle_gpu()
    out.mkdir(parents=True,exist_ok=False)
    state,runtime=efficiency.load('any2bokeh',common)
    patch=EncoderScale1(state['pipe'],args.variant=='candidate');patch.observe_warmup()
    folder=ROOT/common['input']/args.scene;csvpath=out/(args.scene+'.csv')
    with csvpath.open('x') as file:
        csv.writer(file).writerows([['aif_folder','disp_folder','k'],[str(folder/'rgb'),str(folder/'disparity'),K]])
    frames=efficiency.prepare('any2bokeh',state,folder,K,csvpath,25)
    import torch
    import numpy as np
    runtime.update(project_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        native_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent/'any-to-bokeh',text=True).strip(),
        GPU=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
        CPU=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        variant=args.variant,allocator_backend=torch.cuda.memory.get_allocator_backend())
    if args.mode=='correctness':
        observer=CorrectnessTrace(state,args.variant,out,args.reference_dir)
        observer.attach();outputs=[]
        try:
            for clip in ['warmup','measured']:
                assert set(efficiency.gpu_processes())<={os.getpid()}
                observer.begin_clip(clip);state['meter'].reset()
                values=efficiency.infer('any2bokeh',state,frames)
                assert [x['output_frames'] for x in state['meter'].units]==[8,8,9]
                outputs.append(observer.end_clip(values));del values
                if clip=='warmup':audit=patch.stop_observing()
        finally:
            observer.detach()
            write(out/'trace.json',observer.rows)
        samples=[row for row in observer.rows if row['kind']=='posterior_sample']
        assert len(samples)==12 and [x['role'] for x in samples]==['RGB','CoC']*6
        assert len(audit['observed_modules'])==audit['modules']==10
        passed=None
        if args.variant=='candidate':
            reference=read(args.reference_dir/'trace.json')
            # GPU addresses are intentionally absent; compare actual RNG bytes, shapes and ordered content checks.
            def semantic(rows):
                return [{k:v for k,v in row.items() if k!='output'} for row in rows]
            ordered_equal=semantic(reference)==semantic(observer.rows)
            caches=[row for row in observer.rows if row['kind'] in ['RGB_cache','latent_cache']]
            cache_equal=all(v['different_bytes']==0 for row in caches for v in row.get('levels',row.get('contents',[])))
            continuity=all(row['continuity'] for row in caches if row['kind']=='latent_cache')
            passed=ordered_equal and cache_equal and continuity and all(row['different_float_bit_patterns']==0 and row['RGB8_different_bytes']==0 for row in outputs)
        else:ordered_equal=cache_equal=continuity=None
        summary=dict(status='CORRECTNESS_REFERENCE_COMPLETE' if args.variant=='native' else 'CORRECTNESS_COMPARISON_COMPLETE',scene=args.scene,K=K,runtime=runtime,
            warmups=1,observations=1,performance_measurement=False,passed=passed,
            ordered_samples_RNG_and_metadata_equal=ordered_equal,cache_contents_equal=cache_equal,latent_continuity_equal=continuity,
            posterior_calls=len(samples),posterior_calls_per_clip=6,outputs=outputs,warmup_patch_audit=audit,
            cache_comparison='All six warmup/measured RGB encoder caches transferred in32MiB CPU chunks; candidate bytes directly compared to CPU memmaps. No retained GPU tensor copies.')
        write(out/'summary.json',summary)
        print('CORRECTNESS',args.scene,args.variant,passed,outputs,flush=True)
        return
    assert read(args.gate)['passed'], 'Correctness must pass first'
    from profile_any2bokeh_modules import NativeStages
    profiler=NativeStages(state['pipe']);profiler.attach();runs=[]
    try:
        state['meter'].reset();values=efficiency.infer('any2bokeh',state,frames)
        assert len(values)==25;del values
        audit=patch.stop_observing()
        assert len(audit['observed_modules'])==audit['modules']==10
        for repeat in range(2):
            assert set(efficiency.gpu_processes())<={os.getpid()}
            profiler.reset();profiler.enabled=True;state['meter'].reset()
            values=efficiency.infer('any2bokeh',state,frames)
            assert len(values)==25 and all(v.shape==(576,1024,3) and np.isfinite(v).all() for v in values)
            assert [x['output_frames'] for x in state['meter'].units]==[8,8,9]
            profiler.capture_peaks()
            row=dict(scene=args.scene,K=K,repeat=repeat,stages=profiler.rows,
                observed_peak_allocated_GiB=profiler.observed_peak/2**30,observed_peak_reserved_GiB=profiler.observed_reserved_peak/2**30,
                peak_reset_events=profiler.reset_events,encoder_calls=profiler.encoder_calls,decoder_calls=profiler.decoder_calls,
                native_output_units=[8,8,9],output_frames=25,shape=[576,1024,3],finite=True)
            assert len(row['stages'])==15 and len(row['encoder_calls'])==6 and len(row['decoder_calls'])==4
            runs.append(row);write(out/f'repeat{repeat}.json',row);del values
            print('PHASE_MEMORY',args.scene,args.variant,repeat,row['observed_peak_allocated_GiB'],flush=True)
    finally:
        profiler.detach()
    write(out/'summary.json',dict(status='PHASE_MEMORY_COMPLETE',scene=args.scene,K=K,runtime=runtime,
        warmups=1,repeats=2,runs=runs,warmup_patch_audit=audit,
        boundary='Phase synchronization/reset guard; memory diagnostic, not formal time'))


def orchestrate(args):
    output,logs=args.output_dir.resolve(),args.logs_dir.resolve()
    output.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    common=read(ROOT/'configs/efficiency-track-a.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    completed=[]
    for sid,K in zip(common['scenes'],common['K']):
        gate=output/sid/'correctness/candidate/summary.json'
        for mode in ['correctness','phase','timing']:
            for variant in ['native','candidate']:
                folder=output/sid/mode/variant;summary=folder/'summary.json'
                if summary.exists():
                    print('REUSE_COMPLETE',sid,mode,variant,flush=True)
                else:
                    if folder.exists():
                        folder.rename(folder.with_name(variant+'-incomplete-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')))
                    cmd=[sys.executable,str(Path(__file__).resolve()),'--mode',mode,'--variant',variant,'--scene',sid,'--output-dir',str(folder)]
                    if mode=='correctness' and variant=='candidate':cmd+=['--reference-dir',str(output/sid/'correctness/native')]
                    if mode!='correctness':
                        assert read(gate)['passed'], 'Correctness failure: stop before performance'
                        cmd+=['--gate',str(gate)]
                    print('START',sid,mode,variant,flush=True)
                    with (logs/f'{sid}-{mode}-{variant}.log').open('a') as log:
                        subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                if mode=='correctness' and variant=='candidate' and not read(summary)['passed']:
                    write(output/'execution-status.json',dict(status='STOPPED_CORRECTNESS_FAILURE',scene=sid,evidence=str(summary.relative_to(ROOT))))
                    return
                completed.append(dict(scene=sid,mode=mode,variant=variant,summary=str(summary.relative_to(ROOT))))
                write(output/'execution-status.json',dict(status='RUNNING',completed=completed))
        # A representative no-memory-reduction result requires attribution review, not automatic expansion.
        if sid==common['scenes'][0]:
            peaks=[]
            for variant in ['native','candidate']:
                phase=read(output/sid/f'phase/{variant}/summary.json')
                peaks.append(max(row['observed_peak_allocated_GiB'] for row in phase['runs']))
            if peaks[1]>=peaks[0]-0.05:
                write(output/'execution-status.json',dict(status='REVIEW_REPRESENTATIVE_PEAK_NOT_REDUCED',peaks=peaks,completed=completed))
                return
        print('SCENE_COMPLETE',sid,K,flush=True)
    write(output/'execution-status.json',dict(status='FOUR_SCENE_EXPERIMENT_COMPLETE',completed=completed,completed_UTC=datetime.now(timezone.utc).isoformat()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['all','correctness','phase','timing'],default='all')
    p.add_argument('--variant',choices=['native','candidate'])
    p.add_argument('--scene')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--logs-dir',type=Path)
    p.add_argument('--reference-dir',type=Path)
    p.add_argument('--gate',type=Path)
    args=p.parse_args()
    if args.mode=='all':orchestrate(args)
    else:worker(args)


if __name__=='__main__':
    main()
