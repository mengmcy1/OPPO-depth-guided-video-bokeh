"""A separate gated native/candidate history on the representative clip only."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['native','candidate'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gate',type=Path,required=True)
    args=parser.parse_args()
    assert json.loads(args.gate.read_text())['passed'], 'Correctness first'
    import measure_efficiency as efficiency
    from baselines.any2bokeh_scale1 import EncoderScale1
    original_load=efficiency.load
    def load(method,config):
        state,runtime=original_load(method,config)
        state['scale1_patch']=EncoderScale1(state['pipe'],args.variant=='candidate')
        runtime['scale1_variant']=args.variant
        return state,runtime
    efficiency.load=load
    import profile_any2bokeh_modules as profile
    sys.argv=[sys.argv[0],'--allocator-supplement','--memory-history','--output-dir',str(args.output_dir)]
    profile.main()
    profile.analyze_memory_history(args.output_dir)


if __name__=='__main__':
    main()
