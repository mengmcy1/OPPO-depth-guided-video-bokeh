"""Independent shard inventory and continuations of the stage 1a data builders."""
import argparse
from copy import deepcopy
import subprocess

from route3_ovg_1b_common import CONFIG, OUT, SOURCE, guard, link
from route3_ovg_r2_common import CONFIG as ROUND2
import route3_ovg_data as data


def setup():
    guard()
    for name in ('B', 'B-thin-texture', 'C/mesh-check.json'):
        link(SOURCE / name, OUT / name)
    for tier in ('C', 'G'):
        for manifest in (SOURCE / tier / 'windows').glob('*/manifest.json'):
            link(manifest.parent, OUT / tier / 'windows' / manifest.parent.name)
    data.OUT = OUT


def expansion_config():
    config = deepcopy(ROUND2)
    config['G_round2']['environments'].remove('Carwelding')
    n = len(config['G_round2']['environments'])
    config['G_round2']['windows_per_environment'] = (
        CONFIG['tartanair_total_frame_target'] + n * 8 - 1) // (n * 8)
    config['C_round2']['first_seed'] += config['C_round2']['sequences']
    config['C_round2']['sequences'] = CONFIG['card_total_target'] - 49
    return config


def expand(job):
    setup()
    config = expansion_config()
    if job == 'C':
        import route3_ovg_r2_cards as builder
        builder.OUT, builder.CONFIG, builder.guard = OUT, config, guard
        builder.cards()
    elif job == 'G':
        import route3_ovg_r2_tartan as builder
        builder.OUT, builder.CONFIG, builder.guard = OUT, config, guard
        builder.main()
    else:
        import route3_ovg_r2_priors as builder
        builder.OUT, builder.CONFIG, builder.guard = OUT, config, guard
        builder.main(watch=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', choices=['C', 'G', 'priors'])
    expand(parser.parse_args().job)
