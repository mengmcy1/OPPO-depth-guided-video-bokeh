#!/usr/bin/env python3
"""
汇总所有 metrics-*.json 生成最终 200 场景的统计
"""
import json
import glob
from pathlib import Path
import numpy as np

def aggregate_metrics(output_dir):
    output_dir = Path(output_dir)
    
    # 读取所有 metrics-*.json
    metric_files = sorted(output_dir.glob("metrics-*.json"))
    
    all_scenes = []
    for mf in metric_files:
        with open(mf) as f:
            data = json.load(f)
            all_scenes.extend(data['rows'])
    
    print(f"Total scenes loaded: {len(all_scenes)}")
    
    # 计算总体平均
    metrics = ['PSNR', 'PSNR_clip_mse', 'SSIM', 'LPIPS', 'FD', 'RM']
    overall = {}
    for m in metrics:
        values = [s[m] for s in all_scenes if s[m] is not None]
        overall[m] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
    
    # 按 K 值分组（从场景名推断）
    by_k = {}
    for scene in all_scenes:
        # 从 metadata 或场景名推断 K 值（需要读取 metadata.json）
        scene_name = scene['scene']
        scene_dir = output_dir / 'clips' / scene_name
        metadata_file = scene_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file) as f:
                meta = json.load(f)
                k = meta.get('K', 'unknown')
        else:
            k = 'unknown'
        
        if k not in by_k:
            by_k[k] = []
        by_k[k].append(scene)
    
    # 计算每个 K 值的统计
    k_stats = {}
    for k, scenes in by_k.items():
        k_stats[str(k)] = {
            'count': len(scenes),
            'metrics': {}
        }
        for m in metrics:
            values = [s[m] for s in scenes if s[m] is not None]
            if values:
                k_stats[str(k)]['metrics'][m] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
    
    # 汇总结果
    result = {
        'total_scenes': len(all_scenes),
        'overall': overall,
        'by_k': k_stats,
        'top_5_psnr': sorted(all_scenes, key=lambda x: x['PSNR'], reverse=True)[:5],
        'bottom_5_psnr': sorted(all_scenes, key=lambda x: x['PSNR'])[:5]
    }
    
    # 保存
    output_file = output_dir / 'final_summary_200.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Final summary saved to: {output_file}")
    print(f"\n{'='*60}")
    print("📊 200-VIDEO FINAL RESULTS")
    print(f"{'='*60}")
    print(f"\nTotal scenes: {len(all_scenes)}")
    print(f"\n{'Overall Metrics':<20} {'Mean':<12} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 62)
    for m in ['PSNR', 'SSIM', 'LPIPS', 'FD', 'RM']:
        stats = overall[m]
        print(f"{m:<20} {stats['mean']:<12.4f} {stats['std']:<10.4f} {stats['min']:<10.4f} {stats['max']:<10.4f}")
    
    print(f"\n{'K Value Stats':<20}")
    print("-" * 62)
    for k in sorted(k_stats.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        count = k_stats[k]['count']
        psnr = k_stats[k]['metrics'].get('PSNR', {}).get('mean', 0)
        ssim = k_stats[k]['metrics'].get('SSIM', {}).get('mean', 0)
        print(f"K={k:<4} ({count:>3} scenes): PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    
    return result

if __name__ == '__main__':
    result = aggregate_metrics('outputs/any2bokeh-mpib-public-proxy-200-20260928')
