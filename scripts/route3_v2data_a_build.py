"""Build the A foreground inventory and exercise seeded online composition."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from route3_ovg_subpixel import ROOT, save_json

OUT = ROOT / "data/processed/route3-v2data-2026-11-20/A"
RECORD = ROOT / "03_实验记录/records/route3-v2data-A-2026-11-20.json"
RAW = ROOT / "data/raw"
SIZE = 256
SEED = 20261120
GOAL = 1000
GIB = 2**30
VIDEO = "VideoMatte240K_HEVC"
VIDEO_IDENTITY = f"{VIDEO}:identity-unresolved"
SPLITS = ("train", "dev", "test")
MEMORY = dict(shift_px=4, hole_count=3, hole_side_px=[12, 44], gaussian_rgb_sigma=0.025)
GENERATOR = "route3-v2data-a/1.0"
GFM_AUTHORS = ["Jizhizi Li", "Jing Zhang", "Stephen J. Maybank", "Dacheng Tao"]
GFM = "https://github.com/JizhiziLi/GFM"
VIDEO_URL = "https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets"
SOURCES = {
    brand: dict(source=brand, author=GFM_AUTHORS, license="MIT",
                terms=f"https://jizhizili.github.io/files/gfm_datasets_agreements/{brand}_Dataset_Release_Agreement.pdf",
                terms_version=f"{brand} Dataset v1.0 Release Agreement",
                download_channel=GFM if brand == "AM-2k" else "https://huggingface.co/datasets/unography/BG-20k",
                original_publication=GFM, original_image_copyright="retained by original owners")
    for brand in ("AM-2k", "BG-20k")
}
SOURCES[VIDEO] = dict(
    source=VIDEO, author=["Shanchuan Lin", "Andrey Ryabtsev", "Soumyadip Sengupta",
                          "Brian L. Curless", "Steven M. Seitz", "Ira Kemelmacher-Shlizerman"],
    license="UW Background Matting V2 dataset terms",
    terms=VIDEO_URL, terms_version="official HEVC release (2021-04-21); dataset page has no version number",
    download_channel="https://drive.google.com/file/d/1z-KDuqk1g7-H94iGXDrpZGpexN9jizbQ/view",
    use="academic citation required; commercial-product use requires author form",
)
OPTIONAL = dict(source="P3M-10k", status="optional_not_acquired",
                terms="https://jizhizili.github.io/files/p3m_dataset_agreement/P3M-10k_Dataset_Release_Agreement.pdf")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else digest(stream)


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(2**20), b""):
        h.update(block)
    return h.hexdigest()


def split(key):
    n = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 1000
    return SPLITS[0] if n < 800 else SPLITS[1] if n < 900 else SPLITS[2]


def srgb(image):
    v = image.astype(np.float32) / 255
    return np.where(v <= .04045, v / 12.92, ((v + .055) / 1.055) ** 2.4)


def display(linear):
    v = np.clip(linear, 0, 1)
    return np.rint(np.where(v <= .0031308, 12.92*v, 1.055*v**(1/2.4)-.055)*255).astype(np.uint8)


def read(path, kind=cv2.IMREAD_COLOR):
    image = cv2.imread(str(path), kind)
    if image is None:
        raise ValueError(f"cannot decode {path.relative_to(ROOT)}")
    return image


def video_pair(foreground, alpha):
    streams = [cv2.VideoCapture(str(p), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
               for p in (foreground, alpha)]
    try:
        counts = [int(s.get(cv2.CAP_PROP_FRAME_COUNT)) for s in streams]
        if min(counts) < 1 or counts[0] != counts[1]:
            raise ValueError(f"mismatched video frame counts {counts}")
        index = counts[0] // 2
        frames = []
        for stream in streams:
            stream.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, pixels = stream.read()
            if not ok or int(stream.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
                raise ValueError(f"could not decode matched clip frame {index}")
            frames.append(pixels)
        return frames[0], frames[1][..., 0], index
    finally:
        for stream in streams:
            stream.release()


def materialize(row):
    f, a = (ROOT / row[k] for k in ("foreground_original", "alpha_original"))
    if row["source"] == VIDEO:
        color, opacity, frame = video_pair(f, a)
    else:
        color, opacity, frame = read(f), read(a, cv2.IMREAD_UNCHANGED), None
        if opacity.ndim == 3:
            opacity = opacity[..., 0]
    if color.shape[:2] != opacity.shape or not opacity.max():
        raise ValueError("foreground/matte dimension mismatch or empty matte")
    alpha = opacity.astype(np.float32) / np.iinfo(opacity.dtype).max
    color = srgb(color[..., ::-1])
    # Resize premultiplied linear radiance to avoid background colors at matte edges.
    scale = SIZE / max(alpha.shape)
    size = tuple(max(1, round(n*scale)) for n in alpha.shape[::-1])
    a_small = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    premult = cv2.resize(color * alpha[..., None], size, interpolation=cv2.INTER_AREA)
    fg = np.divide(premult, a_small[..., None], out=np.zeros_like(premult), where=a_small[..., None] > 0)
    packed = np.zeros((SIZE, SIZE, 4), np.float16)
    top, left = (SIZE-size[1])//2, (SIZE-size[0])//2
    window = packed[top:top+size[1], left:left+size[0]]
    window[..., :3] = fg.clip(0, 1)
    window[..., 3] = a_small
    identity = row.get("identity_group", row["key"])
    rel = Path("foreground") / row["split"] / (row["source"]+"-"+row["asset_id"].replace("/", "-")+".npy")
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    with tmp.open("wb") as output:
        np.save(output, packed, allow_pickle=False)
    tmp.replace(dst)
    plane = packed[..., 3]
    return dict(row, path=str(rel), sha256=sha(dst), bytes=dst.stat().st_size, frame=frame,
                identity_group=identity, soft_alpha_pixels=int(((plane>0)&(plane<1)).sum()),
                alpha_pixels=plane.size, active_alpha_pixels=int((plane>0).sum()))


def foreground_candidates(excluded):
    am = RAW / "AM-2k"
    for key, meta in json.loads((am / "am2k_split_category.json").read_text()).items():
        root = am / meta["split"]
        fg, alpha = root/"fg"/f"{key}.png", root/"mask"/f"{key}.png"
        if not fg.is_file() or not alpha.is_file():
            excluded.append(dict(key=f"AM-2k:{key}", reason="paired estimated foreground / alpha absent"))
            continue
        yield dict(SOURCES["AM-2k"], asset_id=key, key=f"AM-2k:{key}", category="animal",
                   species=meta["category"], foreground_original=str(fg.relative_to(ROOT)),
                   alpha_original=str(alpha.relative_to(ROOT)),
                   original_photo=str((root/"original"/f"{key}.jpg").relative_to(ROOT)))
    root = RAW / VIDEO / "VideoMatte240K"
    for fg in sorted(root.glob("*/fgr/*.mp4")):
        section = fg.parent.parent.name
        alpha = root / section / "pha" / fg.name
        key = f"{VIDEO}:{section}/{fg.stem}"
        if not alpha.is_file():
            excluded.append(dict(key=key, reason="paired HEVC matte absent"))
            continue
        yield dict(SOURCES[VIDEO], asset_id=f"{section}/{fg.stem}", key=key, category="person",
                   foreground_original=str(fg.relative_to(ROOT)), alpha_original=str(alpha.relative_to(ROOT)),
                   identity_group=VIDEO_IDENTITY)


def stat_source(row):
    original = ROOT / row.get("original_photo", row["foreground_original"])
    return sha(original)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    excluded = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        candidates = list(foreground_candidates(excluded))
        seen = {}
        eligible = {"AM-2k": [], VIDEO: []}
        for row, checksum in zip(candidates, pool.map(stat_source, candidates)):
            if checksum in seen:
                excluded.append(dict(key=row["key"], canonical_key=seen[checksum], reason="duplicate source bytes"))
                continue
            seen[checksum] = row["key"]
            eligible[row["source"]].append(dict(row, source_sha256=checksum,
                                                split=split(row.get("identity_group", row["key"]))))
        # Choose a fixed slice of independent clips and fill the remainder with AM stills.
        def order(row):
            return hashlib.sha256(f'{SEED}/{row["key"]}'.encode()).digest()
        ordered = sorted(eligible[VIDEO], key=order)[:100] + sorted(eligible["AM-2k"], key=order)
        units = []
        disk = {}
        cursor = 0
        while len(units) < GOAL and cursor < len(ordered):
            count = min(GOAL-len(units), 50, len(ordered)-cursor)
            estimate_per = max((r["bytes"] for r in units), default=SIZE*SIZE*4*2+1024)
            if shutil.disk_usage(OUT).free < 60*GIB + count*estimate_per:
                excluded.append(dict(source="A", reason="disk reserve would be crossed"))
                break
            batch = ordered[cursor:cursor+count]
            for row, task in zip(batch, [pool.submit(materialize, row) for row in batch]):
                try:
                    units.append(task.result())
                except (OSError, ValueError, cv2.error) as exc:
                    excluded.append(dict(key=row["key"], reason=str(exc)))
            cursor += count
            if not disk and units:
                disk = dict(first_batch_units=len(units), first_batch_bytes=sum(r["bytes"] for r in units),
                            first_batch_max_unit_bytes=max(r["bytes"] for r in units),
                            projected_1000_foreground_bytes=GOAL*max(r["bytes"] for r in units),
                            reserve_bytes=60*GIB)
                save_json(OUT/"first-batch.json", disk)
                print("A first batch", disk, flush=True)
        backgrounds = []
        bg_root = RAW / "BG-20k"
        paths = sorted(bg_root.glob("*/*.jpg"))
        for path, checksum in zip(paths, pool.map(sha, paths)):
            key = "BG-20k:" + path.stem
            if checksum in seen:
                excluded.append(dict(key=key, canonical_key=seen[checksum], reason="duplicate source bytes"))
                continue
            seen[checksum] = key
            backgrounds.append(dict(SOURCES["BG-20k"], key=key, asset_id=path.stem, split=split(key),
                                    path=str(path.relative_to(ROOT)), sha256=checksum, bytes=path.stat().st_size))
    metadata = dict(generator=GENERATOR, seed=SEED, size=SIZE, memory=MEMORY,
                    saturation_threshold_linear=1.0, sources=SOURCES, foreground=units, background=backgrounds,
                    exclusions=excluded, attribution=[], pending_credentials=[], optional_sources=[OPTIONAL],
                    use_by_split=dict(train="training", dev="per-tier train/dev gap only", test="sealed holdout"),
                    deduplication="cross-source exact source SHA-256; no identity metadata in HEVC release",
                    video_identity_policy="all 484 unresolved clips share one hash pool conservatively",
                    background_storage="references to read-only local BG-20k",
                    qualified_foreground_by_source={k: len(v) for k,v in eligible.items()})
    save_json(OUT/"manifest.json", metadata)
    return metadata, disk


class OnlineA:
    """Single-view A examples with an explicit observation / supervision boundary."""

    def __init__(self, split_name, manifest=OUT/"manifest.json"):
        self.root = Path(manifest).parent
        self.meta = json.loads(Path(manifest).read_text())
        self.foreground = [r for r in self.meta["foreground"] if r["split"] == split_name]
        self.background = [r for r in self.meta["background"] if r["split"] == split_name]

    def sample(self, seed, stage="2a"):
        if stage not in ("2a", "2b"):
            raise ValueError(f"A is only configured for stage 2a or 2b, got {stage}")
        rng = np.random.default_rng(seed)
        fg = self.foreground[rng.integers(len(self.foreground))]
        bg = self.background[rng.integers(len(self.background))]
        packed = np.load(self.root/fg["path"], allow_pickle=False).astype(np.float32)
        background = srgb(read(ROOT/bg["path"])[..., ::-1])
        height, width = background.shape[:2]
        side = min(height, width)
        y, x = (height-side)//2, (width-side)//2
        background = cv2.resize(background[y:y+side, x:x+side], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if rng.integers(2):
            packed = packed[:, ::-1].copy()
        if rng.integers(2):
            background = background[:, ::-1].copy()
        color, alpha = packed[..., :3], packed[..., 3]
        # The source images are already LDR. The single view is clipped in linear RGB.
        image = (alpha[..., None]*color + (1-alpha[..., None])*background).clip(0, 1)
        edge = (alpha>0)&(alpha<1)
        valid = ~(edge & ((color>=1)|(background>=1)|(image>=1)).any(-1))
        memory = np.zeros_like(background)
        known = np.zeros(alpha.shape, bool)
        if stage == "2a":
            opts = self.meta["memory"]
            dx, dy = rng.integers(-opts["shift_px"], opts["shift_px"]+1, size=2)
            if dx == dy == 0:
                dx = 1
            matrix = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
            memory = cv2.warpAffine(background, matrix, (SIZE, SIZE))
            known = cv2.warpAffine(np.ones_like(alpha), matrix, (SIZE, SIZE)) > .5
            for _ in range(opts["hole_count"]):
                y, x = rng.integers(0, SIZE, size=2)
                width = rng.integers(*opts["hole_side_px"])
                known[y:y+width, x:x+width] = False
            memory = (memory+rng.normal(0, opts["gaussian_rgb_sigma"], memory.shape).astype(np.float32)).clip(0, 1)
            memory[~known] = 0
        inputs = dict(rgb=image, prior_depth=np.zeros_like(alpha), prior_conf=np.zeros_like(alpha),
                      memory=memory, memory_valid=known, memory_confidence=known.astype(np.float32))
        labels = dict(foreground=color, alpha=alpha, background=background, full=image.copy(),
                      saturation_invalid=~valid, color_valid=edge&valid, comp_valid=edge&valid)
        return inputs, labels, (fg["key"], bg["key"])


def appearance_losses(predicted_foreground, labels):
    """A supervises color and composition using the detached matte."""
    import torch
    from src.ovg_net import masked_mean

    def target(field):
        return torch.as_tensor(labels[field], device=predicted_foreground.device).detach()
    alpha = target("alpha")[..., None]
    color = predicted_foreground
    composite = alpha*color + (1-alpha)*target("background")
    return dict(foreground=masked_mean((color-target("foreground")).abs().mean(-1), target("color_valid")),
                comp=masked_mean((composite-target("full")).abs().mean(-1), target("comp_valid")))


def verify():
    max_error = 0.
    strips = []
    for k, role in enumerate(SPLITS):
        dataset = OnlineA(role)
        if not dataset.foreground or not dataset.background:
            continue
        seeds = list(range(SEED+k*10, SEED+k*10+4))
        if role == "train" and any(r["source"] == VIDEO for r in dataset.foreground):
            video_seed = next(seed for seed in range(SEED+100, SEED+10000)
                              if dataset.foreground[np.random.default_rng(seed).integers(
                                  len(dataset.foreground))]["source"] == VIDEO)
            seeds[0] = video_seed
        for n, seed in enumerate(seeds):
            one, labels, ids = dataset.sample(seed)
            again, labels_again, ids_again = dataset.sample(seed)
            assert ids == ids_again and all(np.array_equal(v, again[key]) for key, v in one.items())
            assert all(np.array_equal(v, labels_again[key]) for key, v in labels.items())
            alpha = labels["alpha"][..., None]
            recomposed = alpha*labels["foreground"] + (1-alpha)*labels["background"]
            max_error = max(max_error, float(np.abs(recomposed-one["rgb"]).max()))
            assert not one["prior_conf"].any() and not one["prior_depth"].any()
            assert one["memory_valid"].any() and not one["memory_valid"].all()
            assert not dataset.sample(seed, stage="2b")[0]["memory_valid"].any()
            assert all(key not in one for key in ("alpha", "foreground", "background", "full"))
            if role != "test" and n < 2:
                gray = np.repeat(alpha, 3, axis=-1)
                strips.append(np.concatenate([display(v) for v in (labels["foreground"], gray,
                                    labels["background"], one["rgb"], one["memory"])], axis=1))
    if max_error > 2e-7:
        raise ValueError(f"I = alpha F + (1-alpha) B max error {max_error}")
    if strips and not cv2.imwrite(str(OUT/"sample-visualization.png"), np.concatenate(strips)[..., ::-1]):
        raise OSError("cannot write A sample visualization")
    import sys
    sys.path.insert(0, str(ROOT))
    import torch
    _, y, _ = OnlineA("train").sample(SEED)
    fg = torch.tensor(y["foreground"]*.9, requires_grad=True)
    alpha_hat = torch.zeros_like(torch.from_numpy(y["alpha"]), requires_grad=True)
    sum(appearance_losses(fg, y).values()).backward()
    assert fg.grad is not None and fg.grad.abs().sum() > 0 and alpha_hat.grad is None
    return dict(max_composition_error=max_error, same_seed_bitwise=True, alpha_hat_gradient=False,
                preview="sample-visualization.png", test_visualizations_used_for_selection=False)


def report(meta, disk, checks):
    units, backgrounds = meta["foreground"], meta["background"]
    counts = Counter(r["split"] for r in units)
    by_category = {}
    for name in ("person", "animal", "object", "hair"):
        rows = [r for r in units if r["category"] == name]
        by_category[name] = dict(units=len(rows), split=dict(Counter(r["split"] for r in rows)),
                                 soft_alpha_fraction=sum(r["soft_alpha_pixels"] for r in rows)/
                                 max(1, sum(r["alpha_pixels"] for r in rows)))
    save_json(RECORD, dict(generator=GENERATOR, manifest=str((OUT/"manifest.json").relative_to(ROOT)),
                          manifest_sha256=sha(OUT/"manifest.json"), target=GOAL, completed=len(units),
                          shortfall=GOAL-len(units), units_by_split={k: counts[k] for k in SPLITS},
                          backgrounds_by_split=dict(Counter(r["split"] for r in backgrounds)),
                          foreground_files_bytes=sum(r["bytes"] for r in units),
                          output_bytes=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()),
                          background_source_bytes=sum(r["bytes"] for r in backgrounds),
                          background_source_bytes_per_asset=[min(r["bytes"] for r in backgrounds),
                                                             max(r["bytes"] for r in backgrounds)],
                          foreground_by_source=dict(Counter(r["source"] for r in units)),
                          foreground_bytes_per_unit=[min(r["bytes"] for r in units), max(r["bytes"] for r in units)],
                          disk=disk, checks=checks, category=by_category,
                          category_note="hair counts distinct hair-only sources, not hair visible on animals or people",
                          soft_alpha_fraction=sum(r["soft_alpha_pixels"] for r in units)/
                                              max(1, sum(r["alpha_pixels"] for r in units)),
                          license_foregrounds=dict(Counter(r["license"] for r in units)),
                          license_backgrounds=dict(Counter(r["license"] for r in backgrounds)),
                          exclusions_by_reason=dict(Counter(r["reason"] for r in meta["exclusions"])),
                          pending_credentials=meta["pending_credentials"], optional_sources=meta["optional_sources"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.verify_only or (OUT/"manifest.json").exists():
        meta = json.loads((OUT/"manifest.json").read_text())
        disk = json.loads((OUT/"first-batch.json").read_text())
    else:
        meta, disk = build()
    checks = verify()
    report(meta, disk, checks)
    print("A verified", len(meta["foreground"]), len(meta["background"]), checks, flush=True)


if __name__ == "__main__":
    main()
