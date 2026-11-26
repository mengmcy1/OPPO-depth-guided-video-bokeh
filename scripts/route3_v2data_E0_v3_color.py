"""Reuse the registered witness-colour cache in the E0 v3 far comparison."""
from collections import defaultdict
from functools import lru_cache

import route3_v2data_module2_infer as ref
from route3_v2data_E_labels import load
from route3_v2data_hidden_color_purity import render_chunked, substitute_alternatives
from route3_v2data_hidden_color_sources import read_witnesses


S2="S2@0.90"
WITNESS="far_witness"
PEELED="far_S3_f030"
PROTOCOL=dict(cache_commit="c686d5d",development_reference=S2,legacy_reference="same-frame peeled S0",
    F2="Geometry performance without the hidden layer.",
    S2="Geometry performance with ideal witness colours where a qualified witness exists.",
    S2_minus_F2="The contribution needed from the actual hidden layer, on the same pixels.",
    old_development_S0="Old colour protocol with known contamination.",
    qualification="Median per linear RGB channel among point-visible witnesses with 4x4 occupancy >= 0.90. Missing qualified parents fall back to F2.",
    f030="Compare S3 and S2 on exactly the same f030 development far cases.")


@lru_cache(maxsize=12)
def witnesses(scene,case):
    parents,variants,_,_=read_witnesses(scene,{"id":case})
    chosen={k:variants[k] for k in (S2,"S3") if k in variants}
    return parents,chosen


def extra_far(geometry,unit,scene,frame,native,case,K,focus,offsets,fill,s0,f2):
    if unit is None:
        return {WITNESS:s0}
    parents,variants=witnesses(scene,case["id"])
    keys={S2:WITNESS,"S3":PEELED}
    return {keys[name]:render_chunked(geometry[0],cloud,K,focus,offsets,fill) if cloud is not None else f2
            for name,cloud in substitute_alternatives(geometry,unit,scene,frame,native,K,parents,variants)}


def coverage(record):
    old=load(record)["far_witness_by_scene"]
    result={}
    for name,row in old.items():
        if not name.startswith("development-"):
            continue
        qualified=int(row["any_at_threshold"]["0.90"])
        total=row["parents"]
        result[name]=dict(parents=total,qualified=qualified,fallback=total-qualified,
                          qualified_fraction=qualified/total,fallback_fraction=(total-qualified)/total,
                          limited_reference=ref.title(name) in ("dev000","dev014"))
    if len(result)!=15:
        raise ValueError("Expected the registered far witness census for 15 development scenes")
    return result


def far_witness_table(rows, arms, *, legacy=False):
    active=[r for r in rows if r["focus_index"]==1 and r["status"]=="evaluated"]
    f030=[r for r in active if not legacy and r["frame"]==30]

    def summarize(subset,arm,field):
        per_scene=defaultdict(list)
        for r in subset:
            item=r["methods"][arm]
            candidate=item["optical"]["far_recoverable"] if field=="old_S0" else item[field]
            baseline=item["far_off"]
            if candidate["pixels"]!=baseline["pixels"]:
                raise ValueError(f"Far reference and F2 have unequal scored support in {r['scene']}/{r['case']}/{arm}")
            per_scene[r["scene"]].append((baseline,candidate))
        def pair(values):
            a,b=(ref.pooled([q[i] for q in values]) for i in (0,1))
            return dict(F2=a,reference=b,delta_vs_F2_db=b["psnr_db"]-a["psnr_db"] if a["pixels"] else None)
        grouped={name:pair(values) for name,values in per_scene.items()}
        differences=[r["delta_vs_F2_db"] for r in grouped.values() if r["delta_vs_F2_db"] is not None]
        return dict(case_count=len(subset),pooled=pair([p for g in per_scene.values() for p in g]),
                    paired_vs_same_arm_F2=ref.distribution(differences),by_scene=grouped)

    methods={}
    for arm in arms:
        values=dict(primary_reference=summarize(active,arm,WITNESS),old_S0_vs_F2=summarize(active,arm,"old_S0"))
        if f030:
            s2=summarize(f030,arm,WITNESS)
            s3=summarize(f030,arm,PEELED)
            a=s2["pooled"]["reference"]["psnr_db"]
            b=s3["pooled"]["reference"]["psnr_db"]
            values["matched_f030"]=dict(S2=s2,S3=s3,S3_minus_S2_pooled_db=b-a if a is not None and b is not None else None)
        methods[arm]=values
    return dict(reference="S0 (same-frame peeled)" if legacy else S2,methods=methods)
