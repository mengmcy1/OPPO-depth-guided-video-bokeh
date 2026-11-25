"""Aggregate targeted FP32 corrections without replacing the historical scores."""
import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from route3_ovg_subpixel import save_json
from route3_v2data_B_batch import ROOT,sha256
from route3_v2data_B_plan import load
from route3_v2data_baseline_dev3_extended_report import group_regions
from route3_v2data_fp16_fit_audit import FIRST,RUN,STAGE3A,STAGE3B,START_FIXED
from route3_v2data_query_retrain import distribution
from route3_v2data_stage3a_eval import METRICS,match_metric,pair_from_grouped,pair_vs_A
from route3_v2data_stage3a_followup import adoption as adoption_3a
from route3_v2data_stage3b_eval import REGIONS,summary as stage3b_summary

B_VERSIONS=((0,"ema"),(5000,"ema"),(5000,"model"))
A_VERSIONS=((0,"ema"),(5000,"ema"),(10000,"ema"),(15000,"ema"),(20000,"ema"),(20000,"model"))
RECORD=ROOT/"03_实验记录/records/route3-v2data-fp16-fit-2026-11-26.json"


def matches(rows):
    return {(r["scene"],r["case"]):r for r in rows}


def value_delta(old,new,pool,per_scene):
    out={}
    for key, before in old.items():
        after=new[key]
        first,last=pool(before),pool(after)
        pairs={scene:after_score-value for scene,value in per_scene(before).items()
               if value is not None and (after_score:=per_scene(after).get(scene)) is not None}
        out[key]=dict(before=first,after=last,delta=last-first if first is not None and last is not None else None,
                      per_scene_delta=pairs,**distribution(list(pairs.values())))
    return out


def regrade_3b():
    reference=load(START_FIXED)
    if not reference:
        raise FileNotFoundError(START_FIXED)
    audit=load(RUN/"stage3a/step-000000/ema/audit.json")
    ref=matches([r for rows in reference.values() for r in rows])
    affected={(a["scene"],a["case"]) for a in audit["cases"] if a["needs_rescore"]}
    original_start=load(STAGE3B/"evaluation/step-000000/ema/report.json")["rows"]
    result={}
    for step,weights in B_VERSIONS:
        root=STAGE3B/"evaluation"/f"step-{step:06d}"/weights
        source=root/"report.json"
        original=load(source)
        fixed=deepcopy(original)
        actual_changed=defaultdict(int)
        for row in fixed["rows"]:
            key=(row["scene"],row["case"])
            if row["status"]!="evaluated" or key not in affected:
                continue
            metric=REGIONS[row["focus_index"]]
            before=row["gt"][metric]
            after=ref[key]["scores"]["stage3"]["metrics"][metric]
            if before["pixels"]!=after["pixels"]:
                raise ValueError(f"3b GT reference changed scoring masks: {key}")
            if before!=after:
                actual_changed[row["cohort"]]+=1
                row["gt"][metric]=after
        fixed["summary"]=stage3b_summary(fixed["rows"],original_start if step else None)
        fixed["GT_reference"]=str(START_FIXED)
        fixed["original_report_sha256"]=sha256(source)
        fixed["correction"]="Regraded archived Query and F2 measurements against corrected 3a GT substitute cases."
        save_json(root/"report-fp16-fit-corrected.json",fixed)
        before=stage3b_summary(original["rows"], original_start if step else None)
        after=fixed["summary"]
        for scope in before:
            for metric in REGIONS:
                historical=original["summary"][scope]["metrics"][metric]
                recomputed=before[scope]["metrics"][metric]
                for key in ("query_minus_f2","query_vs_start"):
                    if historical[key]!=recomputed[key]:
                        raise ValueError("Archived 3b scores do not reproduce")

        by_group={}
        for group,values in before.items():
            differences={}
            for region in REGIONS[:2]:
                a,b=(x[group]["metrics"][region] for x in (before,after))
                gt=value_delta({"gt_minus_f2":a["gt_minus_f2"]},{"gt_minus_f2":b["gt_minus_f2"]},
                               lambda r:r["pooled_db"],lambda r:r["per_scene"])["gt_minus_f2"]
                differences[region]=dict(GT_minus_F2=gt,query_minus_f2=a["query_minus_f2"]["pooled_db"],
                    query_vs_start=a["query_vs_start"]["pooled_db"],ratio_count_before=a["ratio_count"],
                    ratio_count_after=b["ratio_count"],ratio_median_before=a["middle_ratio"],ratio_median_after=b["middle_ratio"],
                    ratio_pooled_before=a["pooled_ratio"],ratio_pooled_after=b["pooled_ratio"])
            if any(values[k]!=after[group][k] for k in ("zero","far_hidden_fraction","scored_cases")):
                raise ValueError(f"3b direct rows changed in {group}")
            for region in REGIONS:
                a,b=(x[group]["metrics"][region] for x in (before,after))
                if a["query_minus_f2"]!=b["query_minus_f2"] or a["query_vs_start"]!=b["query_vs_start"]:
                    raise ValueError("3b direct Query/F2 or adoption comparison changed")
            by_group[group]=differences
        result[f"{step}-{weights}"]=dict(source_case_counts={k:v["rescore"] for k,v in audit["summary"].items()},
                  scored_case_changes=dict(actual_changed),reference_metrics=by_group,
                  original_source_sha256=fixed["original_report_sha256"],
                  direct_scores_bitwise_equal=True,adoption_inputs_unchanged=True)
    save_json(RUN/"stage3b-regraded.json",result)
    return result


def stage3a_report(run,step,weights,start):
    folder=run/"evaluation"/f"step-{step:06d}"/weights
    source=folder/"report.json"
    before=load(source)
    new_dir=RUN/"stage3a"/f"step-{step:06d}"/weights
    oldrows=load(folder/"cases.json")
    newrows=load(new_dir/"cases.json")
    if run==FIRST and oldrows!=load(STAGE3A/"evaluation/step-000000/ema/cases.json"):
        raise ValueError("Cannot reuse the official M0 replay with a different first-run archive")
    audit=load(new_dir/"audit.json")
    payload=deepcopy(before)
    payload["original_report_sha256"]=sha256(source)
    summary={}
    baseline=None
    # External A pairs use the exact same, checksum-registered historical summary.
    if "paired_vs_A" in payload["report"]["development"]:
        from route3_v2data_stage3a_eval import BASELINE
        receipt=load(BASELINE)["artifacts"]
        external=Path(receipt["summary"])
        if sha256(external)!=receipt["summary_sha256"]:
            raise ValueError("The external baseline archive changed")
        baseline=load(external)["summaries"]["f1.4"]
    for cohort,active in newrows.items():
        new=payload["report"][cohort]
        old=before["report"][cohort]
        paired=matches(start[cohort])
        new["metrics"]={k:group_regions(active,"stage3",k) for k in METRICS}
        new["paired_vs_start"]={k:match_metric(active,paired,k) for k in METRICS}
        near=defaultdict(list)
        for row in active:
            if row["scores"] and row["focus_index"]==0 and row["f2"]["pixels"]:
                near[row["scene"]].append((row["f2"],row["scores"]["stage3"]["metrics"]["near_in_focus"]))
        new["gt_minus_f2_near"]=pair_from_grouped(near)
        for tag,obj in new.get("groups",{}).items():
            ids={"003","004","005"} if tag=="thin_structure" else {"008"}
            block=[r for r in active if r["scene"].split("-")[1] in ids]
            obj["regions"]={k:group_regions(block,"stage3",k) for k in ("near_in_focus","far_recoverable")}
        if "paired_vs_A" in new:
            new["paired_vs_A"]={k:pair_vs_A(new["metrics"],baseline,k) for k in ("near_in_focus","far_recoverable")}
        summary[cohort]=dict(affected=audit["summary"].get(cohort,{}),
            PSNR=value_delta(old["metrics"],new["metrics"],lambda r:r["combined"]["psnr"],
                                   lambda r:{k:v["psnr"] for k,v in r["per_scene"].items()}),
            patch_LPIPS=value_delta(old["metrics"],new["metrics"],lambda r:r["combined"]["patch_lpips"],
                                   lambda r:{k:v["patch_lpips"] for k,v in r["per_scene"].items()}),
            zero_before_after={k:[old["metrics"][k]["pass_cases"],new["metrics"][k]["pass_cases"]]
                               for k in ("zero_visible","zero_combined")})
        summary[cohort]["cohort_score_changes"]=sum(matches(oldrows[cohort])[r["scene"],r["case"]]["scores"]!=r["scores"] for r in active)
    name="report-fp16-fit-corrected.json" if run==STAGE3A else "report-v1-step0-fp16-fit-corrected.json"
    save_json(new_dir/name,payload)
    return before,payload,summary


def regrade_3a():
    start=load(START_FIXED)
    out={}
    before,after=[],[]
    for step,weights in A_VERSIONS:
        a,b,row=stage3a_report(STAGE3A,step,weights,start)
        out[f"{step}-{weights}"]=row
        if step and weights=="ema":
            before.append(a)
            after.append(b)
    out["v1-0-ema"]=stage3a_report(FIRST,0,"ema",start)[2]
    choice=dict(before=adoption_3a(before),after=adoption_3a(after))
    choice["changed"]=choice["before"]["selected"]!=choice["after"]["selected"] or choice["before"].get("best_step")!=choice["after"].get("best_step")
    result=dict(scores=out,adoption=choice)
    save_json(RUN/"stage3a-regraded.json",result)
    return result


def scope_inventory():
    import numpy as np
    from collections import Counter

    c7=load(STAGE3B/"c7/report.json")
    depth=load(STAGE3B/"depth-convention/report.json")
    e=load(RUN/"E-audit.json")
    cases3b={}
    for step,weights in B_VERSIONS:
        d=load(STAGE3B/"evaluation"/f"step-{step:06d}"/weights/"report.json")
        cases3b[f"{step}-{weights}"]=dict(Counter(
            r["cohort"] for r in d["rows"] if r["status"]=="evaluated"))
    from route3_v2data_stage3b import DIRECT
    index=load(DIRECT/"prepared.json")
    direct=[r for r in index["accepted"].values() if r["split"]=="train"]
    dtype_evidence={}
    for tier in ("B","C"):
        candidates=[r for r in direct if r["tier"]==tier]
        checked={}
        for entry in candidates[:3]:
            manifest=load(Path(entry["path"]))
            checked[manifest["scene"]]=[]
            for file in sorted({r["input"] for r in manifest["rows"]}):
                with np.load(file) as packed:
                    dtype=str(packed["target_depth"].dtype)
                    if dtype!="float32":
                        raise ValueError(f"Direct {tier} targets are not FP32: {dtype}")
                checked[manifest["scene"]].append(dtype)
        dtype_evidence[tier]=dict(prebuilt_training_scenes=len(candidates),checked_target_depth=checked)
    actual_3b=load(RUN/"stage3b-regraded.json")
    e_expected={"development":8,"training":10}
    if any({k:v["cases"] for k,v in e["by_arm"][a].items()}!=e_expected for a in e["by_arm"]):
        raise ValueError("Did not audit all E render cases")
    result=dict(
        stage3b_direct=dict(cases=cases3b,affected=0,source="hidden_cloud → learned Query + footprint order",
            training=dict(optical="Same predicted Query with footprint order; no GT substitute.",
                direct_targets="Archived 256-5 caches. Query preparation explicitly casts visible-depth labels to FP32 before calling local_placement.",
                target_depth_dtypes=dtype_evidence,
                new_direct_cache_present=(STAGE3B/"direct-cache").exists())),
        stage3b_GT_reference=dict(affected={v:r["scored_case_changes"] for v,r in actual_3b.items()},
                                  source="Corrected Stage 3a step-0 GT substitute measurements."),
        c7=dict(cases=dict(Counter(r["cohort"] for r in c7["rows"])),affected=0,
                source="Both arms use Query candidates and footprint order; C7 swaps GT foreground/rear/split data."),
        depth_convention=dict(frames=len(depth["frame_rows"]),affected=0,
                source="Depth and prior errors without optical placement.",
                cohorts=sorted(depth["cohorts"])),
        E_preparation=dict(cases=e_expected,affected=0,per_arm=e["by_arm"],
                source="Evaluated GT center depth was explicitly cast to FP32 before placement; audited all three V0 arms."),
        previous_FP32_paths={
            "stage2_oracle_H":"oracle_hidden explicitly casts the reference depth to float32.",
            "legacy_gt_oracle":"Three scenes, sampled at three frames and both focal planes; all 18 placements unchanged.",
            "candidate_training_labels":"route3_v2data_query_prepare.placement uses float32 visible depth; no FP16 reciprocal.",
        },
        rejected_nonfinite_after_correction=0)
    save_json(RUN/"scope.json",result)
    return result


def verify_3b_reference():
    reference=matches([r for rows in load(START_FIXED).values() for r in rows])
    receipt={}
    for weights in ("ema","model"):
        path=STAGE3B/"evaluation/step-010000"/weights/"report.json"
        report=load(path)
        if Path(report["GT_reference"]).resolve()!=START_FIXED.resolve():
            raise ValueError(f"Stage 3b 10k {weights} did not use the corrected reference")
        active=[r for r in report["rows"] if r["status"]=="evaluated"]
        for row in active:
            metric=REGIONS[row["focus_index"]]
            expected=reference[row["scene"],row["case"]]["scores"]["stage3"]["metrics"][metric]
            if row["gt"][metric]!=expected:
                raise ValueError(f"Stage 3b reference differs on {row['scene']}/{row['case']}")
        receipt[weights]=dict(report_sha256=sha256(path),matched_cases=len(active))
    return dict(step=10000,case_reference_sha256=sha256(START_FIXED),weights=receipt)


def collect():
    source=RUN
    a=load(source/"stage3a-regraded.json")
    b=load(source/"stage3b-regraded.json")
    scope=scope_inventory()
    evidence=load(source/"stage3a/step-000000/ema/audit.json")
    rings=load(source/"module2/placement-audit.json")
    prior_path=source/"stage3a-priors/report-fp16-fit-corrected.json"
    priors=load(prior_path)
    out={}
    learned_arms={"v1":("d_full","d_nodepth"),"v2":("v2a","v2b")}
    for kind in ("seven","v1","v2"):
        folder=source/"module2"/kind
        data=load(folder/"report-fp16-fit-corrected.json")
        arms=learned_arms.get(kind,tuple(data["by_arm"]))
        expected={(row["scene"],row["case"],arm) for stamp in rings["by_case"].values()
                  for arm in arms for row in (stamp["gt" if arm in ("V0_GT","a") else "pred"],)
                  if row["needs_rescore"]}
        actual={(row["scene"],row["case"],arm) for path in (folder/"frames").glob("*/*.json")
                for row in load(path) for arm in row["methods"]}
        if actual!=expected:
            raise ValueError(f"{kind} replay coverage differs: missing {len(expected-actual)}, unexpected {len(actual-expected)}")
        out[kind]={k:data[k] for k in ("by_arm","changes")}
    gate=load(source/"module2/a-check/report-fp16-fit-corrected.json")
    out["a-check"]={k:gate[k] for k in ("changes","affected")}
    gaps=load(source/"module2/v1/gaps/comparison.json")
    optics=load(source/"verified-images.json")
    if not a or not b or not rings or not priors or not gaps or len(optics)!=3:
        raise ValueError("Incomplete targeted replay")
    endpoint=source/"stage3a/step-020000/ema/report-fp16-fit-corrected.json"
    endpoint_report=load(endpoint)
    endpoint_report["prior_comparison"]=priors
    endpoint_report["corrected_prior_comparison_sha256"]=sha256(prior_path)
    save_json(endpoint,endpoint_report)
    record=dict(fix_commit="6cff9f9", original_fit_commit="1f61930", audited_scope=scope,
        mechanism="Native stored FP16 GT zeros become infinity in the reciprocal. Zero times infinity contaminates uniform-filter moments.",
        stage3a=a,stage3b=b,stage3b_reference_verification=verify_3b_reference(),
        stage3a_prior_only={"validity":priors["validity"],"changes":priors["changes"]},
        module2=out,module2_geometry_audit=rings["summary"],
        module2_v1_gaps=gaps,regression_clouds=evidence["verification"],unchanged_render_arrays=optics,
        overflow_cases=[r for r in evidence["cases"] if r["old_candidate_nonfinite"]],
        count_convention="A case is rescored if any accepted index or fitted FP32 placement depth changed. A nonfinite fit away from candidates is tracked separately.",
        artifacts={k:str((RUN/p).relative_to(ROOT)) for k,p in (
            ("affected_case_details","stage3a/step-000000/ema/audit.json"),
            ("stage3a","stage3a"),("stage3b","stage3b-regraded.json"),
            ("module2","module2"),("priors","stage3a-priors/report-fp16-fit-corrected.json"),
            ("unchanged_validation","verified-images.json"))})
    save_json(RECORD,record)
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("3a","3b","scope","collect"))
    arg=p.parse_args()
    {"3a":regrade_3a,"3b":regrade_3b,"scope":scope_inventory,"collect":collect}[arg.phase]()
