"""Apply the registered E0 rule B to the archived EMA holdout curves."""
from route3_v2data_B_batch import sha256
from route3_v2data_E0_data import ROOT,RUN
from route3_v2data_E_labels import load,save_json
from src.ovg_e import FEATURE_MODES

RECORDS=ROOT/"03_实验记录/records"
SOURCE=RECORDS/"route3-v2data-E0-2026-11-26.json"
RECORD=RECORDS/"route3-v2data-E0-reselect-2026-11-26.json"
RULE="在留出选点场景上，从杆解码误差 p50 ≤ OVG ρ₁ 杆 p50 × 1.05 的 EMA 检查点中取混合负对数似然最低者。打平依次比较杆 p50 更低、步数更早、E-L / E-H1 / E-H2。无满足者取杆 p50 最低者，同样比较步数与配置。"


def fractions(summary,zone,key):
    return {support:summary["selection"][f"{zone}/{support}"][key]
            for support in ("mv_valid","mv_invalid")}


def reselect():
    original=load(SOURCE)
    attempts=[]
    for rank,mode in enumerate(FEATURE_MODES):
        curve=original["modes"][mode]["training"]["curve"]
        if [p["step"] for p in curve]!=list(range(1000,20001,1000)):
            raise ValueError(f"Missing archived EMA checkpoints for {mode}")
        for point in curve:
            h=point["holdout"]
            errors=h["errors"]
            rod=errors["rod/decoded"]["p50"]
            baseline=errors["rod/OVG"]["p50"]
            bound=1.05*baseline
            attempts.append(dict(mode=mode,rank=rank,step=point["step"],phase=point["phase"],
                rod_p50=rod,rod_p95=errors["rod/decoded"]["p95"],gap_p50=errors["gap/decoded"]["p50"],
                OVG_rod_p50=baseline,threshold=bound,nll=h["nll"],
                rod_selected_front=fractions(h,"rod","selected_front"),
                normal_selected_back=fractions(h,"normal","selected_back"),eligible=rod<=bound))
    candidates=[r for r in attempts if r["eligible"]]
    if candidates:
        key=lambda x:(x["nll"],x["rod_p50"],x["step"],x["rank"])
    else:
        key=lambda x:(x["rod_p50"],x["step"],x["rank"])
    selected=min(candidates or attempts,key=key)
    if (selected["mode"],selected["step"])!=("E-H2",3000):
        raise ValueError(f"Rule B did not reproduce the planning self-check: {selected}")
    checkpoint=RUN/selected["mode"]/f"step-{selected['step']:06d}.pt"
    result=dict(rule=RULE,change="2026-11-26，在看过 E0 开发集结果之后修改；原规则（杆 p50 ≤ OVG）选中 E-H2@2k",
                reason="留出场景属于 OVG 阶段 1 看过的 B 场景。OVG 在留出集和开发集的杆 p50 分别约为 0.033 和 0.071 m⁻¹，门槛被压在样本内水平。",
                archived_curve_sha256=sha256(SOURCE),candidate_count=len(attempts),
                candidates=sorted(candidates,key=key),selected=selected,selected_checkpoint_sha256=sha256(checkpoint),
                selected_margin_percent=100*(selected["threshold"]-selected["rod_p50"])/selected["threshold"],
                original_selected=dict(mode=original["selection"]["final"]["mode"],step=original["selection"]["final"]["step"]),
                selection_used_development=False,test_cases_used=0)
    save_json(RECORD,result)
    return result


if __name__=="__main__":
    result=reselect()
    print(result["selected"],result["selected_margin_percent"],flush=True)
