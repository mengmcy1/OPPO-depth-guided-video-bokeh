"""CPU checks of allowed correction directions and legacy objective compatibility."""
import json
from pathlib import Path
import torch
import train_geometry_student_pilot as t

ROOT=Path(__file__).resolve().parents[1]
old=json.loads((ROOT/'configs/ours-student-pilot-v1-20261023.json').read_text())
new=json.loads((ROOT/'configs/ours-student-interval-v2-20261023.json').read_text())
allowed={'name','output','record_prefix','keep_loss','keep_mode','prior_config','single_factor','acceptance','scope'}
assert all(old[k]==new[k] for k in old if k not in allowed)
x=torch.zeros(1,4,1,5);x[:,3:4]=.5;v=torch.ones(1,1,1,5,dtype=torch.bool)
for target,points in [(.52,[.49,.5,.51,.52,.53]),(.48,[.47,.48,.49,.5,.51])]:
    y=torch.full((1,1,1,5),target);out=torch.tensor(points).reshape(1,1,1,5).requires_grad_()
    t.C=new;_,_,keep=t.objective(out,x,y,v,v,.028637895)
    assert torch.allclose(keep,torch.tensor(.004),atol=1e-7)
    keep.backward();assert torch.allclose(out.grad.flatten(),torch.tensor([-.2,0,0,0,.2]))
    wrong=torch.full_like(y,.49 if target>.5 else .51)
    assert t.objective(wrong,x,y,v,v,.028637895)[2]>.009
    t.C=old;assert t.objective(wrong,x,y,v,v,.028637895)[2]==0
# Both identity and reference, and the degenerate equal-input-target interval.
t.C=new
for target in [.48,.50,.52]:
    y=torch.full((1,1,1,5),target)
    for out in [x[:,3:4],y]:assert t.objective(out,x,y,v,v,.028637895)[2]==0
# Preserve the original v1 path, including empty keep masks.
torch.manual_seed(1);x=torch.rand(2,4,8,8);y=torch.rand(2,1,8,8);out=torch.rand_like(y);v=torch.rand_like(y)>.2;k=torch.rand_like(y)>.5
t.C=old;loss,_,_=t.objective(out,x,y,v,k,.028637895)
manual=((out-y).abs()*v).sum()/v.sum()+.1*(torch.relu((out-x[:,3:4]).abs()-.028637895)*k).sum()/k.sum()
assert torch.equal(loss,manual)
for config in [old,new]:
    t.C=config;assert t.objective(out,x,y,v,torch.zeros_like(k),.028637895)[2]==0
result=dict(status='INTERVAL_LOSS_AND_SINGLE_FACTOR_CHECKS_PASSED',changed_training_factor='keep loss zero-penalty interval',same_keep_weight=True,all_other_training_fields_equal=True,
    inside_interval_zero=True,wrong_direction_and_overshoot_gradients=True,identity_and_reference_zero_penalty=True,legacy_objective_bitwise_equal=True,empty_keep_supported=True)
p=ROOT/(new['record_prefix']+'-loss-check.json')
with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(result)
