"""Continuous total supervision plus within-task candidate gap regression."""
import torch
import torch.nn.functional as F

def task_loss(logits, candidates, weights, mode):
    if mode != 'scalar':
        raise ValueError('Candidate capability-score losses are retired. Use capability_router_20261018/loss.py '
                         'with task capability targets and the new prepared plan.')
    probs=logits.float().sigmoid()
    score=probs[:,0]
    valid=torch.tensor([c['supervision_valid'] for c in candidates],device=logits.device)
    y=torch.tensor([c['quality'] if c['supervision_valid'] else 0. for c in candidates],device=logits.device,dtype=torch.float32)
    zero=logits.sum()*0
    point=F.binary_cross_entropy(score[valid].clamp(1e-6,1-1e-6),y[valid]) if valid.any() else zero
    p,t=score[valid],y[valid]
    if len(p)>1:
        i,j=torch.triu_indices(len(p),len(p),offset=1,device=logits.device)
        gap=((p[i]-p[j])-(t[i]-t[j])).square().mean()
    else:gap=zero
    dimension=zero
    total=point+4.*gap+dimension
    return total,{'point':float(point.detach()),'gap':float(gap.detach()),'dimension':float(dimension.detach())}
