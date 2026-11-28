import math
import numpy as np


def log_softmax(z):
    m=np.max(z,axis=1,keepdims=True)
    log_s=z-m -np.log(np.sum(np.exp(z-m),axis=1,keepdims=True))
    return log_s

def kl_per_position(logits_ref, logits_q):
    p_ref=log_softmax(logits_ref)
    p_q=log_softmax(logits_q)
    p=np.exp(p_ref)
    kl_per_position=np.sum(p*(p_ref-p_q),axis=1)
    return kl_per_position
    


def top1_agreement(logits_ref, logits_q):
    scores=np.argmax(logits_ref,axis=1)
    scores2=np.argmax(logits_q,axis=1)
    return np.sum(scores==scores2)/scores.shape[0]



def quality_report(logits_ref, logits_q):
    # TODO: {'mean_kl', 'max_kl', 'p99_kl', 'top1_agreement'} rounded to 6 decimals
    kl=kl_per_position(logits_ref,logits_q)
    mean_kl=np.mean(kl_per_position(logits_ref,logits_q))
    max_kl=np.max(kl_per_position(logits_ref,logits_q))
    sorted_kl=np.sort(kl)
    n=len(kl)
    pos=math.ceil(0.99*n)-1
    p_99=sorted_kl[pos]
    agreement=top1_agreement(logits_ref,logits_q)
    return {"mean_kl": mean_kl, "max_kl": max_kl,"p99_kl": p_99, 'top1_agreement': agreement}


def passes_gate(report, max_mean_kl, min_agreement):
    return (report['mean_kl']<=max_mean_kl and report['top1_agreement']>=min_agreement)
