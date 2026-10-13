"""Bootstrap 95% confidence intervals and confidence-vs-error analysis (Round-2 reviewer points).

Usage:
    python experiments/run_confidence_intervals.py experiments/outputs/predictions.csv

Reads the per-image test predictions written by run_eval.py and prints:
  * accuracy and macro-F1 with 2,000-sample bootstrap 95% CIs (+ Wilson CI for accuracy)
  * per-class F1 with bootstrap 95% CIs (minority classes have the widest intervals)
  * AUROC of max-softmax confidence for separating correct from incorrect predictions
  * accuracy within confidence bands and the referral curve (errors captured at each referral rate)
Deterministic (seed 42). Pure Python, no third-party dependencies.
"""
import csv, sys, math, random
random.seed(42)
rows=list(csv.DictReader(open(sys.argv[1])))
classes=[k[5:] for k in rows[0].keys() if k.startswith('prob_')]
y=[r['true_label'] for r in rows]; p=[r['pred_label'] for r in rows]
conf=[max(float(r['prob_'+c]) for c in classes) for r in rows]
corr=[a==b for a,b in zip(y,p)]
n=len(rows)
def macro_f1(idx):
    f1s=[]
    for c in classes:
        tp=sum(1 for i in idx if y[i]==c and p[i]==c); fp=sum(1 for i in idx if y[i]!=c and p[i]==c); fn=sum(1 for i in idx if y[i]==c and p[i]!=c)
        pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
        f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return sum(f1s)/len(f1s), f1s
def acc(idx): return sum(corr[i] for i in idx)/len(idx)
B=2000
accs=[];mf1s=[];pcf=[[] for _ in classes]
for b in range(B):
    idx=[random.randrange(n) for _ in range(n)]
    accs.append(acc(idx)); m,f=macro_f1(idx); mf1s.append(m)
    for j,v in enumerate(f): pcf[j].append(v)
def ci(v): v=sorted(v); return v[int(0.025*B)], v[int(0.975*B)-1]
print(f"n={n} acc={acc(range(n))*100:.2f} CI95 {ci(accs)[0]*100:.2f}-{ci(accs)[1]*100:.2f}")
m,f=macro_f1(range(n)); print(f"macroF1={m*100:.2f} CI95 {ci(mf1s)[0]*100:.2f}-{ci(mf1s)[1]*100:.2f}")
for j,c in enumerate(classes):
    sup=sum(1 for t in y if t==c); print(f"  {c:26s} n={sup:4d} F1={f[j]*100:5.1f} CI95 {ci(pcf[j])[0]*100:5.1f}-{ci(pcf[j])[1]*100:5.1f}")
# Wilson CI for accuracy
k=sum(corr); z=1.96; ph=k/n; d=1+z*z/n; ctr=(ph+z*z/(2*n))/d; hw=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/d
print(f"Wilson 95% CI acc: {(ctr-hw)*100:.2f}-{(ctr+hw)*100:.2f}")
# confidence vs error: AUROC of confidence for detecting correct predictions
pos=[conf[i] for i in range(n) if corr[i]]; neg=[conf[i] for i in range(n) if not corr[i]]
wins=sum(1 for a in pos for b in neg if a>b)+0.5*sum(1 for a in pos for b in neg if a==b)
print(f"errors={len(neg)}  AUROC(conf -> correct)={wins/(len(pos)*len(neg)):.3f}")
print(f"mean conf correct={sum(pos)/len(pos):.3f}  mean conf wrong={sum(neg)/len(neg):.3f}")
for lo,hi in [(0,0.5),(0.5,0.7),(0.7,0.9),(0.9,0.99),(0.99,1.01)]:
    idx=[i for i in range(n) if lo<=conf[i]<hi]
    if idx: print(f"  conf [{lo},{hi}): n={len(idx):4d} ({len(idx)/n*100:4.1f}%) acc={acc(idx)*100:5.1f}")
# referral by lowest confidence
order=sorted(range(n), key=lambda i: conf[i])
for r in [0.05,0.1,0.2,0.3]:
    keep=order[int(r*n):]; ref=order[:int(r*n)]
    print(f"  refer {int(r*100)}%: kept acc={acc(keep)*100:.2f}  errors captured={sum(1 for i in ref if not corr[i])}/{len(neg)}")
# confusion pairs
from collections import Counter
cp=Counter((y[i],p[i]) for i in range(n) if not corr[i]); print("top confusions:",cp.most_common(6))
