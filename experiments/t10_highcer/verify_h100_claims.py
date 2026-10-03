"""Assert every numeric claim in docs/GPT_REVIEW_T10_H100_EN.md against the H100 results.

Written 2026-09-26 because I published two substitution ranges from the wrong decoder first.
Any edit to the brief's numbers must keep this passing.

    python3 verify_h100_claims.py [results_h100_dir]
"""
import collections
import glob
import json
import os
import statistics as st
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b0_run import norm_syl, to_jamo, cer

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/Desktop/h100_results/results_h100')
_MF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 't10_manifest.json')
refs = {it['id']: it['ref_text'] for it in json.load(open(_MF, encoding='utf-8'))['items']}

def sdi(r, h):
    prev = [(j, 0, 0, j) for j in range(len(h) + 1)]
    for i in range(1, len(r) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(h) + 1):
            c, s, d, ins = prev[j - 1]
            cand = (c, s, d, ins) if r[i - 1] == h[j - 1] else (c + 1, s + 1, d, ins)
            c, s, d, ins = prev[j]; dl = (c + 1, s, d + 1, ins)
            c, s, d, ins = cur[j - 1]; it_ = (c + 1, s, d, ins + 1)
            cur.append(min(cand, dl, it_))
        prev = cur
    return prev[len(h)]

R = collections.defaultdict(dict)
for p in glob.glob(D + '/longform/*.json'):
    r = json.load(open(p, encoding='utf-8'))
    spk = 'KEJ' if '-KEJ-' in r['id'] else 'DTH'
    assert abs(round(cer(refs[r['id']], r['text'], jamo=True)[0], 4) - r['cer_jamo']) < 1e-4, p
    R[(spk, r['decoder'], r['vad'])][(r['system'], r['eval_seed'])] = r
assert sum(len(v) for v in R.values()) == 112

def seeds(k, spk):
    return ['%s_nall_s%d' % (spk, i) for i in range(5)]

def mean_over_es(cell, sysname):
    return st.mean([cell[(sysname, es)]['cer_jamo'] for es in (0, 1)])

# 4.1 absolute table + improve counts
EXP41 = {('DTH','default','off'):(58.05,35.53,31.64,5), ('DTH','nr5','off'):(48.09,34.75,31.74,5),
         ('DTH','default','silero'):(22.04,24.12,23.14,0), ('DTH','nr5','silero'):(22.08,24.73,23.18,0),
         ('KEJ','default','off'):(83.08,45.56,42.84,5), ('KEJ','nr5','off'):(74.75,44.78,41.30,5),
         ('KEJ','default','silero'):(71.26,65.51,63.83,5), ('KEJ','nr5','silero'):(74.13,65.48,63.44,5)}
for k,(b0,med,best,nimp) in EXP41.items():
    cell = R[k]; spk = k[0]
    g0 = mean_over_es(cell,'b0')*100
    vals = [mean_over_es(cell,s)*100 for s in seeds(k,spk)]
    assert abs(g0-b0)<0.005, (k,'b0',g0,b0)
    assert abs(st.median(vals)-med)<0.005, (k,'median',st.median(vals),med)
    assert abs(min(vals)-best)<0.005, (k,'best',min(vals),best)
    assert sum(v<g0 for v in vals)==nimp, (k,'improving',sum(v<g0 for v in vals),nimp)

# 4.2 median diffs and eval-seed spreads
EXP42 = {('KEJ','default','off'):(-37.52,6.60),('KEJ','nr5','off'):(-29.97,12.13),
         ('KEJ','default','silero'):(-5.75,0.53),('KEJ','nr5','silero'):(-8.65,0.96),
         ('DTH','default','off'):(-22.52,5.14),('DTH','nr5','off'):(-13.34,7.52),
         ('DTH','default','silero'):(2.08,0.00),('DTH','nr5','silero'):(2.65,0.00)}
for k,(med,spread) in EXP42.items():
    cell=R[k]; spk=k[0]
    diffs=[(mean_over_es(cell,s)-mean_over_es(cell,'b0'))*100 for s in seeds(k,spk)]
    assert abs(st.median(diffs)-med)<0.006,(k,st.median(diffs),med)
    # summarize_h100.py: es-spread = max-min of the ADAPTER's CER across eval seeds, within a training seed
    per=[abs(cell[(s,0)]['cer_jamo']-cell[(s,1)]['cer_jamo'])*100 for s in seeds(k,spk)]
    assert abs(max(per)-spread)<0.02,(k,'spread',max(per),spread)

# 4.3 determinism of the DTH masked arm, instability of the KEJ VAD-off arm
for dec in ('default','nr5'):
    c=R[('DTH',dec,'silero')]
    for s in ['b0']+seeds(('DTH',dec,'silero'),'DTH'):
        assert c[(s,0)]['text_sha256']==c[(s,1)]['text_sha256'], ('DTH masked not deterministic',s)
c=R[('KEJ','nr5','off')]
assert (round(c[('b0',0)]['cer_jamo'],4),round(c[('b0',1)]['cer_jamo'],4))==(0.6296,0.8654)
assert (round(c[('KEJ_colab_run1',0)]['cer_jamo'],4),round(c[('KEJ_colab_run1',1)]['cer_jamo'],4))==(0.6860,0.4125)
# 20/20 KEJ adapter wins
wins=sum(R[('KEJ',d,v)][(s,es)]['cer_jamo'] < R[('KEJ',d,v)][('b0',es)]['cer_jamo']
         for d in ('default','nr5') for v in ('off',) for es in (0,1) for s in seeds(0,'KEJ'))
assert wins==20, wins
# archived adapters: different weights, same es0 text in 3 of 4 conditions
pairs=[(R[('KEJ',d,v)][('KEJ_colab_run1',0)],R[('KEJ',d,v)][('KEJ_colab_run2',0)])
       for d in ('default','nr5') for v in ('off','silero')]
assert all(a['adapter_sha256']!=b['adapter_sha256'] for a,b in pairs)
assert sum(a['text_sha256']==b['text_sha256'] for a,b in pairs)==3

# 4.4 decomposition claims (eval seed 0)
def dec0(spk,sysname,d,v):
    r=R[(spk,d,v)][(sysname,0)]
    return sdi(to_jamo(norm_syl(refs[r['id']])), to_jamo(norm_syl(r['text'])))[1:], len(to_jamo(norm_syl(r['text'])))
assert dec0('KEJ','b0','default','off')==((801,175,483),2187)
assert dec0('KEJ','b0','nr5','off')==((776,216,191),1854)
assert dec0('KEJ','b0','default','silero')==((166,1196,32),715)
assert dec0('DTH','b0','default','off')==((632,144,958),3459)
assert dec0('DTH','b0','nr5','off')==((375,279,679),3045)
assert dec0('DTH','b0','default','silero')==((284,68,231),2808)
def rng(spk,d,v):
    xs=[dec0(spk,s,d,v) for s in seeds(0,spk)]
    S=[x[0][0] for x in xs]; Dl=[x[0][1] for x in xs]; I=[x[0][2] for x in xs]; H=[x[1] for x in xs]
    return (min(S),max(S)),(min(Dl),max(Dl)),(min(I),max(I)),(min(H),max(H))
assert rng('KEJ','nr5','off')==((322,434),(153,339),(94,577),(1634,2293)), rng('KEJ','nr5','off')
assert rng('KEJ','default','silero')==((148,175),(1004,1059),(14,24),(838,896)), rng('KEJ','default','silero')
assert rng('DTH','default','silero')==((284,313),(60,81),(222,298),(2790,2883)), rng('DTH','default','silero')
# §6: across ALL masked KEJ cells
allm=[dec0('KEJ',s,d,'silero') for d in ('default','nr5') for s in ['b0']+seeds(0,'KEJ')]
assert (min(x[1] for x in allm),max(x[1] for x in allm))==(538,907)
assert (min(x[0][1] for x in allm),max(x[0][1] for x in allm))==(991,1352)
assert (min(x[0][2] for x in allm),max(x[0][2] for x in allm))==(11,32)

# §1/§3 provenance
r=R[('KEJ','default','silero')][('b0',0)]
assert round(r['source_sec'],1)==970.9 and round(r['vad_retained_sec'],1)==127.7 and r['n_jamo']==1879
r=R[('DTH','default','silero')][('b0',0)]
assert round(r['source_sec'],1)==2401.7 and round(r['vad_retained_sec'],1)==352.0 and r['n_jamo']==2645
# 110.05% is the mean over eval seeds, as the frozen rule reports it
assert round(st.mean([R[('KEJ','default','off')][('b0',es)]['cer_syl'] for es in (0,1)]),4)==1.1005
led=[json.loads(l) for l in open(D+'/ledger.jsonl',encoding='utf-8')]
assert len(led)==128
c=collections.Counter(x['status'] for x in led)
assert c['done']==122 and c['skipped']==5 and c['platform_check']==1, c
assert not [x for x in led if x['status'] in ('failed','fallback')]
for spk,eps in (('KEJ',[1,2,1,6,5]),('DTH',[7,6,4,7,6])):
    got=[json.load(open('%s/%s_nall_s%d/summary.json'%(D,spk,i),encoding='utf-8'))['selected_epoch'] for i in range(5)]
    assert got==eps,(spk,got,eps)
# --- corrections established by the 2026-09-26 GPT review; these must not regress ---
def syl_sdi(spk, sysname, d, v, es):
    r = R[(spk, d, v)][(sysname, es)]
    ref, hyp = norm_syl(refs[r['id']]), norm_syl(r['text'])
    return sdi(ref, hyp)[1:], len(ref), len(hyp)

# CER > 100% is NOT "insertions alone exceed the reference": both insertion counts are below 766
assert syl_sdi('KEJ', 'b0', 'default', 'off', 0) == ((555, 25, 196), 766, 937)
assert syl_sdi('KEJ', 'b0', 'default', 'off', 1) == ((556, 9, 345), 766, 1102)

# jamo are NOT 3 per syllable: the normalised syllable counts are 766 / 1096
assert len(norm_syl(refs[R[('KEJ','default','off')][('b0',0)]['id']])) == 766
assert len(norm_syl(refs[R[('DTH','default','off')][('b0',0)]['id']])) == 1096
# reference-syllables per retained second (NOT a speaking rate, and it certifies nothing)
for spk, exp in (('KEJ', 6.00), ('DTH', 3.11)):
    r = R[(spk, 'default', 'silero')][('b0', 0)]
    assert abs(len(norm_syl(refs[r['id']])) / r['vad_retained_sec'] - exp) < 0.005, spk

# DTH base masking gain is 58.055 -> 22.04 under the frozen rule (65.56 is eval seed 0 only)
c = R[('DTH', 'default', 'off')]
assert round(c[('b0', 0)]['cer_jamo'], 5) == 0.65560
assert round(st.mean([c[('b0', es)]['cer_jamo'] for es in (0, 1)]), 5) == 0.58055

# "no error type improves" is false: adapter minima beat the base on D and I
S, Dl, I, H = rng('DTH', 'default', 'silero')
assert Dl[0] < 68 and I[0] < 231, (Dl, I)

# the paired (b1-b0) spread is much larger than the adapter-CER spread the frozen rule prints
EXP_PAIRED = {('KEJ','default'):17.45, ('KEJ','nr5'):35.71, ('DTH','default'):20.15, ('DTH','nr5'):9.80}
for (spk, dec), exp in EXP_PAIRED.items():
    cell = R[(spk, dec, 'off')]
    pr = max(abs((cell[(s,0)]['cer_jamo']-cell[('b0',0)]['cer_jamo'])
                 - (cell[(s,1)]['cer_jamo']-cell[('b0',1)]['cer_jamo'])) for s in seeds(0, spk))*100
    assert abs(pr-exp) < 0.01, (spk, dec, pr, exp)

# the median hides a 1.12 pp smallest individual win
cell = R[('KEJ','nr5','off')]
wins = [(cell[('b0',es)]['cer_jamo']-cell[(s,es)]['cer_jamo'])*100 for s in seeds(0,'KEJ') for es in (0,1)]
assert min(wins) > 0 and abs(min(wins)-1.12) < 0.01, min(wins)

# 32 KEJ masked outputs, not 8; the range checks above use 12 of them
assert len(glob.glob(D + '/longform/*KEJ*vad-silero*')) == 32
# no fallback trace exists in any segment, so determinism cannot be verified from the artifacts
segkeys = {tuple(sorted(sg)) for p in glob.glob(D+'/longform/*.json')
           for sg in json.load(open(p, encoding='utf-8'))['segments']}
assert segkeys == {('end', 'start', 'text')}, segkeys
print("all brief claims verified, including the 2026-09-26 GPT-review corrections")
