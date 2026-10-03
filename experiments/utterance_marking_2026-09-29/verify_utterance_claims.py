"""Assert every number in docs/GPT_REVIEW_T10_UTTERANCE_EN.md against results_1003."""
import hashlib
import json, os, statistics as st, sys
sys.path.insert(0, os.path.expanduser('~/Desktop/sw_challenge/experiments/b0'))
from b0_run import cer, norm_syl, to_jamo
D = os.path.expanduser('~/Desktop/h100_results_1003')
R = {s: json.load(open('%s/results_utterance/%s.json' % (D, s), encoding='utf-8')) for s in ('KEJ','DTH')}
man = json.load(open(os.path.expanduser('~/Desktop/sw_challenge/experiments/utterance_marking_2026-09-29/utterance_manifest.json'), encoding='utf-8'))
meta = {i['id']+'.wav': i for i in man['items']}

def sdi(r,h):
    prev=[(j,0,0,j) for j in range(len(h)+1)]
    for i in range(1,len(r)+1):
        cur=[(i,0,i,0)]
        for j in range(1,len(h)+1):
            c,s,d,ins=prev[j-1]; cand=(c,s,d,ins) if r[i-1]==h[j-1] else (c+1,s+1,d,ins)
            c,s,d,ins=prev[j]; dl=(c+1,s,d+1,ins)
            c,s,d,ins=cur[j-1]; it=(c+1,s,d,ins+1)
            cur.append(min(cand,dl,it))
        prev=cur
    return prev[len(h)][1:]

def pooled(spk,sysname,dec,drop=None):
    pi=R[spk]['results'][sysname][dec]['per_item']
    num=sum(p['cer_jamo']*p['n_jamo'] for k,p in enumerate(pi) if k!=drop)
    den=sum(p['n_jamo'] for k,p in enumerate(pi) if k!=drop)
    return num/den
seeds=lambda s:['%s_nall_s%d'%(s,i) for i in range(5)]

# §8: all 240 per-item CERs reproduce
n=bad=0
for spk in R:
    for sy,decs in R[spk]['results'].items():
        for dec,v in decs.items():
            for p in v['per_item']:
                c,_=cer(p['ref'],p['hyp'],jamo=True); n+=1
                if abs(round(c,4)-p['cer_jamo'])>1e-4: bad+=1
assert n==240 and bad==0, (n,bad)

# §3: eval set shape
for spk,secs,jam in (('KEJ',157.4,581),('DTH',69.1,485)):
    pi=R[spk]['results']['b0']['default']['per_item']
    assert len(pi)==10
    assert sum(p['n_jamo'] for p in pi)==jam
    assert abs(sum(meta[p['wav']]['crop_sec'] for p in pi)-secs)<0.05
assert len(man['items'])==20 and not man['excluded']
assert not [i for i in man['items'] if i.get('confidence')=='unsure']

# §4.1 table
EXP={('DTH','default'):(30.93,-1.65,4),('DTH','nr5'):(30.93,-1.65,4),
     ('KEJ','default'):(51.12,-7.92,4),('KEJ','nr5'):(50.95,-7.74,4)}
for (spk,dec),(b0,med,imp) in EXP.items():
    g=pooled(spk,'b0',dec)*100
    d=[(pooled(spk,s,dec)-pooled(spk,'b0',dec))*100 for s in seeds(spk)]
    assert abs(g-b0)<0.005,(spk,dec,g,b0)
    assert abs(st.median(d)-med)<0.006,(spk,dec,st.median(d),med)
    assert sum(x<0 for x in d)==imp,(spk,dec,d)
# absolutes quoted in 4.1
# 2026-10-03 review: this used to sort BOTH sides, so it could not see that the brief listed KEJ's
# s0 and s1 the wrong way round. Assert by system id, in seed order.
for spk,exp in (('KEJ',[43.89,40.79,42.17,43.20,51.46]),('DTH',[29.28,29.28,28.25,30.31,34.23])):
    got=[round(pooled(spk,'%s_nall_s%d'%(spk,i),'nr5')*100,2) for i in range(5)]
    assert got==exp,(spk,got,exp)

# §4.3 seed 4 loop
k=json.load(open('%s/results_utterance/KEJ.json'%D,encoding='utf-8'))['results']['KEJ_nall_s4']
assert round(k['default']['pooled_cer_jamo']*100,2)==223.24
assert round(k['nr5']['pooled_cer_jamo']*100,2)==51.46
w04d=k['default']['per_item'][4]; w04n=k['nr5']['per_item'][4]
assert len(w04d['hyp'])==880 and len(w04n['hyp'])==71, (len(w04d['hyp']),len(w04n['hyp']))
assert round(w04d['cer_jamo'],2)==14.70
same=sum(a['hyp']==b['hyp'] for a,b in zip(k['default']['per_item'],k['nr5']['per_item']))
assert same==9, same
for i in range(4):
    s=R['KEJ']['results']['KEJ_nall_s%d'%i]
    assert all(a['hyp']==b['hyp'] for a,b in zip(s['default']['per_item'],s['nr5']['per_item'])), i

# §4.4 decomposition
def dec_of(spk,sysname):
    S=Dl=I=N=H=0
    for p in R[spk]['results'][sysname]['nr5']['per_item']:
        rr=to_jamo(norm_syl(p['ref'])); hh=to_jamo(norm_syl(p['hyp']))
        s,d,i=sdi(rr,hh); S+=s;Dl+=d;I+=i;N+=len(rr);H+=len(hh)
    return S,Dl,I,round(H/N,2)
assert dec_of('KEJ','b0')==(123,143,30,0.81), dec_of('KEJ','b0')
assert dec_of('DTH','b0')==(64,16,70,1.11), dec_of('DTH','b0')
ks=[dec_of('KEJ','KEJ_nall_s%d'%i) for i in range(4)]
assert (min(x[0] for x in ks),max(x[0] for x in ks))==(98,117)
assert (min(x[1] for x in ks),max(x[1] for x in ks))==(89,121)
ds=[dec_of('DTH','DTH_nall_s%d'%i) for i in range(4)]
assert (min(x[0] for x in ds),max(x[0] for x in ds))==(74,83)
assert (min(x[2] for x in ds),max(x[2] for x in ds))==(40,59)

# §5 leave-one-out: only W05 flips DTH
flips=[]
for k_ in range(10):
    b=pooled('DTH','b0','nr5',k_); d=[(pooled('DTH',s,'nr5',k_)-b)*100 for s in seeds('DTH')]
    if st.median(d)>0: flips.append((k_,round(b*100,2),round(st.median(d),2),sum(x<0 for x in d)))
assert flips==[(5,27.33,2.96,0)], flips
kej=[]
for k_ in range(10):
    b=pooled('KEJ','b0','nr5',k_); d=[(pooled('KEJ',s,'nr5',k_)-b)*100 for s in seeds('KEJ')]
    kej.append((st.median(d),sum(x<0 for x in d)))
assert all(m<0 for m,_ in kej) and all(i>=4 for _,i in kej)
assert abs(min(m for m,_ in kej)-(-9.53))<0.01 and abs(max(m for m,_ in kej)-(-5.60))<0.01

# §5 W05 texts and lengths
p0=R['DTH']['results']['b0']['nr5']['per_item'][5]
ps=R['DTH']['results']['DTH_nall_s0']['nr5']['per_item'][5]
# AI-Hub label text must not live in the repo (PROTOCOL §6), so pin the reference by hash.
assert hashlib.sha256(p0['ref'].encode()).hexdigest()[:16]=='7aad746d158ef1fd', p0['ref']
assert len(p0['hyp'])==38 and round(p0['cer_jamo'],3)==0.652
assert len(ps['hyp'])==23 and round(ps['cer_jamo'],3)==0.152

# §7 job records
for spk,enr,spl,inc,new in (('KEJ',27,(18,4,5),0.503,0.3905),('DTH',109,(71,16,22),0.1936,0.1813)):
    j=json.load(open('%s/jobs/wk-%s.json'%(D,spk),encoding='utf-8'))
    assert j['state']=='promoted' and j['n_enroll']==enr
    assert (j['n_train'],j['n_dev'],j['n_gate'])==spl
    assert round(j['incumbent_cer_jamo'],4)==inc and round(j['new_cer_jamo'],4)==new
    assert j['previous_active'] is None

# provenance
for spk in R:
    assert R[spk]['base_revision'].startswith('973afd24965f')
    v=R[spk]['versions']
    assert (v['transformers'],v['peft'],v['torch'])==('5.17.0','0.21.0','2.5.1+cu124')
    assert R[spk]['device']=='cuda'
print("all brief claims verified (240 CERs reproduced, 0 mismatches)")
