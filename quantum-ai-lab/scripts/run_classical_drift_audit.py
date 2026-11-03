"""Replay a frozen bounded subset of existing Step-12 public observations.

Never calls the simulator or rewrites old results. Output is retrospective.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from research.calibration_protocol import COORDINATES, PublicBatch
from research.classical_drift_audit import berger_boos_direction, restricted_score_pvalue
from research.direct_drift_detector import direct_drift_directions, pooled_counts
from research.drift_detector import DriftEnvelope

METHODS = ('direct-bernstein', 'restricted-score-approx', 'berger-boos-envelope')
IDS = ('plan_id','floor_id','allowance_id','cohort_id')


def interval(k,n,failure):
    return [0. if k==0 else float(beta.ppf(failure/2,k,n-k+1)),
            1. if k==n else float(beta.ppf(1-failure/2,k+1,n-k))]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources(directory, saved):
    for relative, expected in saved['source_sha256'].items():
        if fingerprint(ROOT/relative) != expected:
            raise ValueError(f'Prior source mismatch: {relative}')
    for relative, expected in saved['output_sha256'].items():
        if fingerprint(directory/relative) != expected:
            raise ValueError(f'Prior archive mismatch: {relative}')


def row_key(row):
    return tuple(int(row[k]) for k in IDS)


def selected(key):
    p,f,a,c = key
    return p in (9,10) and f==1 and a in (0,1) and c in (0,1,2,3,4,7)


def batches(rows, meta):
    result=[]
    for stage in ('cal','mon'):
        y=np.array([[int(r[f'{stage}_{c}_numerator']) for c in COORDINATES] for r in rows])
        n=np.array([[int(r[f'{stage}_{c}_denominator']) for c in COORDINATES] for r in rows])
        s=np.array([[int(r[f'{stage}_{c}_sifted']) for c in COORDINATES[:4]] for r in rows])
        emitted=int(n[0,4:].sum())
        result.append(PublicBatch(emitted,y,n,s))
    return result


def run(source, output):
    source, output=Path(source),Path(output)
    saved=json.loads((source/'pilot.json').read_text())
    check_sources(source,saved)
    meta={row_key(c):c for c in saved['cohorts']}
    if output.exists():
        raise ValueError('Audit output exists; refusing overwrite')
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    cells=[]
    effects=[]
    statuses=Counter()
    start=time.monotonic()
    with gzip.open(source/'public_counts.csv.gz','rt',newline='') as stream:
        for key, group in itertools.groupby(csv.DictReader(stream),key=row_key):
            if not selected(key):
                continue
            rows=list(group)
            if len(rows)!=512 or [int(r['trial']) for r in rows]!=list(range(512)):
                raise ValueError('Selected cohort membership mismatch')
            c=meta[key]
            env=DriftEnvelope(**c['envelope'])
            cal,mon=batches(rows,c)
            original=direct_drift_directions(cal,mon,env)
            old=original.any(axis=(1,2))
            if not np.array_equal(old,np.array([bool(int(r['direct-bernstein'])) for r in rows])):
                raise ValueError('Original direct decisions do not replay')
            cy,cn=pooled_counts(cal)
            my,mn=pooled_counts(mon)
            bb=original[:,2:].any(axis=(1,2)).copy()
            approx=bb.copy()
            unresolved=np.zeros(512,dtype=int)
            for i in range(512):
                for j,d in enumerate((env.z_error,env.x_error)):
                    yc,nc,ym,nm=(int(x) for x in (cy[i,j],cn[i,j],my[i,j],mn[i,j]))
                    for x,n,z,m in ((yc,nc,ym,nm),(ym,nm,yc,nc)):
                        result=berger_boos_direction(x,n,z,m,d)
                        bb[i] |= result.reject
                        statuses[result.status]+=1
                        unresolved[i] += int('unresolved' in result.status)
                        approx[i] |= restricted_score_pvalue(x,n,z,m,d)<=.05/8
            decisions=dict(zip(METHODS,(old,approx,bb)))
            identity=dict(zip(IDS,key))
            for name,flags in decisions.items():
                count=int(flags.sum())
                cells.append({**identity,'scenario':c['scenario'],'category':c['category'],
                              'drift_per_basis':env.z_error,'method':name,'alarms':count,
                              'trials':512,'rate':count/512,
                              'ci95_marginal':interval(count,512,.05),
                              'ci_joint_nominal':interval(count,512,.025/72)})
            wins=int(np.sum(bb & ~old));losses=int(np.sum(old & ~bb))
            cw,cl=interval(wins,512,.025/(24*2)),interval(losses,512,.025/(24*2))
            effects.append({**identity,'wins_bb':wins,'losses_bb':losses,'trials':512,
                            'difference':(wins-losses)/512,
                            'ci_joint_nominal':[cw[0]-cl[1],cw[1]-cl[0]]})
            for i,row in enumerate(rows):
                records.append({**identity,'trial':i,**{name:int(flags[i]) for name,flags in decisions.items()},
                                'unresolved_directions':int(unresolved[i])})
            print(json.dumps({'cohort':key,'scenario':c['scenario'],
                              'alarms':{name:int(flags.sum()) for name,flags in decisions.items()},
                              'unresolved_directions':int(unresolved.sum()),
                              'elapsed_seconds':round(time.monotonic()-start,2)}),flush=True)
    expected={(p,1,a,c) for p in (9,10) for a in (0,1) for c in (0,1,2,3,4,7)}
    if {tuple(e[k] for k in IDS) for e in effects}!=expected or len(records)!=12288:
        raise ValueError('Audit selection incomplete')
    check_sources(source,saved)
    archive=output/'decisions.csv.gz'
    with gzip.open(archive,'wt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    source_names=['docs/classical-audit-protocol.md','src/research/classical_drift_audit.py',
                  'scripts/run_classical_drift_audit.py','tests/test_classical_drift_audit.py']
    result={'status':'complete-retrospective-replay','pairs':len(records),'cohorts':24,
            'cells':cells,'effects':effects,'direction_statuses':dict(statuses),
            'original_sources_unchanged':True,
            'prior_source_sha256':saved['source_sha256'],'prior_archive_sha256':saved['output_sha256'],
            'audit_source_sha256':{n:fingerprint(ROOT/n) for n in source_names},
            'decisions_sha256':fingerprint(archive),'elapsed_seconds':time.monotonic()-start,
            'limitations':['Retrospective comparator design; not independent fresh confirmation.',
                          'Score is approximate; BB proof is real arithmetic, not certified floating point.',
                          '64 envelope subdivisions per direction; unresolved is nonreject.',
                          'No continuous power certificate, optimality or novelty claim.',
                          'Joint intervals are nominal fixed-analysis binomial intervals for this retrospective selection.']}
    (output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/benchmarks/direct-drift-pilot-v1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.source,args.output)
    print(json.dumps({k:result[k] for k in ('status','pairs','cohorts','elapsed_seconds')}))
