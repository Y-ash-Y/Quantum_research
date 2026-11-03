"""Independent bookkeeping/CP inversion check of the bounded replay artifact."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
IDS=('plan_id','floor_id','allowance_id','cohort_id')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ci(k,n,failure):
    q=binomtest(k,n).proportion_ci(1-failure)
    return [q.low,q.high]


def close(x,y):
    if not np.allclose(x,y,atol=1e-10,rtol=0):
        raise ValueError(f'Interval mismatch {x} {y}')


def verify(directory, source=None):
    directory=Path(directory)
    p=json.loads((directory/'audit.json').read_text())
    source=(ROOT/'results/benchmarks/direct-drift-pilot-v1'
            if source is None else Path(source))
    for name,h in p['prior_source_sha256'].items():
        if digest(ROOT/name)!=h:
            raise ValueError('Original source modified')
    for name,h in p['prior_archive_sha256'].items():
        if digest(source/name)!=h:
            raise ValueError('Original archive modified')
    for name,h in p['audit_source_sha256'].items():
        if digest(ROOT/name)!=h:
            raise ValueError('Comparator source modified since replay')
    if digest(directory/'decisions.csv.gz')!=p['decisions_sha256']:
        raise ValueError('Replay archive modified')
    decisions={}
    groups=defaultdict(list)
    with gzip.open(directory/'decisions.csv.gz','rt',newline='') as stream:
        for row in csv.DictReader(stream):
            key=tuple(int(row[k]) for k in IDS)+ (int(row['trial']),)
            if key in decisions:
                raise ValueError('Duplicate replay row')
            if any(int(row[m]) not in (0,1) for m in
                   ('direct-bernstein','restricted-score-approx','berger-boos-envelope')):
                raise ValueError('Invalid boolean replay')
            decisions[key]=row
            groups[key[:4]].append(row)
    expected={(p,1,a,c) for p in (9,10) for a in (0,1) for c in (0,1,2,3,4,7)}
    if set(groups)!=expected or len(decisions)!=12288:
        raise ValueError('Selection mismatch')
    seen=set()
    with gzip.open(source/'public_counts.csv.gz','rt',newline='') as stream:
        for row in csv.DictReader(stream):
            key=tuple(int(row[k]) for k in IDS)+(int(row['trial']),)
            if key[:4] in expected:
                if key not in decisions or key in seen:
                    raise ValueError('Original row missing or duplicated')
                seen.add(key)
                if row['direct-bernstein']!=decisions[key]['direct-bernstein']:
                    raise ValueError('Original decision changed')
    if seen!=set(decisions):
        raise ValueError('Replay contains foreign rows')
    if len(p['cells'])!=72 or len(p['effects'])!=24:
        raise ValueError('Missing summary entries')
    for cell in p['cells']:
        key=tuple(cell[k] for k in IDS)
        count=sum(int(r[cell['method']]) for r in groups[key])
        if count!=cell['alarms'] or cell['trials']!=512 or cell['rate']!=count/512:
            raise ValueError('Rate mismatch')
        close(ci(count,512,.05),cell['ci95_marginal'])
        close(ci(count,512,.025/72),cell['ci_joint_nominal'])
    for effect in p['effects']:
        key=tuple(effect[k] for k in IDS)
        pairs=Counter((int(r['berger-boos-envelope']),int(r['direct-bernstein'])) for r in groups[key])
        w,l=pairs[(1,0)],pairs[(0,1)]
        if (w,l,(w-l)/512)!=(effect['wins_bb'],effect['losses_bb'],effect['difference']):
            raise ValueError('Paired effect mismatch')
        cw,cl=ci(w,512,.025/48),ci(l,512,.025/48)
        close([cw[0]-cl[1],cw[1]-cl[0]],effect['ci_joint_nominal'])
    unresolved=sum(int(r['unresolved_directions']) for r in decisions.values())
    if unresolved!=p['direction_statuses'].get('budget-unresolved',0):
        raise ValueError('Unresolved tally mismatch')
    return {'status':'passed','selected_original_pairs':len(seen),'rates':72,'paired_effects':24,
            'original_and_comparator_hashes':'match','unresolved_directions':unresolved,
            'limits':'Independent bookkeeping and interval inversion; not a separate-library/hardware/power certificate.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--source',type=Path,
                        help='Direct-pilot output used by this replay; defaults to the published archive.')
    parser.add_argument('--save',action='store_true')
    args=parser.parse_args()
    result=verify(args.directory,source=args.source)
    result['verifier_sha256']=digest(__file__)
    if args.save:
        target=args.directory/'verification.json'
        if target.exists():
            raise ValueError('Verification output exists; refusing overwrite')
        target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
