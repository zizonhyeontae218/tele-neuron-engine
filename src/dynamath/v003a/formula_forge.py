"""Deterministic compact catalog of typed bounded formula graphs."""
import random, hashlib, json, collections
from .ast import canonical, dumps, validate

def leaf(x): return {'op':x}
def unary(op,x): return {'op':op,'arg':x}
def binary(op,a,b): return {'op':op,'left':a,'right':b}

def catalog(count=20000,seed=43):
    r=random.Random(seed); seen=set(); rows=[]; ops=['tanh','sin','softsign','atan','abs','sigmoid','poly']; bins=['add','sub','mul','residual','gate']
    # Ten labeled constructive families; every graph depends on z.
    while len(rows)<count:
        family=len(rows)%10
        z=leaf('z'); u=lambda x:unary(r.choice(ops),x)
        def branch(depth):
            x=leaf(r.choice(['z','S','m']))
            for _ in range(depth): x=u(x)
            return x
        a=z
        for _ in range(r.randint(1,4)): a=u(a)
        b=branch(r.randint(1,4))
        if family==0: n=unary(ops[len(rows)//10],z) if len(rows)<70 else u(a)
        elif family==1: n=u(u(a))
        elif family==2: n=binary('add',a,b)
        elif family==3: n=binary('mul',a,b)
        elif family==4: n=binary('residual',a,b)
        elif family==5: n=binary(r.choice(bins),a,u(leaf('S')))
        elif family==6: n=binary(r.choice(bins),a,u(leaf('m')))
        elif family==7: n=binary(r.choice(['add','mul']),unary('poly',a),unary('sin',b))
        elif family==8: n=binary('gate',a,b)
        else: n=binary(r.choice(bins),u(a),binary(r.choice(bins),b,branch(1)))
        try:
            n=canonical(n); validate(n); key=dumps(n)
        except ValueError: continue
        if key not in seen: seen.add(key);rows.append({'family':family,'ast':n})
    return rows

def manifest(rows,seed):
    raw=''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in rows).encode()
    return {'seed':seed,'count':len(rows),'unique':len({dumps(x['ast']) for x in rows}),'families':dict(sorted(collections.Counter(str(x['family']) for x in rows).items())), 'catalog_sha256':hashlib.sha256(raw).hexdigest(),'serialization':'canonical JSON sorted keys, compact, UTF-8, one row per line including newline', 'generator':'dynamath.v003a.formula_forge.catalog'}
