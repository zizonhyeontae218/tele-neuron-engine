"""General tree edits, with validation and explicit initialization classification."""
import copy, random
from .ast import validate, canonical, UNARY, BINARY
from .formula_forge import leaf,unary,binary

def mutate(parent,operation,seed=43,donor=None):
    r=random.Random(seed); root=copy.deepcopy(parent)
    paths=[]
    def walk(n,path=()):
        paths.append(path)
        for k in ('arg','left','right'):
            if k in n: walk(n[k],path+(k,))
    walk(root)
    path=r.choice(paths); node=root
    for k in path: node=node[k]
    if operation=='operator':
        choices=UNARY if node['op'] in UNARY else BINARY if node['op'] in BINARY else set()
        if not choices: raise ValueError('no replaceable operator')
        replacement=copy.deepcopy(node); replacement['op']=r.choice(sorted(choices-{node['op']}))
    elif operation=='add_unary': replacement=unary(r.choice(['sin','tanh','softsign']),node)
    elif operation=='delete_unary':
        if 'arg' not in node: raise ValueError('not unary')
        replacement=node['arg']
    elif operation=='insert_binary': replacement=binary('add',node,leaf('S'))
    elif operation=='replace_subtree': replacement=copy.deepcopy(donor or leaf('z'))
    elif operation=='swap_subtree':
        if donor is None: raise ValueError('donor required')
        replacement=copy.deepcopy(donor)
    elif operation=='add_state': replacement=binary('residual',node,leaf('S'))
    elif operation=='remove_state':
        if node['op']!='S': raise ValueError('not state')
        replacement=leaf('z')
    elif operation=='residual': replacement=binary('residual',node,leaf('m'))
    elif operation=='simplify':
        if 'arg' not in node: raise ValueError('not unary')
        replacement=node['arg']
    else: raise ValueError('unknown mutation')
    if path:
        p=root
        for k in path[:-1]: p=p[k]
        p[path[-1]]=replacement
    else: root=replacement
    validate(root)
    return canonical(root), ('identity' if operation in ('residual','add_state') else 'nonidentity')
