"""Software wiring only: no logical task solver or transfer of reservoir state."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from tele_neuron.endophalon3.isa import Instruction
from tele_neuron.endophalon3_alpha.model import Model as ParameterBanks
from tele_neuron.endophalon3_alpha.session import Session as PhysicalSession

REGISTER=re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')


class SoftwareModel:
    def __init__(self,banks):
        if any(bank.head is None for bank in banks.banks.values()):
            raise ValueError('Both existing models must have fitted heads')
        self.parameters=banks

    @classmethod
    def from_cards(cls,a,b,root='.'):
        return cls(ParameterBanks.from_cards(a,b,root))

    def to_payload(self):
        return {'schema':'endophalon3-software/v1','family':'Teleneuron-Endophalon-3',
                'architecture':'two independent physical states; explicit software output registers',
                'inactive_model':'paused; no physical or RNG steps',
                'memory_semantics':'model-local input history; cross-model values require explicit registers',
                'parameters':self.parameters.to_payload()}

    @classmethod
    def from_payload(cls,payload):
        if payload.get('schema')!='endophalon3-software/v1':raise ValueError('Unsupported software model schema')
        return cls(ParameterBanks.from_payload(payload['parameters']))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_payload(),sort_keys=True,allow_nan=False).encode()).hexdigest()

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump(self.to_payload(),f,indent=2,allow_nan=False);f.write('\n')

    @classmethod
    def load(cls,path):return cls.from_payload(json.loads(Path(path).read_text()))


class Router:
    def __init__(self,model,streams=(0,)):
        self.model=model;self.selected='A';self._registers={}
        # Only immutable geometry/parameter data is reused, never live state.
        self.sessions={level:PhysicalSession(model.parameters,streams) for level in 'AB'}

    @property
    def registers(self):return dict(self._registers)

    @property
    def events(self):return sum(s.events for s in self.sessions.values())

    def dispatch(self,level,instructions):
        if level not in self.sessions or any(not isinstance(i,Instruction) or i.level!=level for i in instructions):
            raise ValueError('Dispatch requires valid instructions for exactly the selected model')
        result=self.sessions[level].step(instructions)
        self.sessions[level].selected=level
        self.selected=level
        return result

    def _bit(self,token):
        if token in ('0','1'):return int(token)
        if token.startswith('@') and REGISTER.fullmatch(token[1:]):
            if token[1:] not in self._registers:raise ValueError('Unknown output register: '+token)
            return self._registers[token[1:]]
        raise ValueError('Input must be 0, 1 or @output_register')

    def execute(self,line):
        parts=line.strip().split('->')
        if len(parts)>2:raise ValueError('Only one output register per command')
        destination=parts[1].strip() if len(parts)==2 else None
        if destination is not None and not REGISTER.fullmatch(destination):raise ValueError('Invalid output register')
        tokens=parts[0].split()
        if len(tokens)==2 and tokens[0].lower()=='level':
            if destination is not None or tokens[1].upper() not in self.sessions:raise ValueError('Invalid level selection')
            self.selected=tokens[1].upper();return None
        if len(tokens)==3:tokens=[self.selected,*tokens]
        if len(tokens)!=4:raise ValueError('Expected [A|B] OP bit bit [-> register]')
        level,op,a,b=tokens;instruction=Instruction(level.upper(),op.lower(),self._bit(a),self._bit(b))
        if self.sessions[instruction.level].count!=1:raise ValueError('Text/register execution requires one stream per model')
        # No mutation precedes full validation and successful physical execution.
        result=self.dispatch(instruction.level,[instruction])[0]
        if result.output is None:raise RuntimeError('A fitted model output is required')
        if destination is not None:self._registers[destination]=result.output
        self.selected=instruction.level
        row=asdict(result);row['features']=row['features'].tolist()
        row['source_command']=line;row['output_register']=destination
        row['model_local_event']=result.event;row['router_event']=self.events-1
        return row

    def snapshot(self):
        return {'schema':'endophalon3-software-session/v1','model':self.model.fingerprint,'selected':self.selected,
                'registers':self.registers,'sessions':{level:s.snapshot() for level,s in self.sessions.items()}}

    @classmethod
    def restore(cls,model,payload):
        if payload.get('schema')!='endophalon3-software-session/v1' or payload.get('model')!=model.fingerprint:raise ValueError('Software checkpoint/model mismatch')
        if set(payload['sessions'])!={'A','B'} or payload['selected'] not in ('A','B'):raise ValueError('Invalid router metadata')
        registers=payload['registers']
        if not isinstance(registers,dict) or any(not isinstance(k,str) or not REGISTER.fullmatch(k) or type(v) is not int or v not in (0,1) for k,v in registers.items()):raise ValueError('Invalid output registers')
        restored={}
        for level,record in payload['sessions'].items():
            if record['active'] is not None and any(bit!=int(level=='B') for bit in record['active']):raise ValueError('Physical state belongs to another local model')
            restored[level]=PhysicalSession.restore(model.parameters,record)
        result=cls.__new__(cls);result.model=model;result.sessions=restored;result.selected=payload['selected'];result._registers=deepcopy(registers)
        return result
