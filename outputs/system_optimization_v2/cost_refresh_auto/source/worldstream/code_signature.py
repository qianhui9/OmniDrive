"""Content fingerprints independent of marshal's runtime reference flags."""
from functools import lru_cache
import hashlib,json,struct,types


def _encoded(value):
    if value is None:return ['none']
    if value is Ellipsis:return ['ellipsis']
    if isinstance(value,bool):return ['bool',value]
    if isinstance(value,int):return ['int',str(value)]
    if isinstance(value,float):return ['float64',struct.pack('>d',value).hex()]
    if isinstance(value,complex):return ['complex128',struct.pack('>dd',value.real,value.imag).hex()]
    if isinstance(value,str):return ['str',value]
    if isinstance(value,bytes):return ['bytes',value.hex()]
    if isinstance(value,types.CodeType):return ['code',_code(value)]
    if isinstance(value,(tuple,list)):return [type(value).__name__,[_encoded(v) for v in value]]
    if isinstance(value,frozenset):
        items=[_encoded(v) for v in value];items.sort(key=_json);return ['frozenset',items]
    if isinstance(value,dict):return ['dict',[[_encoded(k),_encoded(v)] for k,v in value.items()]]
    # Offline module/forward probes can carry object defaults. Such wrappers
    # deliberately get a local witness and cannot inherit portable native keys.
    return ['nonportable_default',type(value).__module__,type(value).__qualname__,id(value)]


def _json(value):return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'))


def _code(code):
    fields=('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals','co_stacksize','co_flags',
        'co_code','co_consts','co_names','co_varnames','co_freevars','co_cellvars','co_exceptiontable')
    # Filenames/line tables do not change execution. Bytecode, literal values,
    # variable binding and exception tables do; encode each by value, not ID.
    return {name:_encoded(getattr(code,name,None)) for name in fields}


@lru_cache(maxsize=4096)
def stable_code_hash(code):
    return hashlib.sha256(_json(['code-content-v1',_code(code)]).encode()).hexdigest()


def stable_function_hash(function):
    rows=[];seen=set();function=getattr(function,'__func__',function)
    while function is not None and id(function) not in seen:
        seen.add(id(function));actual=getattr(function,'__func__',function);code=getattr(actual,'__code__',None)
        rows.append([getattr(actual,'__module__',type(actual).__module__),getattr(actual,'__qualname__',type(actual).__qualname__),
            stable_code_hash(code) if code is not None else str(type(actual)),
            _encoded(getattr(actual,'__defaults__',None)),_encoded(getattr(actual,'__kwdefaults__',None))])
        function=getattr(actual,'__wrapped__',None)
    return hashlib.sha256(_json(['function-content-v1',rows]).encode()).hexdigest()
