import hashlib,marshal
from worldstream.code_signature import stable_code_hash,stable_function_hash


def example(value=0):return dict(state=value,labels=('one','two'))


def test_content_fingerprint_survives_runtime_constant_reference_changes():
    before=stable_function_hash(example)
    retained=[example.__code__.co_consts,example.__code__.co_names,example.__code__.co_varnames]
    objects=[example(i) for i in range(20)]
    assert before==stable_function_hash(example)
    assert retained and objects


def test_body_and_default_changes_are_distinguished():
    original=stable_function_hash(example);defaults=example.__defaults__
    try:
        example.__defaults__=(1,)
        assert stable_function_hash(example)!=original
    finally:example.__defaults__=defaults
    a=compile('x=1','first.py','exec');b=compile('x=1','other.py','exec');c=compile('x=2','first.py','exec')
    assert stable_code_hash(a)==stable_code_hash(b)
    assert stable_code_hash(a)!=stable_code_hash(c)
