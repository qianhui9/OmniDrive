from dataclasses import replace
import pytest
from worldstream.m3.historical_draft import DraftContract, HistoricalDraftCache


def contract(**kw):
    return replace(DraftContract('scene', tuple('abcdef'), 'planning', 'question',
        'prompt', 'weights', 'tokenizer', 'sampling'), **kw)


def commit(cache, c=None, frame='f0', timestamp=1.):
    c = c or contract()
    assert cache.begin_frame(c, frame, timestamp)
    return cache.commit(c, frame, timestamp, [1, 2], 'eos', completed=True,
                        parsed=True, legally_submitted=True)


def test_requires_completed_parsed_legal_answer_and_owns_tokens():
    cache=HistoricalDraftCache();c=contract();cache.begin_frame(c,'f0',1.)
    assert not cache.commit(c,'f0',1.,[1,2],'eos',completed=True,parsed=True)
    assert not cache.commit(c,'f0',1.,[1,2],'unknown',completed=True,parsed=True,legally_submitted=True)
    tokens=[1,2]
    assert cache.commit(c,'f0',1.,tokens,'eos',completed=True,parsed=True,legally_submitted=True)
    tokens.clear()
    assert cache.lookup(c).token_ids==(1,2)


@pytest.mark.parametrize('field,value', [('scene','new'),('model_signature','new'),
    ('tokenizer_signature','new'),('generation_signature','new'),('cameras',tuple('ghijkl'))])
def test_protocol_change_resets(field,value):
    cache=HistoricalDraftCache();assert commit(cache)
    c=contract(**{field:value});cache.begin_frame(c,'f1',2.)
    assert not cache.entries


@pytest.mark.parametrize('field', ['question','task','prompt_signature'])
def test_task_and_full_text_prompt_must_match(field):
    cache=HistoricalDraftCache();assert commit(cache)
    assert cache.lookup(contract(**{field:'changed'})) is None
    assert cache.lookup(contract()) is not None


def test_multiframe_same_frame_intervals_reset_and_budgets():
    cache=HistoricalDraftCache(max_entries=1);c=contract();assert commit(cache)
    assert cache.begin_frame(c,'f1',2.) and cache.lookup(c).frame=='f0'
    assert cache.begin_frame(c,'f1',2.) and cache.lookup(c) is not None
    assert cache.begin_frame(c,'f2',2.) and cache.lookup(c) is None
    assert commit(cache,frame='f3',timestamp=3.)
    cache.begin_frame(c,'f0',1.);assert not cache.entries
    assert commit(cache)
    other=contract(question='other');assert commit(cache,other)
    assert cache.lookup(c) is None
    cache.reset();assert cache.nbytes==0


def test_nonfinite_timestamp_or_missing_cameras_fails_closed():
    cache=HistoricalDraftCache();assert commit(cache)
    assert not cache.begin_frame(contract(),'bad',float('nan'))
    assert not cache.entries
    assert not cache.begin_frame(contract(cameras=('a',)),'bad',1.)
