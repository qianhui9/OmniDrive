"""R0-required utilities; historical alternative runners have been retired."""

import re

from tools.worldstream_m123.world_guided_decode_v2_quality import facts as v2_facts

from worldstream.m3.task_conditioner import question_dependencies

def facts(question,answer):
    """Condition-blind v3 offline scope correction, applied equally to GT/A/B/C.

    Incidental 'from where we are' is not a relation query. Contrasted multiple
    objects cannot share one direction fact. Stay-alert and watch-for describe
    the same reference precaution. No generated answer becomes truth.
    """
    result=v2_facts(question,answer);intent=question_dependencies(question).intent
    if intent!='position' or re.search(r'\b(but|while|however)\b',answer.lower()):
        result.pop('relative_direction',None)
    if intent=='safety' and re.search(r'\balert\b',answer.lower()):
        result['reference_action_set']=tuple(sorted(set(result.get('reference_action_set',()))|{'observe'}))
    return result
