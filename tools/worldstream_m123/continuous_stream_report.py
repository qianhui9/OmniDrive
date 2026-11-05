"""R0-required utilities; historical alternative runners have been retired."""

from .target_complete_report import grade_current

from .fact_bypass_identity import question_target, evaluate

def grade(q,text,token,objects):
    from worldstream.m3.task_conditioner import question_facts
    if question_facts(q).intent=='unsupported':return grade_current(q,text,token,objects,'val')
    return evaluate(q,text,objects[token],question_target(q,objects[token]),token)
