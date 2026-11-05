"""R0-required utilities; historical alternative runners have been retired."""

from .selective_answer_reuse import QUESTIONS

def question_schedule(workload,frequency,fi,natural,fixed):
    if frequency in ('every4','every2'):
        period=int(frequency[-1])
        if fi%period:return []
        count=1;ordinal=fi//period
    else:count=3 if frequency=='three' else 1;ordinal=fi
    if workload=='addressable':return list(QUESTIONS) if count==3 else [QUESTIONS[ordinal%3]]
    if workload=='mixed':return [QUESTIONS[fi%3]]+list(natural[:2])
    return list((fixed if workload=='fixed' else natural)[:count])
