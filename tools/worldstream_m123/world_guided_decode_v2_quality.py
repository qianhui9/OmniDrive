"""R0-required utilities; historical alternative runners have been retired."""

from pathlib import Path

import json

import re

from worldstream.m3.task_conditioner import question_dependencies

NUMBER={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}


def qa_pairs(token):
    result={}
    for folder in ('conv/val','vqa/val'):
        p=Path('data/nuscenes')/folder/(token+'.json')
        if p.exists():
            for row in json.loads(p.read_text()):result[row['question']]=row['answer']
    return result


def polarity(text):
    text=text.strip().lower()
    if re.match(r'yes\b',text):return 'yes'
    if re.match(r'no\b',text) or re.match(r'there (?:is|are) (?:no|not any)\b',text):return 'no'
    return None


def facts(question,answer):
    q=question.lower();a=answer.lower();dep=question_dependencies(q)
    result={}
    if re.match(r'(?:is|are|do|does|can|should)\b',q):
        p=polarity(a)
        if p is not None:result['binary_answer']=p
    if dep.intent=='signal':
        colors=set(re.findall(r'\b(red|green|yellow|amber)\b',a));colors={'yellow' if c=='amber' else c for c in colors}
        if len(colors)==1:result['signal_color']=next(iter(colors))
    if dep.intent=='category':
        found=[]
        for name,pattern in (('car',r'cars?|sedans?|suvs?|taxi|taxis|vans?'),('truck',r'trucks?'),
            ('bus',r'bus|buses'),('pedestrian',r'pedestrians?|people'),('bicycle',r'bicycles?|cyclists?'),('motorcycle',r'motorcycles?')):
            if (re.search(r'\b(?:'+pattern+r')\b',a)
                    and not re.search(r'\b(?:no|not|without)\s+(?:\w+\s+){0,2}(?:'+pattern+r')\b',a)):
                found.append(name)
        if found:result['reference_class_set']=tuple(sorted(found))
    if dep.intent=='position' or re.search(r'\bwhere\b',q):
        text=re.sub(r'\bahead\b','front',a);text=re.sub(r'\bbehind\b','rear',text)
        found=set(re.findall(r'\b(front|rear|left|right)\b',text))
        if found and not ({'left','right'}<=found or {'front','rear'}<=found):result['relative_direction']=tuple(sorted(found))
    if dep.intent=='motion':
        stationary=bool(re.search(r'\b(stationary|parked|stopped)\b',a))
        moving=bool(re.search(r'\b(moving|travelling|traveling|driving)\b',a))
        if stationary!=moving:result['motion_state']='stationary' if stationary else 'moving'
    if re.search(r'how many|number of|count',q):
        values=re.findall(r'\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b',a)
        if 'lane' in q:
            lane=re.findall(r'\b(\d+|one|two|three|four|five|six)\s+(?:traffic )?lanes?\b',a)
            if lane:values=lane
        unique={NUMBER.get(v,int(v) if v.isdigit() else None) for v in values}
        if len(unique)==1:result['object_count']=next(iter(unique))
    if dep.intent=='distance':
        values=[float(v) for v in re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:meters?|metres?|m)\b',a)]
        if len(values)==1:result['distance_m']=values[0]
    if dep.intent=='safety':
        actions=[]
        for name,pattern in (('slow',r'slow|reduce (?:the )?speed'),('yield',r'yield|give way'),
            ('stop',r'stop|halt|wait'),('observe',r'cautio\w*|aware|watch|look|check|vigilant|observe')):
            if re.search(r'\b(?:'+pattern+r')\b',a):actions.append(name)
        if actions:result['reference_action_set']=tuple(sorted(actions))
    if dep.intent=='road':
        types=set(re.findall(r'\b(t-junction|four-way|three-way|roundabout|crossroads)\b',a))
        if len(types)==1:result['road_subtype']=next(iter(types))
    return result


def compare_facts(expected,observed):
    rows=[]
    for name,value in expected.items():
        actual=observed.get(name)
        if name in ('reference_class_set','reference_action_set'):correct=actual is not None and set(value)<=set(actual)
        elif name=='distance_m':correct=actual is not None and abs(value-actual)<=max(2.,.1*value)
        else:correct=actual==value
        rows.append(dict(fact=name,expected=value,observed=actual,correct=bool(correct)))
    return rows
