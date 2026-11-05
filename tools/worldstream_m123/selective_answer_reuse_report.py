"""R0-required utilities; historical alternative runners have been retired."""

from dataclasses import asdict

import math

import re

from worldstream.m3.task_conditioner import question_facts

from worldstream.m3.answer_validity import fact_region, fact_directions

from worldstream.m3.world_delta import VEHICLES

from .answer_validity_quality import original_qa_grade

def mean(values):return sum(values)/len(values) if values else None


def label(question,objects):
    """Positive visible witnesses only; no physical/visual-absence equivalence."""
    profile=question_facts(question)
    if profile.intent=='unsupported':return dict(known=False,reason='outside_structured_domain')
    if profile.classes==('object',):return dict(known=False,reason='generic_visual_object_taxonomy_incomplete')
    classes=set(profile.classes)
    if 'vehicle' in classes:classes=(classes-{'vehicle'})|VEHICLES
    if 'object' in classes:classes=set()
    eligible=[g for g in objects if g['visible'] and (not classes or g['semantic_class'] in classes)
              and fact_region(g['center_ego'],profile)]
    if not eligible:return dict(known=False,reason='no_positive_visible_witness_absence_UNKNOWN')
    eligible.sort(key=lambda g:math.hypot(*g['center_ego'][:2]))
    first=eligible[0]
    if profile.selector=='nearest':
        near=[g for g in eligible[1:] if math.hypot(*g['center_ego'][:2])-math.hypot(*first['center_ego'][:2])<2.]
        if (profile.intent=='category' and any(g['semantic_class']!=first['semantic_class'] for g in near)):
            return dict(known=False,reason='annotated_nearest_class_ambiguous')
        if profile.intent=='position' and (near or any(abs(abs(v)-1.)<.5 for v in first['center_ego'][:2])):
            return dict(known=False,reason='annotated_nearest_or_sector_ambiguous')
    motion=None
    if profile.intent=='motion':
        attrs=set(first.get('attributes',[]))
        if 'vehicle.moving' in attrs:motion='moving'
        elif attrs&{'vehicle.parked','vehicle.stopped'}:motion='stationary'
        else:return dict(known=False,reason='independent_current_motion_label_unavailable')
    return dict(known=True,task=profile.intent,semantic_class=first['semantic_class'],positive=True,
                motion=motion,available_classes=sorted({g['semantic_class'] for g in eligible}),
                directions=fact_directions(first['center_ego']),annotation_ids=[g['annotation_id'] for g in eligible],
                selected_annotation=first['annotation_id'],source_frame=first['source_frame'],
                center_ego=first['center_ego'],required_facts=profile.required,
                metric='required_current_visible_object_facts',profile=asdict(profile))


def extracted_classes(text):
    text=str(text).lower()
    return {c for c,p in [('car',r'cars?|sedans?|suvs?|vans?|taxis?'),('truck',r'trucks?'),
        ('bus',r'bus|buses'),('pedestrian',r'pedestrians?|people|person'),('bicycle',r'bicycles?|cyclists?'),
        ('motorcycle',r'motorcycles?'),('trailer',r'trailers?'),('construction_vehicle',r'construction vehicles?')]
        if re.search(r'\b(?:'+p+r')\b',text)}


def grade(question,answer,token,objects):
    ref=label(question,objects)
    if question_facts(question).intent=='unsupported':
        qa=original_qa_grade(token,question,answer)
        return dict(known=qa['score'] is not None,score=qa['score'],reference=qa,
                    metric='limited_exact_current_QA_fact_recall',severe_error=False)
    if not ref['known']:return dict(known=False,score=None,reference=ref,severe_error=None)
    text=' '.join(str(answer).lower().split())
    negative=bool(re.match(r'no\b',text) or re.search(r'\b(?:there (?:is|are)|i (?:can |do )?see) (?:no|not any|not a)\b',text)
                  or re.search(r'\b(?:cannot|can\x27t|do not) (?:see|determine|identify)\b',text))
    uncertain=bool(re.search(r'\b(uncertain|unsure|cannot determine|could be|might be|not visible|not clear)\b',text))
    classes=extracted_classes(text)
    dirs=tuple(d for d,p in [('front','front|ahead'),('rear','rear|behind'),('left','left'),('right','right')]
               if re.search(r'\b(?:'+p+r')\b',text))
    task=ref['task'];correct=False
    if task=='category':correct=classes=={ref['semantic_class']} and not negative and not uncertain
    elif task=='existence':
        positive=bool(re.match(r'yes\b',text) or re.search(r'\b(?:there (?:is|are)|i (?:can )?see|visible|present)\b',text))
        correct=positive and not negative and not uncertain
        if classes and not classes&set(ref['available_classes']):correct=False
        expected_dirs=set(question_facts(question).directions)
        if dirs and expected_dirs and not expected_dirs<=set(dirs):correct=False
    elif task=='position':correct=set(dirs)==set(ref['directions']) and not negative and not uncertain
    elif task=='motion':
        moving=bool(re.search(r'\b(moving|driving|travelling|traveling)\b',re.sub(r'\bnot moving\b','',text)))
        stationary=bool(re.search(r'\b(stationary|stopped|parked)\b',text))
        correct=(moving!=stationary and ('moving' if moving else 'stationary')==ref['motion'] and not uncertain)
    severe=bool((task=='existence' and negative) or (task=='position' and any(
        a in dirs and b in ref['directions'] for a,b in [('front','rear'),('rear','front'),('left','right'),('right','left')])))
    return dict(known=True,score=float(correct),reference=ref,metric='required_current_visible_object_facts',
                extracted=dict(classes=sorted(classes),directions=dirs,negative=negative,uncertain=uncertain),severe_error=severe)
