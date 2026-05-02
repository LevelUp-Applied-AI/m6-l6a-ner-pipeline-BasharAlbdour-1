import pytest
import pandas as pd
import sys
import os
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tier3_evaluator import NEREvaluator
 
 
@pytest.fixture
def evaluator():
    return NEREvaluator()
 
 
def make_df(rows):
    return pd.DataFrame(rows, columns=[
        "text_id", "entity_text", "entity_label", "start_char", "end_char"
    ])
 
 
def test_empty_predictions(evaluator):
    pred = make_df([])
    gold = make_df([
        {"text_id": 1, "entity_text": "IPCC",   "entity_label": "ORG",
         "start_char": 0, "end_char": 4},
        {"text_id": 1, "entity_text": "Jordan", "entity_label": "GPE",
         "start_char": 10, "end_char": 16},
    ])
    result = evaluator.exact_match(pred, gold)
    assert result.tp == 0
    assert result.fp == 0
    assert result.fn == 2
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
 
 
def test_empty_gold(evaluator):
    pred = make_df([
        {"text_id": 1, "entity_text": "IPCC", "entity_label": "ORG",
         "start_char": 0, "end_char": 4},
    ])
    gold = make_df([])
    result = evaluator.exact_match(pred, gold)
    assert result.tp == 0
    assert result.fp == 1
    assert result.fn == 0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
 
 
def test_perfect_predictions(evaluator):
    data = [
        {"text_id": 1, "entity_text": "IPCC",   "entity_label": "ORG",
         "start_char": 4,  "end_char": 8},
        {"text_id": 1, "entity_text": "Jordan", "entity_label": "GPE",
         "start_char": 20, "end_char": 26},
    ]
    pred = make_df(data)
    gold = make_df(data)
    result = evaluator.exact_match(pred, gold)
    assert result.tp == 2
    assert result.fp == 0
    assert result.fn == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
 
 
def test_type_error_exact_vs_type_agnostic(evaluator):
    pred = make_df([
        {"text_id": 1, "entity_text": "Paris Agreement",
         "entity_label": "EVENT",
         "start_char": 10, "end_char": 25},
    ])
    gold = make_df([
        {"text_id": 1, "entity_text": "Paris Agreement",
         "entity_label": "LAW",
         "start_char": 10, "end_char": 25},
    ])
    exact    = evaluator.exact_match(pred, gold)
    agnostic = evaluator.type_agnostic_match(pred, gold)
 
    assert exact.tp == 0
    assert exact.fp == 1
    assert exact.fn == 1
 
    assert agnostic.tp == 1
    assert agnostic.fp == 0
    assert agnostic.fn == 0
 
 
def test_boundary_error_partial_vs_exact(evaluator):
    pred = make_df([
        {"text_id": 1, "entity_text": "the Paris Agreement",
         "entity_label": "LAW",
         "start_char": 7, "end_char": 26},
    ])
    gold = make_df([
        {"text_id": 1, "entity_text": "Paris Agreement",
         "entity_label": "LAW",
         "start_char": 11, "end_char": 26},
    ])
    exact   = evaluator.exact_match(pred, gold)
    partial = evaluator.partial_match(pred, gold)
 
    assert exact.tp == 0
 
    assert partial.tp == 1
 
 
def test_single_vs_multi_token(evaluator):
    pred = make_df([
        {"text_id": 1, "entity_text": "IPCC",
         "entity_label": "ORG", "start_char": 4, "end_char": 8},
        {"text_id": 1, "entity_text": "Paris Agreement",
         "entity_label": "LAW", "start_char": 20, "end_char": 35},
    ])
    gold = make_df([
        {"text_id": 1, "entity_text": "IPCC",
         "entity_label": "ORG", "start_char": 4, "end_char": 8},
        {"text_id": 1, "entity_text": "Paris Agreement",
         "entity_label": "LAW", "start_char": 20, "end_char": 35},
    ])
    result = evaluator.exact_match(pred, gold)
    assert result.tp == 2
    assert result.f1 == 1.0
 
 
def test_overlapping_predictions(evaluator):
    pred = make_df([
        {"text_id": 1, "entity_text": "Bonn Climate Conference",
         "entity_label": "ORG",   "start_char": 10, "end_char": 33},
        {"text_id": 1, "entity_text": "Bonn Climate Conference",
         "entity_label": "EVENT", "start_char": 10, "end_char": 33},
    ])
    gold = make_df([
        {"text_id": 1, "entity_text": "Bonn Climate Conference",
         "entity_label": "EVENT", "start_char": 10, "end_char": 33},
    ])
    result = evaluator.exact_match(pred, gold)
    assert result.tp == 1
    assert result.fp == 1
    assert result.fn == 0
 
 
def test_micro_vs_macro_differ(evaluator):
    pred = make_df([
        {"text_id": 1, "entity_text": "IPCC",
         "entity_label": "ORG", "start_char": 0, "end_char": 4},
    ])
    gold = make_df([
        {"text_id": 1, "entity_text": "IPCC",
         "entity_label": "ORG", "start_char": 0, "end_char": 4},
        {"text_id": 2, "entity_text": "Jordan",
         "entity_label": "GPE", "start_char": 5, "end_char": 11},
    ])
    micro = evaluator.micro_average(pred, gold, strategy="exact")
    macro = evaluator.macro_average(pred, gold, strategy="exact")
 
    assert macro["f1"] == pytest.approx(0.5, abs=0.01)
 
    assert micro.tp == 1
    assert micro.fn == 1
    assert micro.f1 == pytest.approx(0.667, abs=0.01)
 
 
def test_error_analysis_categories(evaluator):
    pred = make_df([
        {"text_id": 1, "entity_text": "IPCC",
         "entity_label": "ORG",  "start_char": 0,  "end_char": 4},
        {"text_id": 1, "entity_text": "the Paris Agreement",
         "entity_label": "LAW",  "start_char": 7,  "end_char": 26},
        {"text_id": 1, "entity_text": "random text",
         "entity_label": "ORG",  "start_char": 50, "end_char": 61},
    ])
    gold = make_df([
        {"text_id": 1, "entity_text": "IPCC",
         "entity_label": "WORK_OF_ART", "start_char": 0,  "end_char": 4},
        {"text_id": 1, "entity_text": "Paris Agreement",
         "entity_label": "LAW",         "start_char": 11, "end_char": 26},
        {"text_id": 1, "entity_text": "Jordan",
         "entity_label": "GPE",         "start_char": 70, "end_char": 76},
    ])
    errors = evaluator.error_analysis(pred, gold)
    s      = errors["summary"]
 
    assert s["type_errors"]      >= 1
    assert s["boundary_errors"]  >= 1
    assert s["missing_entities"] >= 1
    assert s["spurious_entities"] >= 1
 
 
def test_multi_text_evaluation(evaluator):
    pred = make_df([
        {"text_id": 1, "entity_text": "IPCC",
         "entity_label": "ORG", "start_char": 0, "end_char": 4},
        {"text_id": 2, "entity_text": "Jordan",
         "entity_label": "GPE", "start_char": 0, "end_char": 6},
    ])
    gold = make_df([
        {"text_id": 1, "entity_text": "IPCC",
         "entity_label": "ORG", "start_char": 0, "end_char": 4},
        {"text_id": 2, "entity_text": "Jordan",
         "entity_label": "GPE", "start_char": 0, "end_char": 6},
    ])
    result = evaluator.exact_match(pred, gold)
    assert result.tp == 2
    assert result.fp == 0
    assert result.fn == 0
    assert result.f1 == 1.0