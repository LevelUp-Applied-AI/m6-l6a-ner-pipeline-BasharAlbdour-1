import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
 
 
@dataclass
class Entity:
    text_id:    int
    text:       str
    label:      str
    start_char: int
    end_char:   int
 
    def overlaps(self, other: "Entity") -> bool:
        if self.text_id != other.text_id:
            return False
        return self.start_char < other.end_char and self.end_char > other.start_char
 
    def span_matches(self, other: "Entity") -> bool:
        return (self.text_id    == other.text_id and
                self.start_char == other.start_char and
                self.end_char   == other.end_char)
 
    def exact_matches(self, other: "Entity") -> bool:
        return self.span_matches(other) and self.label == other.label
 
 
@dataclass
class MatchResult:
    precision: float
    recall:    float
    f1:        float
    tp:        int
    fp:        int
    fn:        int
    strategy:  str = ""
 
    def to_dict(self) -> dict:
        return {
            "strategy":  self.strategy,
            "precision": round(self.precision, 4),
            "recall":    round(self.recall,    4),
            "f1":        round(self.f1,        4),
            "tp":        self.tp,
            "fp":        self.fp,
            "fn":        self.fn,
        }
 
 
def df_to_entities(df: pd.DataFrame) -> List[Entity]:
    entities = []
    for _, row in df.iterrows():
        entities.append(Entity(
            text_id    = int(row["text_id"]),
            text       = str(row["entity_text"]).lower().strip(),
            label      = str(row["entity_label"]),
            start_char = int(row.get("start_char", -1)),
            end_char   = int(row.get("end_char",   -1)),
        ))
    return entities
 
 
def _compute_scores(tp: int, fp: int, fn: int,
                    strategy: str = "") -> MatchResult:
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return MatchResult(precision=p, recall=r, f1=f1,
                       tp=tp, fp=fp, fn=fn, strategy=strategy)
 
 
class NEREvaluator:
 
 
    def exact_match(self,
                    predicted_df: pd.DataFrame,
                    gold_df:      pd.DataFrame) -> MatchResult:
        pred_set = set(zip(
            predicted_df["text_id"],
            predicted_df["entity_text"].str.lower().str.strip(),
            predicted_df["entity_label"],
        ))
        gold_set = set(zip(
            gold_df["text_id"],
            gold_df["entity_text"].str.lower().str.strip(),
            gold_df["entity_label"],
        ))
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        return _compute_scores(tp, fp, fn, strategy="exact")
 
 
    def partial_match(self,
                      predicted_df: pd.DataFrame,
                      gold_df:      pd.DataFrame) -> MatchResult:
        preds = df_to_entities(predicted_df)
        golds = df_to_entities(gold_df)
 
        matched_gold  = set()
        matched_pred  = set()
 
        for i, pred in enumerate(preds):
            for j, gold in enumerate(golds):
                if j in matched_gold:
                    continue
                if pred.label == gold.label and pred.overlaps(gold):
                    matched_gold.add(j)
                    matched_pred.add(i)
                    break
 
        tp = len(matched_pred)
        fp = len(preds) - tp
        fn = len(golds) - len(matched_gold)
        return _compute_scores(tp, fp, fn, strategy="partial")
 
 
    def type_agnostic_match(self,
                            predicted_df: pd.DataFrame,
                            gold_df:      pd.DataFrame) -> MatchResult:
        preds = df_to_entities(predicted_df)
        golds = df_to_entities(gold_df)
 
        pred_spans = {(e.text_id, e.start_char, e.end_char) for e in preds}
        gold_spans = {(e.text_id, e.start_char, e.end_char) for e in golds}
 
        tp = len(pred_spans & gold_spans)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)
        return _compute_scores(tp, fp, fn, strategy="type_agnostic")
 
 
    def micro_average(self,
                      predicted_df: pd.DataFrame,
                      gold_df:      pd.DataFrame,
                      strategy:     str = "exact") -> MatchResult:
        if strategy == "exact":
            return self.exact_match(predicted_df, gold_df)
        elif strategy == "partial":
            return self.partial_match(predicted_df, gold_df)
        elif strategy == "type_agnostic":
            return self.type_agnostic_match(predicted_df, gold_df)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. "
                             f"Use 'exact', 'partial', or 'type_agnostic'.")
 
 
    def macro_average(self,
                      predicted_df: pd.DataFrame,
                      gold_df:      pd.DataFrame,
                      strategy:     str = "exact") -> dict:
        gold_text_ids = gold_df["text_id"].unique()
        per_text = []
 
        for text_id in gold_text_ids:
            pred_text = predicted_df[predicted_df["text_id"] == text_id]
            gold_text = gold_df[gold_df["text_id"] == text_id]
 
            if strategy == "exact":
                result = self.exact_match(pred_text, gold_text)
            elif strategy == "partial":
                result = self.partial_match(pred_text, gold_text)
            elif strategy == "type_agnostic":
                result = self.type_agnostic_match(pred_text, gold_text)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
 
            per_text.append({
                "text_id":   text_id,
                "precision": result.precision,
                "recall":    result.recall,
                "f1":        result.f1,
                "tp":        result.tp,
                "fp":        result.fp,
                "fn":        result.fn,
            })
 
        per_text_df = pd.DataFrame(per_text)
 
        macro = {
            "strategy":        strategy,
            "averaging":       "macro",
            "precision":       round(per_text_df["precision"].mean(), 4),
            "recall":          round(per_text_df["recall"].mean(),    4),
            "f1":              round(per_text_df["f1"].mean(),        4),
            "n_texts":         len(per_text_df),
            "per_text_scores": per_text_df,
        }
        return macro
 
 
    def error_analysis(self,
                       predicted_df: pd.DataFrame,
                       gold_df:      pd.DataFrame) -> dict:
        preds = df_to_entities(predicted_df)
        golds = df_to_entities(gold_df)
 
        boundary_errors  = []
        type_errors      = []
        missing_entities = []
        spurious_entities = []
 
        matched_gold = set()
        matched_pred = set()
 
        for i, pred in enumerate(preds):
            for j, gold in enumerate(golds):
                if pred.exact_matches(gold):
                    matched_gold.add(j)
                    matched_pred.add(i)
                    break
 
        for i, pred in enumerate(preds):
            if i in matched_pred:
                continue
 
            found_overlap    = False
            found_span_match = False
 
            for j, gold in enumerate(golds):
                if pred.overlaps(gold):
                    found_overlap = True
                    if pred.span_matches(gold):
                        type_errors.append({
                            "text_id":        pred.text_id,
                            "pred_text":      pred.text,
                            "pred_label":     pred.label,
                            "gold_text":      gold.text,
                            "gold_label":     gold.label,
                            "start_char":     pred.start_char,
                            "end_char":       pred.end_char,
                        })
                        found_span_match = True
                        break
                    else:
                        boundary_errors.append({
                            "text_id":        pred.text_id,
                            "pred_text":      pred.text,
                            "pred_label":     pred.label,
                            "gold_text":      gold.text,
                            "gold_label":     gold.label,
                            "pred_start":     pred.start_char,
                            "pred_end":       pred.end_char,
                            "gold_start":     gold.start_char,
                            "gold_end":       gold.end_char,
                        })
                        break
 
            if not found_overlap:
                spurious_entities.append({
                    "text_id":    pred.text_id,
                    "pred_text":  pred.text,
                    "pred_label": pred.label,
                    "start_char": pred.start_char,
                    "end_char":   pred.end_char,
                })
 
        for j, gold in enumerate(golds):
            if j in matched_gold:
                continue
 
            has_any_overlap = any(
                gold.overlaps(pred) for pred in preds
            )
            if not has_any_overlap:
                missing_entities.append({
                    "text_id":    gold.text_id,
                    "gold_text":  gold.text,
                    "gold_label": gold.label,
                    "start_char": gold.start_char,
                    "end_char":   gold.end_char,
                })
 
        total_errors = (len(boundary_errors) + len(type_errors) +
                        len(missing_entities) + len(spurious_entities))
 
        def rate(n):
            return round(n / total_errors, 4) if total_errors > 0 else 0.0
 
        return {
            "summary": {
                "boundary_errors":   len(boundary_errors),
                "type_errors":       len(type_errors),
                "missing_entities":  len(missing_entities),
                "spurious_entities": len(spurious_entities),
                "total_errors":      total_errors,
                "boundary_rate":     rate(len(boundary_errors)),
                "type_rate":         rate(len(type_errors)),
                "missing_rate":      rate(len(missing_entities)),
                "spurious_rate":     rate(len(spurious_entities)),
            },
            "boundary_errors":   pd.DataFrame(boundary_errors),
            "type_errors":       pd.DataFrame(type_errors),
            "missing_entities":  pd.DataFrame(missing_entities),
            "spurious_entities": pd.DataFrame(spurious_entities),
        }
 
 
if __name__ == "__main__":
 
    import spacy
    from ner_pipeline import extract_spacy_entities
 
    nlp     = spacy.load("en_core_web_sm")
    df      = pd.read_csv("data/climate_articles.csv")
    gold_df = pd.read_csv("data/gold_entities.csv")
 
    predicted_df = extract_spacy_entities(df, nlp)
    evaluator    = NEREvaluator()
 
 
    print("=" * 65)
    print("MATCHING STRATEGY COMPARISON")
    print("=" * 65)
 
    exact    = evaluator.exact_match(predicted_df, gold_df)
    partial  = evaluator.partial_match(predicted_df, gold_df)
    agnostic = evaluator.type_agnostic_match(predicted_df, gold_df)
 
    print(f"\n{'Strategy':<18} {'Precision':>10} {'Recall':>8} "
          f"{'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 60)
    for result in [exact, partial, agnostic]:
        print(f"{result.strategy:<18} {result.precision:>10} "
              f"{result.recall:>8} {result.f1:>8} "
              f"{result.tp:>5} {result.fp:>5} {result.fn:>5}")
 
 
    print("\n" + "=" * 65)
    print("MICRO vs MACRO AVERAGING (exact match)")
    print("=" * 65)
 
    micro = evaluator.micro_average(predicted_df, gold_df, strategy="exact")
    macro = evaluator.macro_average(predicted_df, gold_df, strategy="exact")
 
    print(f"\nMicro: P={micro.precision}  R={micro.recall}  F1={micro.f1}")
    print(f"Macro: P={macro['precision']}  R={macro['recall']}  F1={macro['f1']}")
 
    print(f"\nPer-text scores (macro breakdown):")
    print(f"\n{'text_id':>8} {'Precision':>10} {'Recall':>8} "
          f"{'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 55)
    for _, row in macro["per_text_scores"].iterrows():
        print(f"{int(row['text_id']):>8} {row['precision']:>10} "
              f"{row['recall']:>8} {row['f1']:>8} "
              f"{int(row['tp']):>5} {int(row['fp']):>5} "
              f"{int(row['fn']):>5}")
 
 
    print("\n" + "=" * 65)
    print("ERROR ANALYSIS")
    print("=" * 65)
 
    errors = evaluator.error_analysis(predicted_df, gold_df)
    s      = errors["summary"]
 
    print(f"\nError distribution:")
    print(f"  Boundary errors  : {s['boundary_errors']:>4}  "
          f"({s['boundary_rate']*100:.1f}%)")
    print(f"  Type errors      : {s['type_errors']:>4}  "
          f"({s['type_rate']*100:.1f}%)")
    print(f"  Missing entities : {s['missing_entities']:>4}  "
          f"({s['missing_rate']*100:.1f}%)")
    print(f"  Spurious entities: {s['spurious_entities']:>4}  "
          f"({s['spurious_rate']*100:.1f}%)")
    print(f"  Total errors     : {s['total_errors']:>4}")
 
    if not errors["type_errors"].empty:
        print(f"\nType errors (wrong label, correct span):")
        for _, row in errors["type_errors"].iterrows():
            print(f"  text {row['text_id']}: '{row['pred_text']}' "
                  f"predicted {row['pred_label']} "
                  f"but gold says {row['gold_label']}")
 
    if not errors["boundary_errors"].empty:
        print(f"\nBoundary errors (overlapping spans, wrong boundaries):")
        for _, row in errors["boundary_errors"].head(5).iterrows():
            print(f"  text {row['text_id']}: "
                  f"pred='{row['pred_text']}'[{row['pred_start']}:{row['pred_end']}] "
                  f"gold='{row['gold_text']}'[{row['gold_start']}:{row['gold_end']}]")
 
    if not errors["missing_entities"].empty:
        print(f"\nMissing entities (gold not found at all):")
        for _, row in errors["missing_entities"].iterrows():
            print(f"  text {row['text_id']}: "
                  f"'{row['gold_text']}' ({row['gold_label']}) "
                  f"[{row['start_char']}:{row['end_char']}]")