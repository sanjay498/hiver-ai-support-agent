"""
Comprehensive Automated Evaluation Pipeline.
Evaluates:
1. Intent Classification: Accuracy, Macro F1, Per-intent P/R, Confusion Matrix.
2. Escalation Policy: Precision, Recall, F1, Confusion Matrix, and Unsafe Auto Rate.
3. Historical Retrieval: Recall@1, Recall@3, Recall@5.
4. Response Quality: LLM-as-a-judge 6-dimension rubric (1-5 scale).
5. Baseline Comparison: Majority Class, TF-IDF + Logistic Regression, and Main AI.
6. Automated Failure Mode Analysis (Top 5 failure categories).
Saves artifacts to data/processed/evaluation_results.json and report/figures/.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from dotenv import load_dotenv

from src.pipeline import CustomerSupportPipeline, SupportAgentResponse
from src.evaluation.baselines import train_and_eval_baselines
from src.evaluation.judge import LLMJudge, RubricScores
from src.evaluation.human_agreement import run_human_judge_agreement_study

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("src.evaluation.evaluate")

GOLDEN_PATH = Path("data/golden/golden_set.json")
RESULTS_OUTPUT_PATH = Path("data/processed/evaluation_results.json")
FIGURES_DIR = Path("report/figures")


def evaluate_system(golden_path: Path = GOLDEN_PATH, output_path: Path = RESULTS_OUTPUT_PATH) -> Dict[str, Any]:
    logger.info("Starting comprehensive evaluation on %s...", golden_path)
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = CustomerSupportPipeline()
    judge = LLMJudge()
    
    true_intents = []
    pred_intents = []
    true_actions = []
    pred_actions = []
    
    retrieval_at_1 = []
    retrieval_at_3 = []
    retrieval_at_5 = []
    
    judge_scores_list = []
    
    detailed_records = []
    failures = []
    
    logger.info("Executing pipeline over %d evaluation examples...", len(golden_data))
    for idx, ex in enumerate(golden_data):
        msg = ex["customer_message"]
        gold_intent = ex["gold_intent"]
        gold_action = ex["gold_action"]  # "AUTO" or "ESCALATE"
        cid = ex.get("conversation_id")
        
        # Process through full pipeline (excluding current conversation_id for leakage prevention)
        resp: SupportAgentResponse = pipeline.process(
            customer_message=msg,
            context=ex.get("context"),
            query_conversation_id=cid,
            top_k_evidence=5
        )
        
        p_intent = resp.predicted_intent
        # Standardize action labels: "AUTO" vs "ESCALATE"
        p_action = "AUTO" if resp.decision == "AUTO_HANDLE" else "ESCALATE"
        
        true_intents.append(gold_intent)
        pred_intents.append(p_intent)
        true_actions.append(gold_action)
        pred_actions.append(p_action)
        
        # Retrieval Recall Evaluation
        # Precedent is relevant if any retrieved evidence shares the customer intent or keywords
        retrieved_texts = [e.customer_message.lower() for e in resp.evidence]
        gold_kws = gold_intent.split("_")
        
        hit_1 = False
        hit_3 = False
        hit_5 = False
        
        if len(retrieved_texts) >= 1:
            hit_1 = any(kw in retrieved_texts[0] for kw in gold_kws) or (resp.evidence[0].similarity_score >= 0.55)
        if len(retrieved_texts) >= 3:
            hit_3 = hit_1 or any(any(kw in t for kw in gold_kws) for t in retrieved_texts[:3])
        if len(retrieved_texts) >= 5:
            hit_5 = hit_3 or any(any(kw in t for kw in gold_kws) for t in retrieved_texts[:5])
            
        retrieval_at_1.append(1 if hit_1 else 0)
        retrieval_at_3.append(1 if hit_3 else 0)
        retrieval_at_5.append(1 if hit_5 else 0)
        
        # Judge Scoring
        reqs = ex.get("expected_response_requirements", ["Acknowledge question", "Provide accurate guidance"])
        rubric: RubricScores = judge.evaluate(
            customer_message=msg,
            generated_reply=resp.draft_reply,
            expected_requirements=reqs,
            context=ex.get("context")
        )
        judge_scores_list.append(rubric.model_dump())
        
        # Failure tracking
        is_intent_fail = (gold_intent != p_intent)
        is_action_fail = (gold_action != p_action)
        
        record = {
            "id": ex["id"],
            "customer_message": msg,
            "gold_intent": gold_intent,
            "pred_intent": p_intent,
            "intent_confidence": resp.intent_confidence,
            "gold_action": gold_action,
            "pred_action": p_action,
            "escalation_reason": resp.escalation_reason,
            "draft_reply": resp.draft_reply,
            "judge_overall": rubric.overall_score
        }
        detailed_records.append(record)
        
        if is_intent_fail or is_action_fail:
            failures.append({
                "id": ex["id"],
                "message": msg,
                "gold_intent": gold_intent,
                "pred_intent": p_intent,
                "gold_action": gold_action,
                "pred_action": p_action,
                "escalation_reason": resp.escalation_reason,
                "top_similarity": resp.evidence[0].similarity_score if resp.evidence else 0.0,
                "failure_type": "UNSAFE_AUTO" if (gold_action == "ESCALATE" and p_action == "AUTO") else
                                ("UNNECESSARY_ESCALATION" if (gold_action == "AUTO" and p_action == "ESCALATE") else "INTENT_MISMATCH")
            })

    logger.info("Evaluation complete over %d examples. Computing metrics...", len(golden_data))

    # --- 1. INTENT METRICS ---
    unique_intents = sorted(list(set(true_intents) | set(pred_intents)))
    intent_acc = accuracy_score(true_intents, pred_intents)
    intent_macro_f1 = f1_score(true_intents, pred_intents, average="macro", zero_division=0)
    intent_weighted_f1 = f1_score(true_intents, pred_intents, average="weighted", zero_division=0)
    
    intent_report = classification_report(true_intents, pred_intents, output_dict=True, zero_division=0)
    intent_cm = confusion_matrix(true_intents, pred_intents, labels=unique_intents)
    
    # Save Intent Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(intent_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=unique_intents, yticklabels=unique_intents)
    plt.title("Intent Classification Confusion Matrix", fontsize=14, pad=12)
    plt.xlabel("Predicted Intent", fontsize=11)
    plt.ylabel("Gold Intent", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_plot_path = FIGURES_DIR / "intent_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    logger.info("Saved intent confusion matrix plot to %s", cm_plot_path)

    # --- 2. ESCALATION METRICS ---
    # Binary: ESCALATE (positive) vs AUTO (negative)
    esc_labels = ["ESCALATE", "AUTO"]
    esc_cm = confusion_matrix(true_actions, pred_actions, labels=esc_labels)
    # esc_cm format: [[TP_esc, FN_esc], [FP_esc, TN_esc]]
    tp_esc = int(esc_cm[0, 0])
    fn_esc = int(esc_cm[0, 1])  # Should have been ESCALATE, predicted AUTO (UNSAFE AUTO!)
    fp_esc = int(esc_cm[1, 0])  # Should have been AUTO, predicted ESCALATE (UNNECESSARY ESCALATION)
    tn_esc = int(esc_cm[1, 1])
    
    total_true_escalate = tp_esc + fn_esc
    unsafe_auto_rate = (fn_esc / total_true_escalate) if total_true_escalate > 0 else 0.0
    
    esc_precision = precision_score(true_actions, pred_actions, pos_label="ESCALATE", zero_division=0)
    esc_recall = recall_score(true_actions, pred_actions, pos_label="ESCALATE", zero_division=0)
    esc_f1 = f1_score(true_actions, pred_actions, pos_label="ESCALATE", zero_division=0)
    
    # Save Escalation Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(esc_cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["ESCALATE", "AUTO"], yticklabels=["ESCALATE", "AUTO"])
    plt.title("Escalation Decision Matrix (AUTO vs ESCALATE)", fontsize=12, pad=10)
    plt.xlabel("Predicted Action", fontsize=10)
    plt.ylabel("Gold Action", fontsize=10)
    plt.tight_layout()
    esc_plot_path = FIGURES_DIR / "escalation_confusion_matrix.png"
    plt.savefig(esc_plot_path, dpi=200)
    plt.close()

    # --- 3. RETRIEVAL METRICS ---
    recall_1 = float(np.mean(retrieval_at_1))
    recall_3 = float(np.mean(retrieval_at_3))
    recall_5 = float(np.mean(retrieval_at_5))

    # --- 4. LLM JUDGE QUALITY SCORES ---
    df_judge = pd.DataFrame(judge_scores_list)
    judge_means = {col: round(float(df_judge[col].mean()), 2) for col in df_judge.columns if col != "critique"}

    # --- 5. BASELINES COMPARISON ---
    baseline_results = train_and_eval_baselines()

    # --- 6. HUMAN-JUDGE AGREEMENT ---
    agreement_results = run_human_judge_agreement_study()

    # --- 7. FAILURE MODE CATEGORIZATION (Top 5) ---
    failure_types = pd.Series([f["failure_type"] for f in failures]).value_counts().to_dict()
    top_failures = failures[:10]

    final_results = {
        "dataset": {
            "total_examples": len(golden_data),
            "brand": "AmazonHelp",
            "intents_count": len(unique_intents)
        },
        "intent_metrics": {
            "accuracy": round(float(intent_acc), 4),
            "macro_f1": round(float(intent_macro_f1), 4),
            "weighted_f1": round(float(intent_weighted_f1), 4),
            "per_intent_report": intent_report,
            "confusion_matrix": intent_cm.tolist(),
            "labels": unique_intents
        },
        "escalation_metrics": {
            "precision_escalate": round(float(esc_precision), 4),
            "recall_escalate": round(float(esc_recall), 4),
            "f1_escalate": round(float(esc_f1), 4),
            "unsafe_auto_rate": round(float(unsafe_auto_rate), 4),
            "unsafe_auto_count": fn_esc,
            "total_true_escalate": total_true_escalate,
            "confusion_matrix": {
                "TP_escalate": tp_esc,
                "FN_unsafe_auto": fn_esc,
                "FP_over_escalate": fp_esc,
                "TN_safe_auto": tn_esc
            }
        },
        "retrieval_metrics": {
            "recall_at_1": round(recall_1, 4),
            "recall_at_3": round(recall_3, 4),
            "recall_at_5": round(recall_5, 4)
        },
        "response_quality_judge": judge_means,
        "baselines_comparison": {
            "baseline_majority": baseline_results["baseline_majority"],
            "baseline_tfidf_lr": baseline_results["baseline_tfidf_lr"],
            "main_ai_system": {
                "name": "Main AI System",
                "accuracy": round(float(intent_acc), 4),
                "macro_f1": round(float(intent_macro_f1), 4),
                "weighted_f1": round(float(intent_weighted_f1), 4)
            }
        },
        "human_judge_agreement": agreement_results["overall_summary"],
        "failure_analysis": {
            "total_failures": len(failures),
            "failure_breakdown": failure_types,
            "representative_failures": top_failures
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
        
    logger.info("Comprehensive evaluation results written to %s", output_path)
    
    # Print Executive Summary Table
    print("\n=======================================================")
    print("           COMPREHENSIVE EVALUATION SUMMARY             ")
    print("=======================================================")
    print(f"Total Golden Evaluation Examples: {len(golden_data)}")
    print("\n[1] Intent Classification Baselines Comparison:")
    print(f"  - Baseline A (Majority Class): Acc = {baseline_results['baseline_majority']['accuracy']:.3f} | Macro F1 = {baseline_results['baseline_majority']['macro_f1']:.3f}")
    print(f"  - Baseline B (TF-IDF + LogReg): Acc = {baseline_results['baseline_tfidf_lr']['accuracy']:.3f} | Macro F1 = {baseline_results['baseline_tfidf_lr']['macro_f1']:.3f}")
    print(f"  - Main AI Intent Classifier:   Acc = {intent_acc:.3f} | Macro F1 = {intent_macro_f1:.3f}")
    print("\n[2] Escalation Safety Metrics:")
    print(f"  - Precision (ESCALATE): {esc_precision:.3f}")
    print(f"  - Recall (ESCALATE):    {esc_recall:.3f}")
    print(f"  - F1 (ESCALATE):        {esc_f1:.3f}")
    print(f"  - UNSAFE AUTO RATE:     {unsafe_auto_rate * 100:.1f}% ({fn_esc}/{total_true_escalate} hazardous cases missed)")
    print("\n[3] Retrieval Quality:")
    print(f"  - Recall@1: {recall_1:.3f} | Recall@3: {recall_3:.3f} | Recall@5: {recall_5:.3f}")
    print("\n[4] LLM Judge Response Quality (1-5 scale):")
    for dim, score in judge_means.items():
        print(f"  - {dim.capitalize()}: {score:.2f} / 5.0")
    print("\n[5] Human-Judge Agreement:")
    print(f"  - Overall MAE: {agreement_results['overall_summary']['overall_mae']:.3f} | Within-1 Agreement: {agreement_results['overall_summary']['overall_within_one_pct']:.1f}%")
    print("=======================================================\n")
    return final_results


if __name__ == "__main__":
    evaluate_system()
