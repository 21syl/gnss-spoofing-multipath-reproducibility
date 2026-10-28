"""Window-level and physical-state metrics adapted from the transfer experiment."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def summarize(y, probability, state):
    y, state = np.asarray(y), np.asarray(state)
    prediction = probability.argmax(1)
    support = np.unique(y)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, prediction, labels=support, average="macro", zero_division=0)
    truth, alarm = y == 1, prediction == 1

    def rate(mask):
        return float(alarm[mask].mean()) if np.any(mask) else None

    # Missing-class macro AUC is explicitly scoped, never silently called full macro AUC.
    auc = float(np.mean([roc_auc_score(y == c, probability[:, c]) for c in support])) if len(support) > 1 else None
    return {"samples": len(y), "accuracy": float(accuracy_score(y, prediction)),
            "macro_precision": float(precision), "macro_recall": float(recall),
            "macro_f1": float(f1), "macro_label_scope": "observed_true_classes",
            "observed_labels": "|".join(map(str, support)),
            "macro_auc_present_classes": auc,
            "macro_auc_all_classes": auc if len(support) == probability.shape[1] else None,
            "spoofing_auc": float(roc_auc_score(truth, probability[:, 1])) if len(np.unique(truth)) == 2 else None,
            "spoofing_detection_rate": rate(truth),
            "false_spoofing_alarm_rate": rate(~truth),
            "multipath_false_alarm_rate": rate(state == 2),
            "mixed_detection_rate": rate(state == 3)}


def stage_results(y, probability, state):
    rows = [{"stage": "all", **summarize(y, probability, state)}]
    for code, name in enumerate(["authentic", "authentic_spoofing", "authentic_multipath", "simultaneous"]):
        mask = state == code
        if mask.any():
            rows.append({"stage": name, **summarize(y[mask], probability[mask], state[mask])})
    return rows
