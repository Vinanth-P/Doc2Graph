"""Audit development errors from learned relations baseline."""
import json
from pathlib import Path
from collections import Counter, defaultdict

def audit():
    data = json.loads(Path("output/evaluation/handoff_learned_dev.json").read_text(encoding="utf-8"))
    docs = data["documents_detail"]
    
    # Analyze false positives: how many have unmatched endpoints?
    fp_unmatched_source = 0
    fp_unmatched_target = 0
    fp_unmatched_both = 0
    fp_both_matched = 0
    fp_by_label = Counter()
    fp_matched_by_label = Counter()
    
    fn_by_label = Counter()
    tp_by_label = Counter()
    
    sampled_fps = []
    sampled_fns = []
    sampled_tps = []
    
    for doc in docs:
        e2e = doc["end_to_end"]
        gold_ents = doc["gold_entities"]
        gold = set(tuple(x) for x in doc["gold"])
        pred = set(tuple(x) for x in e2e["predicted"])
        evidence_map = {tuple(x["fact"]): x["text"] for x in doc.get("false_positive_evidence", [])}
        
        for p in e2e["predicted"]:
            s, r, t = p
            fp_by_label[r] += 1
            is_unmatched_s = s.startswith("unmatched:")
            is_unmatched_t = t.startswith("unmatched:")
            if (s, r, t) in gold:
                tp_by_label[r] += 1
                if len(sampled_tps) < 15:
                    sampled_tps.append({
                        "doc_idx": doc["index"],
                        "title": doc["title"],
                        "fact": (s, r, t),
                        "evidence": evidence_map.get((s, r, t), "")
                    })
            else:
                if is_unmatched_s and is_unmatched_t:
                    fp_unmatched_both += 1
                elif is_unmatched_s:
                    fp_unmatched_source += 1
                elif is_unmatched_t:
                    fp_unmatched_target += 1
                else:
                    fp_both_matched += 1
                    fp_matched_by_label[r] += 1
                
                if len(sampled_fps) < 50:
                    sampled_fps.append({
                        "doc_idx": doc["index"],
                        "title": doc["title"],
                        "fact": (s, r, t),
                        "unmatched_s": is_unmatched_s,
                        "unmatched_t": is_unmatched_t,
                        "evidence": evidence_map.get((s, r, t), "")
                    })
                    
        for g in doc["gold"]:
            s, r, t = g
            fn_by_label[r] += 1
            if (s, r, t) not in pred:
                if len(sampled_fns) < 50:
                    # check if oracle predicted it
                    oracle_pred = (s, r, t) in set(tuple(x) for x in gold_ents["predicted"])
                    sampled_fns.append({
                        "doc_idx": doc["index"],
                        "title": doc["title"],
                        "fact": (s, r, t),
                        "recovered_by_oracle": oracle_pred
                    })

    print("=== FALSE POSITIVES BREAKDOWN ===")
    print(f"Total FPs: {sum(fp_by_label.values()) - sum(tp_by_label.values())}")
    print(f"FP unmatched both: {fp_unmatched_both}")
    print(f"FP unmatched source: {fp_unmatched_source}")
    print(f"FP unmatched target: {fp_unmatched_target}")
    total_unmatched = fp_unmatched_both + fp_unmatched_source + fp_unmatched_target
    print(f"Total FP with at least one unmatched endpoint: {total_unmatched} ({total_unmatched/880:.1%})")
    print(f"FP with both endpoints aligned to gold entities: {fp_both_matched} ({fp_both_matched/880:.1%})")
    print(f"Matched FPs by label: {dict(fp_matched_by_label)}")
    print(f"All FPs by label: {dict(fp_by_label)}")
    print(f"TP by label: {dict(tp_by_label)}")
    
    Path("output/evaluation/audit_samples.json").write_text(json.dumps({
        "stats": {
            "total_fp": 880,
            "unmatched_endpoints_fp": total_unmatched,
            "both_matched_fp": fp_both_matched,
            "fp_by_label": dict(fp_by_label),
            "fp_matched_by_label": dict(fp_matched_by_label),
            "tp_by_label": dict(tp_by_label),
            "fn_by_label": dict(fn_by_label),
        },
        "sampled_fps": sampled_fps,
        "sampled_fns": sampled_fns,
        "sampled_tps": sampled_tps,
    }, indent=2), encoding="utf-8")

if __name__ == "__main__":
    audit()
