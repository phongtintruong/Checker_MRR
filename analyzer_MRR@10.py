import pandas as pd
from collections import defaultdict


def parse_cid_list(cid_str):
    """Parse a string representation of cids into a list of integers."""
    return list(map(int, cid_str.strip("[]").split()))


def analyze_predictions(predicted_file, ground_truth_file, k=10):
    # Load ground truth and predictions
    ground_truth_df = pd.read_csv(ground_truth_file)
    predictions = {}

    # Read the predicted results from predict.txt
    with open(predicted_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            qid = int(parts[0])
            top_cids = list(map(int, parts[1:1 + k]))  # Only take top-k predictions
            predictions[qid] = top_cids

    # Metrics for analysis
    reciprocal_ranks = []
    no_match_count = 0
    first_match_ranks = []
    error_analysis = defaultdict(lambda: {"false_positives": [], "false_negatives": []})

    for idx, row in ground_truth_df.iterrows():
        qid = row['qid']
        ground_truth_cids = parse_cid_list(row['cid'])

        # Check if we have predictions for this query ID
        if qid not in predictions:
            print(f"No predictions found for qid {qid}")
            no_match_count += 1
            continue

        top_cids = predictions[qid]
        found = False
        rank = None

        # Analyze top-k predictions for relevant CIDs
        for i, pred_cid in enumerate(top_cids):
            if pred_cid in ground_truth_cids:
                found = True
                rank = i + 1  # Rank is 1-based
                first_match_ranks.append(rank)
                reciprocal_ranks.append(1 / rank)
                break

        if not found:
            # No relevant prediction found in top-k
            reciprocal_ranks.append(0)
            no_match_count += 1

        # Error analysis: false positives and false negatives
        false_positives = [cid for cid in top_cids if cid not in ground_truth_cids]
        false_negatives = [cid for cid in ground_truth_cids if cid not in top_cids]

        error_analysis[qid]["false_positives"] = false_positives
        error_analysis[qid]["false_negatives"] = false_negatives

    # Aggregated insights
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    avg_rank_first_match = sum(first_match_ranks) / len(first_match_ranks) if first_match_ranks else None
    total_queries = len(ground_truth_df)

    # Summary report
    print(f"\nMRR@10: {mrr:.4f}")
    print(f"Average rank of first relevant prediction: {avg_rank_first_match if avg_rank_first_match else 'N/A'}")
    print(f"Queries with no relevant predictions in top-{k}: {no_match_count}/{total_queries}")
    print("\nError Analysis (Sample):")
    sample_size = 5  # Number of examples to display
    sample_errors = dict(list(error_analysis.items())[:sample_size])

    for qid, errors in sample_errors.items():
        print(f"\nQuery ID {qid}:")
        print(f"  False Positives: {errors['false_positives']}")
        print(f"  False Negatives: {errors['false_negatives']}")

    # Overall error insights
    total_false_positives = sum(len(errors['false_positives']) for errors in error_analysis.values())
    total_false_negatives = sum(len(errors['false_negatives']) for errors in error_analysis.values())
    print(f"\nTotal False Positives in top-{k}: {total_false_positives}")
    print(f"Total False Negatives in top-{k}: {total_false_negatives}")

    return {
        "MRR@10": mrr,
        "avg_rank_first_match": avg_rank_first_match,
        "no_match_count": no_match_count,
        "total_queries": total_queries,
        "total_false_positives": total_false_positives,
        "total_false_negatives": total_false_negatives,
        "sample_errors": sample_errors
    }


# Usage
predicted_file = './predict_top200_beta640.txt'
ground_truth_file = './val_tokenized.csv'
analysis_results = analyze_predictions(predicted_file, ground_truth_file, k=10)
# print('\nAnalysis Results:', analysis_results)
