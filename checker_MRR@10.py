import pandas as pd

def parse_cid_list(cid_str):
    # Parse a string representation of cids like '[73560 85057 69992]' into a list of integers
    return list(map(int, cid_str.strip("[]").split()))

def calculate_mrr(predicted_file, ground_truth_file, k=10):
    # Load ground truth and predictions
    ground_truth_df = pd.read_csv(ground_truth_file)
    predictions = {}

    # Read the predicted results from predict.txt
    with open(predicted_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            qid = int(parts[0])
            top_cids = list(map(int, parts[1:1+k]))  # Only take top-k predictions
            predictions[qid] = top_cids

    reciprocal_ranks = []

    # Calculate MRR@10
    for idx, row in ground_truth_df.iterrows():
        qid = row['qid']
        ground_truth_cids = parse_cid_list(row['cid'])  # Parse ground truth `cid` list for this query

        # Check if the query ID has predictions
        if qid not in predictions:
            print(f"No predictions found for qid {qid}")
            continue

        # Find the first rank where any ground truth `cid` is present in predictions
        rank = None
        for i, pred_cid in enumerate(predictions[qid]):
            if pred_cid in ground_truth_cids:
                rank = i + 1  # Rank is 1-based
                break

        # If we found a relevant document, calculate its reciprocal rank
        if rank is not None:
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)  # No relevant document found in top-k predictions

    # Calculate MRR@10
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    return mrr

# Usage
predicted_file = './predict_val_oldscore.txt'
ground_truth_file = './val_tokenized.csv'
mrr_score = calculate_mrr(predicted_file, ground_truth_file, k=10)
print(f"MRR@10: {mrr_score:.4f}")
