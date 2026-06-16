"""Commit Stage 0 metrics using the framework's compute_metric."""
import json
import sys

sys.path.insert(0, "src")

def main():
    # Load behavioral results
    with open('scratch/stage0_behavioral_results.json') as f:
        data = json.load(f)
    
    predictions = [1 if r['correct'] else 0 for r in data['results']]
    labels = [1] * len(predictions)
    
    accuracy = data['accuracy']
    mean_logit_diff = data['mean_logit_diff']
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean logit_diff: {mean_logit_diff:.4f}")
    
    # For now, just output the data
    print("\nPredictions sample:", predictions[:10])
    print("Labels sample:", labels[:10])
    print(f"Total samples: {len(predictions)}")

if __name__ == "__main__":
    main()
