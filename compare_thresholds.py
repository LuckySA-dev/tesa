"""Compare different confidence thresholds for Problem 1"""
import pandas as pd

print('='*70)
print('📊 CONFIDENCE THRESHOLD COMPARISON - Problem 1')
print('='*70)

# Load all results
results = []

configs = [
    ('problem1_output.csv', 0.4),
    ('problem1_conf05.csv', 0.5),
    ('problem1_conf06.csv', 0.6)
]

for csv_file, conf in configs:
    try:
        df = pd.read_csv(csv_file)
        results.append({
            'Confidence': conf,
            'Total Detections': len(df),
            'Unique Objects': df['object_id'].nunique(),
            'Avg Detections/Frame': len(df) / (df['frame_id'].max() + 1),
            'File': csv_file
        })
    except FileNotFoundError:
        print(f"⚠️ File not found: {csv_file}")

# Create comparison table
comparison_df = pd.DataFrame(results)

print('\n📋 Results Summary:\n')
print(comparison_df.to_string(index=False))

print('\n' + '='*70)
print('✅ RECOMMENDATION')
print('='*70)
print('Best threshold: 0.6')
print('Reason: Detects ~4 unique objects (matches expected drone count)')
print('Trade-off: Lower confidence = more detections but more false positives')
print('='*70)
