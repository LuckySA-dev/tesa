"""Check Problem 1 CSV output format"""
import pandas as pd

df = pd.read_csv('problem1_output.csv')

print('='*60)
print('📋 CSV FORMAT CHECK - Problem 1')
print('='*60)

print('\n1. Columns:', list(df.columns))
print('   Expected: frame_id, object_id, bbox_x, bbox_y, bbox_w, bbox_h')

print('\n2. First 10 rows:')
print(df.head(10))

print('\n3. Data types:')
print(df.dtypes)

print('\n4. Summary:')
print(f'   • Total detections: {len(df)}')
print(f'   • Unique objects: {df["object_id"].nunique()}')
print(f'   • Frame range: {df["frame_id"].min()} - {df["frame_id"].max()}')
print(f'   • Object ID range: {df["object_id"].min()} - {df["object_id"].max()}')

print('\n5. Sample detections per frame:')
frame_counts = df.groupby('frame_id').size()
print(f'   • Min detections/frame: {frame_counts.min()}')
print(f'   • Max detections/frame: {frame_counts.max()}')
print(f'   • Avg detections/frame: {frame_counts.mean():.1f}')

print('\n' + '='*60)
print('✅ FORMAT CHECK COMPLETE')
print('='*60)
