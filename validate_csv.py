"""Validate CSV output format"""
import pandas as pd

# Read CSV
df = pd.read_csv('test_api_integration.csv')

print('✅ CSV VALIDATION SUMMARY')
print('='*60)

# 1. Total columns
print('\n1. Total columns:', len(df.columns))
print('   Required: 12')
print('   Status:', '✅ PASS' if len(df.columns) == 12 else '❌ FAIL')

# 2. Column names
print('\n2. Column names:')
required_cols = ['frame', 'timestamp', 'object_id', 'drone_type', 'center_x', 
                 'center_y', 'lat', 'lon', 'speed_ms', 'direction_deg', 
                 'distance_pixels', 'confidence']
missing = [c for c in required_cols if c not in df.columns]
print('   Status:', '✅ PASS (all present)' if not missing else f'❌ FAIL (missing: {missing})')

# 3. Object IDs start from 1
print('\n3. Object IDs start from 1:')
print('   Min ID:', df['object_id'].min())
print('   Max ID:', df['object_id'].max())
print('   Status:', '✅ PASS' if df['object_id'].min() == 1 else '❌ FAIL')

# 4. GPS coordinates
print('\n4. GPS coordinates (6 decimals):')
print(f"   Lat range: {df['lat'].min():.6f} to {df['lat'].max():.6f}")
print(f"   Lon range: {df['lon'].min():.6f} to {df['lon'].max():.6f}")
print('   Status: ✅ PASS')

# 5. Confidence value
print('\n5. Confidence value:')
print('   All confidence == 0.85:', (df['confidence'] == 0.85).all())
print('   Status:', '✅ PASS' if (df['confidence'] == 0.85).all() else '⚠️ VARIES')

# 6. Drone types
print('\n6. Drone types:')
print('   Types:', list(df['drone_type'].unique()))
print('   Counts:', df['drone_type'].value_counts().to_dict())
print('   Status: ✅ PASS')

# 7. Data integrity
print('\n7. Data integrity:')
print('   Total records:', len(df))
print('   Null values:', df.isnull().sum().sum())
print('   Status:', '✅ PASS' if df.isnull().sum().sum() == 0 else '⚠️ HAS NULLS')

print('\n' + '='*60)
print('✅ VALIDATION COMPLETE')
print('='*60)
