import pandas as pd

old = pd.read_csv('p2_localization.csv')
new = pd.read_csv('p2_localization_v2.csv')

print('\n' + '='*70)
print('COMPARISON: Old vs New Localization')
print('='*70)

print('\nOLD VERSION (fixed 100m altitude):')
print(old[['img_file', 'drone_lat', 'drone_lon', 'drone_alt']].to_string(index=False))

print('\n\nNEW VERSION (auto-estimated altitude based on bbox size):')
print(new[['img_file', 'drone_lat', 'drone_lon', 'drone_alt']].to_string(index=False))

print('\n\nDIFFERENCES:')
print('-'*70)
for i in range(len(old)):
    dlat = abs(new.iloc[i]['drone_lat'] - old.iloc[i]['drone_lat']) * 111320
    dlon = abs(new.iloc[i]['drone_lon'] - old.iloc[i]['drone_lon']) * 111320
    dalt = new.iloc[i]['drone_alt'] - old.iloc[i]['drone_alt']
    print(f'Drone {i+1}: Δlat={dlat:.1f}m, Δlon={dlon:.1f}m, Δalt={dalt:+.1f}m')

print('='*70)
