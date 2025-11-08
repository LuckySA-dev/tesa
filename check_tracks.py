import pandas as pd

df = pd.read_csv('p3_tracking_obb.csv')
print('Track details:')
for tid in sorted(df.track_id.unique()):
    track_df = df[df.track_id == tid]
    first = track_df.iloc[0]
    last = track_df.iloc[-1]
    print(f'Track {tid}: frames {first.frame_id}-{last.frame_id} ({len(track_df)} frames), start pos: ({first.center_x:.3f}, {first.center_y:.3f})')
