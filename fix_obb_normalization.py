"""
YOLO OBB Format Normalizer
Converts pixel coordinates to normalized 0-1 format as required by competition
"""

import pandas as pd
import numpy as np
import argparse


def normalize_obb_coordinates(input_csv, output_csv, image_width, image_height):
    """
    Normalize bounding box coordinates to 0-1 range
    
    Input format (pixels):
        frame_id, object_id, bbox_x, bbox_y, bbox_w, bbox_h
        
    Output format (normalized 0-1):
        frame_id, object_id, center_x, center_y, w, h
        
    Args:
        input_csv: Input CSV with pixel coordinates
        output_csv: Output CSV with normalized coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
    """
    print(f"📥 Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"   • Records: {len(df)}")
    print(f"   • Image size: {image_width}×{image_height}")
    
    # Check if columns exist
    if 'bbox_x' in df.columns and 'bbox_y' in df.columns:
        # Format: bbox_x, bbox_y, bbox_w, bbox_h (top-left corner)
        print(f"\n🔄 Converting from bbox (top-left) to center coordinates...")
        
        # Calculate center coordinates (pixels)
        cx_pixel = df['bbox_x'] + df['bbox_w'] / 2
        cy_pixel = df['bbox_y'] + df['bbox_h'] / 2
        w_pixel = df['bbox_w']
        h_pixel = df['bbox_h']
        
    elif 'cx' in df.columns and 'cy' in df.columns:
        # Already center coordinates
        print(f"\n🔄 Using existing center coordinates...")
        cx_pixel = df['cx']
        cy_pixel = df['cy']
        w_pixel = df['w'] if 'w' in df.columns else df['bbox_w']
        h_pixel = df['h'] if 'h' in df.columns else df['bbox_h']
    else:
        raise ValueError("No valid coordinate columns found!")
    
    # Normalize to 0-1
    print(f"📐 Normalizing coordinates to 0-1 range...")
    
    output_df = pd.DataFrame()
    output_df['frame_id'] = df['frame_id']
    output_df['object_id'] = df['object_id']
    output_df['center_x'] = (cx_pixel / image_width).round(6)
    output_df['center_y'] = (cy_pixel / image_height).round(6)
    output_df['w'] = (w_pixel / image_width).round(6)
    output_df['h'] = (h_pixel / image_height).round(6)
    
    # Add theta if exists
    if 'theta' in df.columns:
        output_df['theta'] = df['theta'].round(2)
    
    # Validation
    print(f"\n✅ Validation:")
    assert output_df['center_x'].min() >= 0 and output_df['center_x'].max() <= 1, "center_x out of range!"
    assert output_df['center_y'].min() >= 0 and output_df['center_y'].max() <= 1, "center_y out of range!"
    assert output_df['w'].min() >= 0 and output_df['w'].max() <= 1, "w out of range!"
    assert output_df['h'].min() >= 0 and output_df['h'].max() <= 1, "h out of range!"
    print(f"   ✅ All coordinates in valid range [0, 1]")
    
    # Save
    output_df.to_csv(output_csv, index=False)
    
    print(f"\n📊 Normalized Statistics:")
    print(f"   • center_x: {output_df['center_x'].min():.4f} - {output_df['center_x'].max():.4f}")
    print(f"   • center_y: {output_df['center_y'].min():.4f} - {output_df['center_y'].max():.4f}")
    print(f"   • w: {output_df['w'].min():.4f} - {output_df['w'].max():.4f}")
    print(f"   • h: {output_df['h'].min():.4f} - {output_df['h'].max():.4f}")
    
    print(f"\n💾 Saved: {output_csv}")
    
    # Show sample
    print(f"\n📋 Sample (first 5 rows):")
    print(output_df.head())
    
    return output_df


def normalize_submission_format(input_csv, output_csv, image_width, image_height):
    """
    Normalize submission.csv format (integration format)
    
    Input: video_id, frame, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred
    Output: video_id, frame, center_x, center_y, range_m_pred, azimuth_deg_pred, elevation_deg_pred
    """
    print(f"📥 Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"   • Records: {len(df)}")
    print(f"   • Image size: {image_width}×{image_height}")
    
    # Normalize cx, cy
    print(f"\n📐 Normalizing cx, cy to 0-1 range...")
    
    output_df = df.copy()
    output_df['center_x'] = (df['cx'] / image_width).round(6)
    output_df['center_y'] = (df['cy'] / image_height).round(6)
    
    # Drop original pixel columns, keep predictions
    output_df = output_df.drop(columns=['cx', 'cy'])
    
    # Reorder columns
    cols = ['video_id', 'frame', 'center_x', 'center_y', 'range_m_pred', 'azimuth_deg_pred', 'elevation_deg_pred']
    output_df = output_df[cols]
    
    # Validation
    print(f"\n✅ Validation:")
    assert output_df['center_x'].min() >= 0 and output_df['center_x'].max() <= 1, "center_x out of range!"
    assert output_df['center_y'].min() >= 0 and output_df['center_y'].max() <= 1, "center_y out of range!"
    print(f"   ✅ All coordinates in valid range [0, 1]")
    
    # Save
    output_df.to_csv(output_csv, index=False)
    
    print(f"\n📊 Statistics:")
    print(f"   • center_x: {output_df['center_x'].min():.4f} - {output_df['center_x'].max():.4f}")
    print(f"   • center_y: {output_df['center_y'].min():.4f} - {output_df['center_y'].max():.4f}")
    
    print(f"\n💾 Saved: {output_csv}")
    print(f"\n📋 Sample (first 3 rows):")
    print(output_df.head(3))
    
    return output_df


# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Normalize OBB coordinates to 0-1 range')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file')
    parser.add_argument('--width', type=int, default=2048,
                       help='Image width in pixels (default: 2048)')
    parser.add_argument('--height', type=int, default=1364,
                       help='Image height in pixels (default: 1364)')
    parser.add_argument('--format', type=str, default='detection',
                       choices=['detection', 'submission'],
                       help='Input format type')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔧 YOLO OBB COORDINATE NORMALIZER")
    print("="*70)
    print(f"Task: Convert pixel coordinates to normalized 0-1 range")
    print(f"Image size: {args.width}×{args.height}")
    print("="*70)
    
    if args.format == 'detection':
        normalize_obb_coordinates(args.input, args.output, args.width, args.height)
    elif args.format == 'submission':
        normalize_submission_format(args.input, args.output, args.width, args.height)
    
    print("\n" + "="*70)
    print("✅ NORMALIZATION COMPLETE")
    print("="*70)
    print("📝 YOLO OBB Format Compliance:")
    print("   • center_x, center_y: normalized 0-1 ✅")
    print("   • w, h: normalized 0-1 ✅")
    print("   • theta: degrees (unchanged) ✅")
    print("="*70)
