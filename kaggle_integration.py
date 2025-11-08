"""
Kaggle Dataset Integration
Instructions and utilities for downloading and testing Kaggle drone datasets
"""

import os
import sys
import json
from pathlib import Path


def setup_kaggle_credentials():
    """
    Setup Kaggle API credentials
    
    Instructions:
    1. Go to https://www.kaggle.com/settings/account
    2. Click "Create New API Token"
    3. Download kaggle.json
    4. Place it in: C:\\Users\\User\\.kaggle\\kaggle.json (Windows)
    """
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    print("="*70)
    print("🔑 KAGGLE API SETUP")
    print("="*70)
    
    if kaggle_json.exists():
        print(f"✅ Kaggle credentials found: {kaggle_json}")
        
        # Verify permissions (Windows equivalent)
        import stat
        current_mode = kaggle_json.stat().st_mode
        print(f"   • File permissions: {oct(current_mode)}")
        
        return True
    else:
        print("❌ Kaggle credentials not found!")
        print("\n📋 Setup Instructions:")
        print("   1. Go to: https://www.kaggle.com/settings/account")
        print("   2. Scroll to 'API' section")
        print("   3. Click 'Create New API Token'")
        print("   4. Download kaggle.json")
        print(f"   5. Create directory: {kaggle_dir}")
        print(f"   6. Move kaggle.json to: {kaggle_json}")
        print()
        
        # Create directory if not exists
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {kaggle_dir}")
        print(f"   ⚠️  Now place kaggle.json in this directory!")
        
        return False


def list_popular_drone_datasets():
    """List popular drone detection datasets on Kaggle"""
    datasets = [
        {
            'name': 'Drone Detection Dataset',
            'kaggle_id': 'dasmehdixtr/drone-dataset-uav',
            'description': 'UAV drone images for object detection',
            'size': '~500MB',
            'format': 'Images + YOLO annotations'
        },
        {
            'name': 'Anti Drone Dataset',
            'kaggle_id': 'soumenksarker/anti-drones',
            'description': 'Thermal and RGB drone images',
            'size': '~2GB',
            'format': 'Images + XML annotations'
        },
        {
            'name': 'Drone vs Bird Dataset',
            'kaggle_id': 'kmader/drone-vs-bird',
            'description': 'Classification dataset for drones and birds',
            'size': '~100MB',
            'format': 'Images'
        },
        {
            'name': 'Aerial Vehicle Detection',
            'kaggle_id': 'kmader/aerial-vehicles',
            'description': 'Aerial view vehicle detection',
            'size': '~300MB',
            'format': 'Images + annotations'
        }
    ]
    
    print("="*70)
    print("📦 POPULAR DRONE DATASETS ON KAGGLE")
    print("="*70)
    
    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds['name']}")
        print(f"   • Kaggle ID: {ds['kaggle_id']}")
        print(f"   • Description: {ds['description']}")
        print(f"   • Size: {ds['size']}")
        print(f"   • Format: {ds['format']}")
        print(f"   • Download: kaggle datasets download -d {ds['kaggle_id']}")
    
    print("\n" + "="*70)
    
    return datasets


def download_kaggle_dataset(dataset_id, output_dir='external_data'):
    """
    Download dataset from Kaggle
    
    Args:
        dataset_id: Kaggle dataset ID (e.g., 'username/dataset-name')
        output_dir: Where to save the downloaded data
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*70)
    print(f"📥 DOWNLOADING: {dataset_id}")
    print("="*70)
    
    try:
        # Import Kaggle API
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Authenticate
        api = KaggleApi()
        api.authenticate()
        print("✅ Kaggle API authenticated")
        
        # Download dataset
        print(f"\n📦 Downloading to: {output_path.absolute()}")
        api.dataset_download_files(
            dataset_id,
            path=str(output_path),
            unzip=True
        )
        
        print(f"\n✅ Download complete!")
        
        # List downloaded files
        files = list(output_path.iterdir())
        print(f"\n📁 Downloaded files ({len(files)}):")
        for f in files[:10]:  # Show first 10
            print(f"   • {f.name}")
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more")
        
        return True
        
    except ImportError:
        print("❌ Kaggle API not installed!")
        print("   Install with: pip install kaggle")
        return False
    
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def prepare_dataset_for_training(dataset_dir, output_format='yolo'):
    """
    Prepare downloaded dataset for training/testing
    
    Args:
        dataset_dir: Directory containing downloaded dataset
        output_format: Output format (yolo, coco, etc.)
    """
    dataset_path = Path(dataset_dir)
    
    print("="*70)
    print(f"🔧 PREPARING DATASET: {dataset_path.name}")
    print("="*70)
    
    # Check dataset structure
    print("\n📁 Analyzing dataset structure...")
    
    # Find image files
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in image_exts:
        images.extend(dataset_path.rglob(f'*{ext}'))
    
    print(f"   • Images found: {len(images)}")
    
    # Find annotation files
    annotation_exts = ['.txt', '.xml', '.json']
    annotations = []
    for ext in annotation_exts:
        annotations.extend(dataset_path.rglob(f'*{ext}'))
    
    print(f"   • Annotations found: {len(annotations)}")
    
    # Check for videos
    video_exts = ['.mp4', '.avi', '.mov']
    videos = []
    for ext in video_exts:
        videos.extend(dataset_path.rglob(f'*{ext}'))
    
    print(f"   • Videos found: {len(videos)}")
    
    # Summary
    print("\n📊 Dataset Summary:")
    if images:
        print(f"   ✅ Can use for training/fine-tuning")
    if videos:
        print(f"   ✅ Can test pipeline directly")
    if not images and not videos:
        print(f"   ⚠️  No images or videos found")
    
    print("\n" + "="*70)
    
    return {
        'images': len(images),
        'annotations': len(annotations),
        'videos': len(videos),
        'video_files': videos
    }


def test_with_kaggle_videos(dataset_dir):
    """Test pipeline with videos from Kaggle dataset"""
    info = prepare_dataset_for_training(dataset_dir)
    
    if info['videos'] == 0:
        print("\n⚠️  No videos found in dataset.")
        print("   💡 Try creating test videos with synthetic data instead.")
        return
    
    print("\n🧪 Testing pipeline with Kaggle videos...")
    
    from test_external_dataset import ExternalDatasetTester
    tester = ExternalDatasetTester()
    
    for video_path in info['video_files'][:3]:  # Test first 3 videos
        prefix = f"kaggle_{video_path.stem}"
        print(f"\n{'='*70}")
        print(f"Testing: {video_path.name}")
        print(f"{'='*70}")
        
        success = tester.test_video(video_path, output_prefix=prefix)
        
        if not success:
            print(f"   ⚠️  Testing failed for {video_path.name}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kaggle Dataset Integration')
    parser.add_argument('--setup', action='store_true',
                       help='Setup Kaggle API credentials')
    parser.add_argument('--list', action='store_true',
                       help='List popular drone datasets')
    parser.add_argument('--download', type=str,
                       help='Download dataset by Kaggle ID')
    parser.add_argument('--prepare', type=str,
                       help='Prepare dataset directory for testing')
    parser.add_argument('--test', type=str,
                       help='Test pipeline with videos from dataset directory')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_kaggle_credentials()
    
    if args.list:
        list_popular_drone_datasets()
    
    if args.download:
        if not setup_kaggle_credentials():
            print("\n⚠️  Please setup Kaggle credentials first!")
            return
        download_kaggle_dataset(args.download)
    
    if args.prepare:
        prepare_dataset_for_training(args.prepare)
    
    if args.test:
        test_with_kaggle_videos(args.test)
    
    if not any([args.setup, args.list, args.download, args.prepare, args.test]):
        parser.print_help()
        print("\n" + "="*70)
        print("💡 QUICK START GUIDE")
        print("="*70)
        print("\n1. Setup Kaggle credentials:")
        print("   python kaggle_integration.py --setup")
        print("\n2. List available datasets:")
        print("   python kaggle_integration.py --list")
        print("\n3. Download a dataset:")
        print("   python kaggle_integration.py --download dasmehdixtr/drone-dataset-uav")
        print("\n4. Prepare dataset:")
        print("   python kaggle_integration.py --prepare external_data/drone-dataset-uav")
        print("\n5. Test with videos (if available):")
        print("   python kaggle_integration.py --test external_data/drone-dataset-uav")
        print("\n" + "="*70)


if __name__ == '__main__':
    main()
