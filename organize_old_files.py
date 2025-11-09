"""
Organize old/unnecessary files into categorized directories
Moves files to old/ directory with proper categorization
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


class FileOrganizer:
    """Organize old files into categorized directories"""
    
    def __init__(self, base_dir: str = ".", old_dir: str = "old"):
        self.base_dir = Path(base_dir)
        self.old_dir = self.base_dir / old_dir
        
        # Define categories
        self.categories = {
            'csv_old': {
                'name': 'Old CSVs & Data Files',
                'files': [
                    'compare_results.py',
                    'ground_truth_mock.csv',
                    'problem1_bytetrack.csv',
                    'problem1_output.csv',
                    'problem1_optimized.csv',
                    'problem2_predictions.csv',
                    'p2_localization.csv',
                    'p2_localization_v2.csv',
                    'submission_normalized.csv',
                    'training_dataset.csv',
                    'training_data.csv',
                    'mock_training_data.csv',
                ]
            },
            'backups': {
                'name': 'Backup Files',
                'files': [
                    'centroid_tracker_backup.py',
                    'problem1_video_tracking_backup.py',
                    'problem2_inference_backup.py',
                    'problem3_integration_backup.py',
                ]
            },
            'experimental': {
                'name': 'Experimental & Test Scripts',
                'files': [
                    'test_external_dataset.py',
                    'test_api_integration.py',
                    'optimize_performance.py',
                    'test_bytetrack.py',
                    'check_problem1_csv.py',
                    'check_theta.py',
                    'check_tracks.py',
                    'compare_thresholds.py',
                    'compare_trackers.py',
                    'fix_obb_normalization.py',
                ]
            },
            'alternative_solutions': {
                'name': 'Alternative Solutions (Not Used)',
                'files': [
                    'centroid_tracker.py',
                    'problem1_detection.py',
                    'problem1_video_tracking.py',
                    'problem1_optimized.py',
                    'problem2_localization.py',
                    'problem2_train.py',
                    'problem3_tracking.py',
                ]
            },
            'analysis': {
                'name': 'Analysis & Optimization Scripts',
                'files': [
                    'analyze_tracking.py',
                    'auto_tune_confidence.py',
                    'final_threshold_analysis.py',
                    'validate_csv.py',
                    'validate_problem2.py',
                ]
            },
            'misc': {
                'name': 'Miscellaneous',
                'files': [
                    'kaggle.json',
                    'performance_dashboard.png',
                    'test_kaggle_image.png',
                    'make_sample_video.py',
                ]
            },
            'logs_old': {
                'name': 'Old Logs',
                'directories': [
                    'logs',
                ]
            }
        }
        
        # Files to keep (not move)
        self.keep_files = {
            # Production
            'problem1_competition.py',
            'problem1_raspberry_pi.py',
            'problem2_inference.py',
            'problem2_dataset.py',
            'problem3_integration.py',
            
            # Utilities
            'byte_track_wrapper.py',
            'fix_problem2_format.py',
            'check_compliance.py',
            'validate_submission.py',
            'kaggle_integration.py',
            'raspberry_pi_deployment.py',
            'cleanup_project.py',
            'organize_old_files.py',
            'api_client.py',
            'utils.py',
            'config.py',
            'visualize.py',
            
            # Dashboards
            'dashboard_performance.py',
            'dashboard_realtime.py',
            'dashboard_streamlit.py',
            
            # Raspberry Pi
            'raspberry_pi_deployment.py',
            'setup_raspberry_pi.sh',
            
            # Documentation
            'ORGANIZATION.md',
            'OPTIMIZATION_SUMMARY.txt',
            'EXTERNAL_DATASET_SUMMARY.txt',
            'requirements.txt',
            
            # Models
            'yolov8n-obb.pt',
            'yolov8m-obb.pt',
        }
    
    def create_directory_structure(self):
        """Create old/ directory structure"""
        print("="*70)
        print("📁 CREATING DIRECTORY STRUCTURE")
        print("="*70)
        
        # Create main old directory
        self.old_dir.mkdir(exist_ok=True)
        print(f"\n✅ Created: {self.old_dir}/")
        
        # Create category directories
        for category_key, category_info in self.categories.items():
            category_dir = self.old_dir / category_key
            category_dir.mkdir(exist_ok=True)
            print(f"   ✅ {category_key}/ - {category_info['name']}")
        
        # Create README
        self._create_readme()
        
        print("\n✅ Directory structure created!")
    
    def _create_readme(self):
        """Create README in old/ directory"""
        readme_content = f"""# Old Files Archive
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This directory contains files that are no longer needed for production but kept for reference.

## Categories:

"""
        for category_key, category_info in self.categories.items():
            readme_content += f"### {category_key}/\n"
            readme_content += f"{category_info['name']}\n\n"
        
        readme_content += """
## Note:
- These files are archived but not deleted
- Can be restored if needed
- Safe to delete this entire directory if storage is needed

## Production Files (Kept in Root):
- problem1_competition.py - Main detection system
- problem1_raspberry_pi.py - Pi-optimized version
- problem2_inference.py - Localization system
- problem3_integration.py - Complete pipeline
- All dashboard files
- All documentation in reports/
"""
        
        readme_path = self.old_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"   ✅ Created: README.md")
    
    def move_files(self, dry_run: bool = True):
        """Move files to categorized directories"""
        print("\n" + "="*70)
        if dry_run:
            print("🔍 DRY RUN - Showing what will be moved")
        else:
            print("📦 MOVING FILES")
        print("="*70)
        
        total_files = 0
        total_size = 0
        moved_count = 0
        
        for category_key, category_info in self.categories.items():
            category_dir = self.old_dir / category_key
            
            print(f"\n📁 {category_info['name']}:")
            
            # Move files
            if 'files' in category_info:
                for filename in category_info['files']:
                    source = self.base_dir / filename
                    
                    if source.exists() and source.is_file():
                        size = source.stat().st_size
                        total_files += 1
                        total_size += size
                        
                        if dry_run:
                            print(f"   → {filename} ({self._format_size(size)})")
                        else:
                            dest = category_dir / filename
                            try:
                                shutil.move(str(source), str(dest))
                                print(f"   ✅ Moved: {filename}")
                                moved_count += 1
                            except Exception as e:
                                print(f"   ❌ Error moving {filename}: {e}")
            
            # Move directories
            if 'directories' in category_info:
                for dirname in category_info['directories']:
                    source = self.base_dir / dirname
                    
                    if source.exists() and source.is_dir():
                        # Calculate directory size
                        dir_size = sum(f.stat().st_size for f in source.rglob('*') if f.is_file())
                        total_files += 1
                        total_size += dir_size
                        
                        if dry_run:
                            print(f"   → {dirname}/ ({self._format_size(dir_size)})")
                        else:
                            dest = category_dir / dirname
                            try:
                                if dest.exists():
                                    shutil.rmtree(dest)
                                shutil.move(str(source), str(dest))
                                print(f"   ✅ Moved: {dirname}/")
                                moved_count += 1
                            except Exception as e:
                                print(f"   ❌ Error moving {dirname}: {e}")
        
        # Summary
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        print(f"   • Total items: {total_files}")
        print(f"   • Total size: {self._format_size(total_size)}")
        
        if not dry_run:
            print(f"   • Successfully moved: {moved_count}")
            print(f"\n✅ Files organized into: {self.old_dir}/")
        else:
            print(f"\n💡 Run with --move to actually move files")
    
    def _format_size(self, size: int) -> str:
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def verify_production_files(self):
        """Verify that production files are still in place"""
        print("\n" + "="*70)
        print("✅ VERIFYING PRODUCTION FILES")
        print("="*70)
        
        missing = []
        present = []
        
        for filename in sorted(self.keep_files):
            filepath = self.base_dir / filename
            if filepath.exists():
                present.append(filename)
            else:
                missing.append(filename)
        
        print(f"\n✅ Production files present: {len(present)}/{len(self.keep_files)}")
        
        if missing:
            print(f"\n⚠️  Missing files:")
            for filename in missing:
                print(f"   ❌ {filename}")
        
        # Check critical directories
        critical_dirs = ['submissions', 'reports', 'models', 'videos']
        print(f"\n📁 Critical directories:")
        for dirname in critical_dirs:
            dirpath = self.base_dir / dirname
            if dirpath.exists():
                files = list(dirpath.rglob('*'))
                print(f"   ✅ {dirname}/ ({len(files)} items)")
            else:
                print(f"   ⚠️  {dirname}/ (not found)")
        
        print("\n" + "="*70)
    
    def create_backup(self):
        """Create a backup list before moving"""
        backup_file = self.base_dir / "backup_list.txt"
        
        print("\n📝 Creating backup list...")
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(f"Backup List - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            for category_key, category_info in self.categories.items():
                f.write(f"\n{category_info['name']}:\n")
                f.write("-"*50 + "\n")
                
                if 'files' in category_info:
                    for filename in category_info['files']:
                        source = self.base_dir / filename
                        if source.exists():
                            f.write(f"  {filename}\n")
                
                if 'directories' in category_info:
                    for dirname in category_info['directories']:
                        source = self.base_dir / dirname
                        if source.exists():
                            f.write(f"  {dirname}/\n")
        
        print(f"✅ Backup list saved: {backup_file}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Organize old files into categorized directories'
    )
    parser.add_argument('--create', action='store_true',
                       help='Create directory structure')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what will be moved (default)')
    parser.add_argument('--move', action='store_true',
                       help='Actually move files')
    parser.add_argument('--verify', action='store_true',
                       help='Verify production files')
    parser.add_argument('--backup', action='store_true',
                       help='Create backup list')
    parser.add_argument('--all', action='store_true',
                       help='Do everything (create, backup, move, verify)')
    
    args = parser.parse_args()
    
    organizer = FileOrganizer()
    
    # Default to dry-run if no args
    if not any(vars(args).values()):
        args.dry_run = True
    
    if args.all:
        print("\n🚀 FULL ORGANIZATION PROCESS")
        print("="*70)
        
        # Step 1: Create directories
        organizer.create_directory_structure()
        
        # Step 2: Create backup list
        organizer.create_backup()
        
        # Step 3: Show what will be moved
        print("\n")
        organizer.move_files(dry_run=True)
        
        # Step 4: Ask for confirmation
        print("\n" + "="*70)
        response = input("\n⚠️  Proceed with moving files? (yes/no): ")
        
        if response.lower() in ['yes', 'y']:
            organizer.move_files(dry_run=False)
            organizer.verify_production_files()
            
            print("\n" + "="*70)
            print("🎉 ORGANIZATION COMPLETE!")
            print("="*70)
            print(f"\n✅ Old files moved to: old/")
            print(f"✅ Backup list: backup_list.txt")
            print(f"✅ Production files verified")
            print(f"\n📂 Directory structure:")
            print(f"   old/")
            for category_key, category_info in organizer.categories.items():
                print(f"      ├── {category_key}/ - {category_info['name']}")
            print(f"      └── README.md")
            print("\n" + "="*70)
        else:
            print("\n❌ Cancelled. No files were moved.")
    
    else:
        if args.create:
            organizer.create_directory_structure()
        
        if args.backup:
            organizer.create_backup()
        
        if args.dry_run:
            organizer.move_files(dry_run=True)
        
        if args.move:
            organizer.move_files(dry_run=False)
        
        if args.verify:
            organizer.verify_production_files()


if __name__ == '__main__':
    main()
