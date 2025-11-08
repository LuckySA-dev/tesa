"""
Problem 2: Train Regression Models
Train separate models for range, azimuth, and elevation prediction
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb


class RegressionTrainer:
    """Train regression models for range, azimuth, elevation"""
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize trainer
        
        Args:
            model_type: 'random_forest' or 'xgboost'
        """
        self.model_type = model_type
        self.models = {}
        self.feature_columns = None
        self.target_columns = ['range_m', 'azimuth_deg', 'elevation_deg']
        
    def load_dataset(self, dataset_path):
        """Load training dataset"""
        print(f"📂 Loading dataset: {dataset_path}")
        self.df = pd.read_csv(dataset_path)
        
        # Define feature columns (exclude labels and frame_id)
        exclude_cols = self.target_columns + ['frame_id']
        self.feature_columns = [c for c in self.df.columns if c not in exclude_cols]
        
        print(f"   • Total samples: {len(self.df)}")
        print(f"   • Features: {len(self.feature_columns)}")
        print(f"   • Targets: {len(self.target_columns)}")
        
        return self.df
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data into train/test sets"""
        print(f"\n📊 Preparing data (test_size={test_size})...")
        
        X = self.df[self.feature_columns]
        
        self.splits = {}
        
        for target in self.target_columns:
            y = self.df[target]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            self.splits[target] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
            print(f"   • {target}: Train={len(X_train)}, Test={len(X_test)}")
        
        return self.splits
    
    def train_model(self, target):
        """Train model for specific target"""
        print(f"\n🎯 Training model for: {target}")
        
        split = self.splits[target]
        X_train = split['X_train']
        y_train = split['y_train']
        
        if self.model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        elif self.model_type == 'xgboost':
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Store model
        self.models[target] = model
        
        print(f"   ✅ Training complete")
        
        return model
    
    def evaluate_model(self, target):
        """Evaluate model performance"""
        split = self.splits[target]
        model = self.models[target]
        
        X_train = split['X_train']
        y_train = split['y_train']
        X_test = split['X_test']
        y_test = split['y_test']
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results = {
            'target': target,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        print(f"\n📊 {target} Results:")
        print(f"   • Train MAE: {train_mae:.3f}")
        print(f"   • Test MAE:  {test_mae:.3f}")
        print(f"   • Train R²:  {train_r2:.3f}")
        print(f"   • Test R²:   {test_r2:.3f}")
        
        return results
    
    def train_all(self):
        """Train and evaluate all models"""
        print(f"\n{'='*70}")
        print(f"🚀 TRAINING ALL MODELS ({self.model_type.upper()})")
        print(f"{'='*70}")
        
        results = []
        
        for target in self.target_columns:
            self.train_model(target)
            result = self.evaluate_model(target)
            results.append(result)
        
        return results
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving models to: {output_dir}/")
        
        for target, model in self.models.items():
            model_file = output_path / f'{target}_{self.model_type}.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"   ✅ Saved: {model_file.name}")
        
        # Save feature columns
        metadata = {
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }
        metadata_file = output_path / f'metadata_{self.model_type}.pkl'
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"   ✅ Saved: {metadata_file.name}")
        
        print(f"\n✅ All models saved successfully!")
    
    def get_feature_importance(self, target, top_n=10):
        """Get feature importance for a target"""
        model = self.models[target]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n🎯 Top {top_n} features for {target}:")
            for i, row in feature_imp.head(top_n).iterrows():
                print(f"   {row['feature']:20s}: {row['importance']:.4f}")
            
            return feature_imp
        else:
            print(f"   ⚠️ Model doesn't support feature importance")
            return None


# Main execution
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train regression models')
    parser.add_argument('--dataset', type=str, default='submissions/training_dataset.csv',
                       help='Training dataset path')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['random_forest', 'xgboost'],
                       help='Model type')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🎓 PROBLEM 2: TRAIN REGRESSION MODELS")
    print("="*70)
    
    # Initialize trainer
    trainer = RegressionTrainer(model_type=args.model)
    
    # Load data
    trainer.load_dataset(args.dataset)
    
    # Prepare data
    trainer.prepare_data(test_size=args.test_size)
    
    # Train all models
    results = trainer.train_all()
    
    # Feature importance
    print(f"\n{'='*70}")
    print("📈 FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    for target in trainer.target_columns:
        trainer.get_feature_importance(target, top_n=5)
    
    # Save models
    trainer.save_models()
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"📊 Overall Performance (Test MAE):")
    for result in results:
        print(f"   • {result['target']:15s}: {result['test_mae']:.3f}")
    print("="*70)
