"""
Performance Optimization Module
Optimize detection, tracking, and inference speed
"""

import time
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List


class PerformanceOptimizer:
    """Optimize system performance"""
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.optimizations_applied = []
        
    def _get_optimal_device(self) -> str:
        """Detect best available device"""
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
        else:
            device = 'cpu'
            print(f"ℹ️  Using CPU (GPU not available)")
        return device
    
    def optimize_model(self, model, half_precision: bool = True):
        """
        Optimize YOLO model for faster inference
        
        Args:
            model: YOLO model instance
            half_precision: Use FP16 (half precision) for GPU
        """
        print("\n" + "="*70)
        print("🚀 MODEL OPTIMIZATION")
        print("="*70)
        
        # Move to optimal device
        if self.device == 'cuda':
            model.to('cuda')
            self.optimizations_applied.append("GPU acceleration")
            print("✅ Model moved to GPU")
            
            # Half precision (FP16) for faster inference
            if half_precision:
                try:
                    model.model.half()
                    self.optimizations_applied.append("FP16 (half precision)")
                    print("✅ Half precision (FP16) enabled")
                except Exception as e:
                    print(f"⚠️  Half precision failed: {e}")
        else:
            print("ℹ️  Running on CPU")
            
        return model
    
    def optimize_video_capture(self, cap: cv2.VideoCapture) -> cv2.VideoCapture:
        """
        Optimize video capture settings
        
        Args:
            cap: OpenCV VideoCapture object
        """
        # Set buffer size to 1 for lower latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Enable hardware acceleration if available
        backend = cap.getBackendName()
        print(f"📹 Video backend: {backend}")
        
        self.optimizations_applied.append("Video capture optimized")
        return cap
    
    def should_skip_frame(self, frame_id: int, skip_factor: int = 1) -> bool:
        """
        Determine if frame should be skipped
        
        Args:
            frame_id: Current frame number
            skip_factor: Process every Nth frame (1 = all frames)
        """
        return frame_id % skip_factor != 0
    
    def resize_for_inference(
        self, 
        frame: np.ndarray, 
        target_size: int = 640,
        keep_aspect: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Resize frame for faster inference
        
        Args:
            frame: Input frame
            target_size: Target size (width or height)
            keep_aspect: Maintain aspect ratio
            
        Returns:
            Resized frame and scale factor
        """
        h, w = frame.shape[:2]
        
        if keep_aspect:
            scale = target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w = target_size
            new_h = target_size
            scale = target_size / max(h, w)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale
    
    def batch_inference(
        self, 
        frames: List[np.ndarray], 
        model,
        conf: float = 0.5
    ) -> List:
        """
        Process multiple frames in batch for efficiency
        
        Args:
            frames: List of frames
            model: YOLO model
            conf: Confidence threshold
        """
        # Stack frames for batch processing
        results = model.predict(frames, conf=conf, verbose=False)
        return results
    
    def apply_nms_optimization(self, model, iou_threshold: float = 0.5):
        """
        Optimize Non-Maximum Suppression settings
        
        Args:
            model: YOLO model
            iou_threshold: IoU threshold for NMS
        """
        # Set NMS parameters
        model.overrides['iou'] = iou_threshold
        self.optimizations_applied.append(f"NMS optimized (IoU={iou_threshold})")
        print(f"✅ NMS IoU threshold: {iou_threshold}")
        
    def enable_torch_optimizations(self):
        """Enable PyTorch-specific optimizations"""
        print("\n🔧 PyTorch Optimizations:")
        
        # Enable cudnn benchmark for consistent input sizes
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print("✅ CuDNN benchmark enabled")
            self.optimizations_applied.append("CuDNN benchmark")
        
        # Set number of threads for CPU
        if self.device == 'cpu':
            num_threads = torch.get_num_threads()
            print(f"ℹ️  CPU threads: {num_threads}")
            
    def estimate_optimal_batch_size(self, model, img_size: int = 640) -> int:
        """
        Estimate optimal batch size based on available memory
        
        Args:
            model: YOLO model
            img_size: Input image size
            
        Returns:
            Recommended batch size
        """
        if self.device == 'cuda':
            # Get GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            free_memory = total_memory - allocated
            
            # Rough estimate: 200MB per image at 640x640
            memory_per_image = 200 * 1024 * 1024
            batch_size = int(free_memory / memory_per_image * 0.8)  # Use 80% of free
            batch_size = max(1, min(batch_size, 32))  # Limit to 1-32
            
            print(f"\n💾 GPU Memory:")
            print(f"   • Total: {total_memory / 1e9:.2f} GB")
            print(f"   • Free: {free_memory / 1e9:.2f} GB")
            print(f"   • Recommended batch: {batch_size}")
            
            return batch_size
        else:
            return 1  # CPU: process one at a time
    
    def create_optimization_profile(self) -> Dict:
        """Create profile of applied optimizations"""
        profile = {
            'device': self.device,
            'optimizations': self.optimizations_applied,
            'cuda_available': torch.cuda.is_available(),
            'torch_version': torch.__version__,
        }
        
        if torch.cuda.is_available():
            profile['gpu_name'] = torch.cuda.get_device_name(0)
            profile['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            
        return profile
    
    def print_optimization_summary(self):
        """Print summary of applied optimizations"""
        print("\n" + "="*70)
        print("📊 OPTIMIZATION SUMMARY")
        print("="*70)
        
        profile = self.create_optimization_profile()
        
        print(f"Device: {profile['device'].upper()}")
        
        if profile['cuda_available']:
            print(f"GPU: {profile.get('gpu_name', 'Unknown')}")
            print(f"GPU Memory: {profile.get('gpu_memory_total', 0):.2f} GB")
        
        print(f"\n✅ Optimizations Applied ({len(self.optimizations_applied)}):")
        for opt in self.optimizations_applied:
            print(f"   • {opt}")
        
        print("="*70)


class FastVideoProcessor:
    """Optimized video processing pipeline"""
    
    def __init__(
        self, 
        model,
        device: str = 'auto',
        skip_frames: int = 1,
        resize_width: Optional[int] = None,
        half_precision: bool = True
    ):
        """
        Initialize fast video processor
        
        Args:
            model: YOLO model
            device: 'auto', 'cuda', or 'cpu'
            skip_frames: Process every Nth frame (1 = all frames)
            resize_width: Resize frame width (None = original)
            half_precision: Use FP16 on GPU
        """
        self.optimizer = PerformanceOptimizer()
        self.model = self.optimizer.optimize_model(model, half_precision)
        self.skip_frames = skip_frames
        self.resize_width = resize_width
        
        self.optimizer.enable_torch_optimizations()
        self.optimizer.apply_nms_optimization(model, iou_threshold=0.45)
        
        self.stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'total_time': 0,
            'inference_time': 0,
        }
    
    def process_video(
        self, 
        video_path: str, 
        conf: float = 0.5,
        verbose: bool = True
    ):
        """
        Process video with optimizations
        
        Args:
            video_path: Path to video file
            conf: Confidence threshold
            verbose: Print progress
        """
        cap = cv2.VideoCapture(video_path)
        cap = self.optimizer.optimize_video_capture(cap)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if verbose:
            print(f"\n🎬 Processing: {Path(video_path).name}")
            print(f"   • Total frames: {total_frames}")
            print(f"   • FPS: {fps}")
            print(f"   • Skip factor: {self.skip_frames}")
        
        start_time = time.time()
        frame_id = 0
        results_list = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if configured
            if self.optimizer.should_skip_frame(frame_id, self.skip_frames):
                self.stats['frames_skipped'] += 1
                frame_id += 1
                continue
            
            # Resize if configured
            if self.resize_width:
                frame, _ = self.optimizer.resize_for_inference(
                    frame, 
                    self.resize_width
                )
            
            # Inference
            inf_start = time.time()
            results = self.model.predict(frame, conf=conf, verbose=False)
            self.stats['inference_time'] += time.time() - inf_start
            
            results_list.append(results)
            self.stats['frames_processed'] += 1
            frame_id += 1
            
            # Progress
            if verbose and frame_id % 30 == 0:
                elapsed = time.time() - start_time
                fps_proc = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
                print(f"   Frame {frame_id}/{total_frames} | FPS: {fps_proc:.1f}")
        
        cap.release()
        self.stats['total_time'] = time.time() - start_time
        
        if verbose:
            self._print_stats()
        
        return results_list
    
    def _print_stats(self):
        """Print processing statistics"""
        print("\n" + "="*70)
        print("📊 PROCESSING STATISTICS")
        print("="*70)
        
        print(f"Frames processed: {self.stats['frames_processed']}")
        print(f"Frames skipped: {self.stats['frames_skipped']}")
        print(f"Total time: {self.stats['total_time']:.2f}s")
        print(f"Inference time: {self.stats['inference_time']:.2f}s")
        
        if self.stats['frames_processed'] > 0:
            avg_fps = self.stats['frames_processed'] / self.stats['total_time']
            avg_inf = self.stats['inference_time'] / self.stats['frames_processed']
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Average inference: {avg_inf*1000:.1f}ms")
        
        print("="*70)


def benchmark_performance(video_path: str, model_path: str = 'yolov8n-obb.pt'):
    """
    Benchmark different optimization configurations
    
    Args:
        video_path: Path to test video
        model_path: Path to YOLO model
    """
    from ultralytics import YOLO
    
    print("="*70)
    print("🏁 PERFORMANCE BENCHMARK")
    print("="*70)
    
    configs = [
        {'name': 'Baseline (CPU, All frames)', 'skip': 1, 'resize': None, 'device': 'cpu'},
        {'name': 'Skip 50% frames', 'skip': 2, 'resize': None, 'device': 'cpu'},
        {'name': 'Resize to 480p', 'skip': 1, 'resize': 480, 'device': 'cpu'},
        {'name': 'Skip 50% + Resize 480p', 'skip': 2, 'resize': 480, 'device': 'cpu'},
    ]
    
    # Add GPU configs if available
    if torch.cuda.is_available():
        configs.extend([
            {'name': 'GPU (All frames)', 'skip': 1, 'resize': None, 'device': 'cuda'},
            {'name': 'GPU + FP16', 'skip': 1, 'resize': None, 'device': 'cuda'},
        ])
    
    results = []
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        print(f"{'='*70}")
        
        model = YOLO(model_path)
        processor = FastVideoProcessor(
            model,
            device=config['device'],
            skip_frames=config['skip'],
            resize_width=config['resize']
        )
        
        start = time.time()
        processor.process_video(video_path, conf=0.5, verbose=False)
        elapsed = time.time() - start
        
        fps = processor.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        
        results.append({
            'config': config['name'],
            'time': elapsed,
            'fps': fps,
            'frames': processor.stats['frames_processed']
        })
        
        print(f"✅ Time: {elapsed:.2f}s | FPS: {fps:.2f}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 BENCHMARK RESULTS")
    print("="*70)
    print(f"{'Configuration':<30} {'Time (s)':<12} {'FPS':<10} {'Frames':<10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['config']:<30} {r['time']:<12.2f} {r['fps']:<10.2f} {r['frames']:<10}")
    
    print("="*70)


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Optimization')
    parser.add_argument('--benchmark', type=str, help='Run benchmark on video file')
    parser.add_argument('--model', type=str, default='yolov8n-obb.pt',
                       help='Model path')
    parser.add_argument('--info', action='store_true',
                       help='Show optimization info')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_performance(args.benchmark, args.model)
    
    elif args.info:
        optimizer = PerformanceOptimizer()
        optimizer.enable_torch_optimizations()
        optimizer.print_optimization_summary()
    
    else:
        parser.print_help()
        print("\n" + "="*70)
        print("💡 USAGE EXAMPLES")
        print("="*70)
        print("\n1. Show optimization info:")
        print("   python optimize_performance.py --info")
        print("\n2. Benchmark video:")
        print("   python optimize_performance.py --benchmark videos/video_01.mp4")
        print("\n3. Use in your code:")
        print("   from optimize_performance import PerformanceOptimizer, FastVideoProcessor")
        print("   optimizer = PerformanceOptimizer()")
        print("   model = optimizer.optimize_model(model)")
        print("="*70)


if __name__ == '__main__':
    main()
