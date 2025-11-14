"""
Batch processing module for student attire verification.
Allows processing multiple images/videos at once.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .config import AppConfig
from .features import extract_features_from_image, extract_pose
from .model import AttireClassifier
from .verify import verify_attire_and_safety
from .db import insert_event


class BatchProcessor:
    """Process multiple images/videos in batch"""
    
    def __init__(self, config: Optional[AppConfig] = None, classifier: Optional[AttireClassifier] = None):
        self.config = config or AppConfig()
        self.classifier = classifier or AttireClassifier()
        self.results = []
    
    def process_image(self, image_path: str, student_id: Optional[str] = None, zone: str = "Batch") -> Dict[str, Any]:
        """Process a single image and return results"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "file": image_path,
                    "status": "ERROR",
                    "error": "Could not load image",
                    "student_id": student_id,
                    "zone": zone
                }
            
            # Extract features
            pose = extract_pose(image)
            features = extract_features_from_image(image, pose_landmarks=pose, bins=self.config.hist_bins)
            
            # Verify attire
            result = verify_attire_and_safety(features, self.config, self.classifier)
            
            # Log event
            event_id = insert_event({
                "student_id": student_id,
                "zone": zone,
                "status": result["status"],
                "score": result["score"],
                "label": result.get("label"),
                "details": str(result.get("details")),
                "image_path": image_path,
            })
            
            return {
                "file": image_path,
                "status": result["status"],
                "success_score": result["success_score"],
                "fail_score": result["fail_score"],
                "violations": result.get("violations", {}),
                "student_id": student_id,
                "zone": zone,
                "event_id": event_id,
                "error": None
            }
            
        except Exception as e:
            return {
                "file": image_path,
                "status": "ERROR",
                "error": str(e),
                "student_id": student_id,
                "zone": zone
            }
    
    def process_video(self, video_path: str, student_id: Optional[str] = None, zone: str = "Batch", 
                     sample_interval: int = 30) -> Dict[str, Any]:
        """Process a video file by sampling frames"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    "file": video_path,
                    "status": "ERROR",
                    "error": "Could not open video",
                    "student_id": student_id,
                    "zone": zone
                }
            
            frame_results = []
            frame_count = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Sample frames based on interval
                if frame_count % sample_interval == 0:
                    # Extract features
                    pose = extract_pose(frame)
                    features = extract_features_from_image(frame, pose_landmarks=pose, bins=self.config.hist_bins)
                    
                    # Verify attire
                    result = verify_attire_and_safety(features, self.config, self.classifier)
                    
                    frame_results.append({
                        "frame": frame_count,
                        "status": result["status"],
                        "success_score": result["success_score"],
                        "fail_score": result["fail_score"],
                        "violations": result.get("violations", {})
                    })
                    
                    processed_frames += 1
            
            cap.release()
            
            if not frame_results:
                return {
                    "file": video_path,
                    "status": "ERROR",
                    "error": "No frames processed",
                    "student_id": student_id,
                    "zone": zone
                }
            
            # Calculate overall video statistics
            total_violations = sum(len(fr["violations"].get("violations", [])) for fr in frame_results)
            avg_success = np.mean([fr["success_score"] for fr in frame_results])
            avg_fail = np.mean([fr["fail_score"] for fr in frame_results])
            
            # Determine overall video status
            if avg_success >= 0.8:
                overall_status = "PASS"
            elif avg_success >= 0.6:
                overall_status = "WARNING"
            else:
                overall_status = "FAIL"
            
            return {
                "file": video_path,
                "status": overall_status,
                "success_score": avg_success,
                "fail_score": avg_fail,
                "total_violations": total_violations,
                "frames_processed": processed_frames,
                "frame_results": frame_results,
                "student_id": student_id,
                "zone": zone,
                "error": None
            }
            
        except Exception as e:
            return {
                "file": video_path,
                "status": "ERROR",
                "error": str(e),
                "student_id": student_id,
                "zone": zone
            }
    
    def process_directory(self, directory_path: str, file_extensions: List[str] = None, 
                         max_workers: int = 4) -> List[Dict[str, Any]]:
        """Process all images/videos in a directory"""
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']
        
        directory = Path(directory_path)
        if not directory.exists():
            return [{"error": f"Directory {directory_path} does not exist"}]
        
        # Find all files with supported extensions
        files = []
        for ext in file_extensions:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"*{ext.upper()}"))
        
        if not files:
            return [{"error": f"No supported files found in {directory_path}"}]
        
        results = []
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {}
            
            for file_path in files:
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    future = executor.submit(self.process_image, str(file_path))
                elif file_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                    future = executor.submit(self.process_video, str(file_path))
                else:
                    continue
                
                future_to_file[future] = str(file_path)
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "file": file_path,
                        "status": "ERROR",
                        "error": str(e)
                    })
        
        self.results = results
        return results
    
    def generate_report(self, output_path: str = "batch_report.json") -> str:
        """Generate a detailed report of batch processing results"""
        if not self.results:
            return "No results to report"
        
        # Calculate statistics
        total_files = len(self.results)
        successful_files = len([r for r in self.results if r.get("status") != "ERROR"])
        error_files = total_files - successful_files
        
        pass_files = len([r for r in self.results if r.get("status") == "PASS"])
        warning_files = len([r for r in self.results if r.get("status") == "WARNING"])
        fail_files = len([r for r in self.results if r.get("status") == "FAIL"])
        
        avg_success = np.mean([r.get("success_score", 0) for r in self.results if r.get("success_score") is not None])
        avg_fail = np.mean([r.get("fail_score", 0) for r in self.results if r.get("fail_score") is not None])
        
        # Create report
        report = {
            "summary": {
                "total_files": total_files,
                "successful_files": successful_files,
                "error_files": error_files,
                "pass_files": pass_files,
                "warning_files": warning_files,
                "fail_files": fail_files,
                "average_success_score": float(avg_success) if not np.isnan(avg_success) else 0.0,
                "average_fail_score": float(avg_fail) if not np.isnan(avg_fail) else 0.0
            },
            "results": self.results
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return f"Report saved to {output_path}"
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of processed results"""
        if not self.results:
            return {}
        
        return {
            "total_files": len(self.results),
            "successful_files": len([r for r in self.results if r.get("status") != "ERROR"]),
            "error_files": len([r for r in self.results if r.get("status") == "ERROR"]),
            "pass_files": len([r for r in self.results if r.get("status") == "PASS"]),
            "warning_files": len([r for r in self.results if r.get("status") == "WARNING"]),
            "fail_files": len([r for r in self.results if r.get("status") == "FAIL"]),
            "compliance_rate": len([r for r in self.results if r.get("status") == "PASS"]) / max(1, len(self.results)) * 100
        }
