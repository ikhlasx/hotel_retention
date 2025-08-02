# src/core/tracking_manager.py - Ultra-Optimized Tracking
import numpy as np
from collections import OrderedDict, deque
import cv2
import time
from datetime import datetime


class OptimizedFaceTracker:
    def __init__(self, track_id, bbox, embedding):
        self.track_id = track_id
        self.bbox = bbox
        self.embedding = embedding

        # MODIFIED: Increased display duration for smoother coasting.
        self.last_seen = time.time()
        self.display_bbox = bbox
        self.display_confidence = 0.0
        self.display_duration = 3.0  # Changed from 2.0 seconds

        # MODIFIED: Increased stability requirement to filter noise from the more sensitive detector.
        self.stability_frames = 0
        self.min_stability = 12  # Changed from 10
        self.state = "checking"
        self.state_timer = time.time()
        self.display_message = "Checking..."
        self.color = (128, 128, 128)

        # Recognition tracking
        self.recognition_complete = False
        self.database_checked = False
        self.visit_processed = False


class TrackingManager:
    def __init__(self, gpu_mode=True):
        self.gpu_mode = gpu_mode
        self.active_tracks = {}
        self.next_track_id = 1

        # MODIFIED: Increased timeouts for better track stability.
        self.track_timeout = 5.0  # Changed from 3.0
        self.display_timeout = 3.0  # Changed from 2.0
        self.max_distance_threshold = 9000  # Slightly more lenient matching

    def draw_tracks(self, frame, tracks):
        """Draw stabilized tracking boxes without blinking"""
        try:
            current_time = time.time()

            for track in tracks:
                # Only draw tracks that have been seen recently
                if current_time - track.last_seen > track.display_duration:
                    continue

                if not hasattr(track, 'display_bbox'):
                    continue

                # ALWAYS use display_bbox for stable rendering
                bbox = track.display_bbox
                x1, y1, x2, y2 = [int(coord) for coord in bbox]

                # Fade color for tracks that are coasting (not seen in the last second)
                if current_time - track.last_seen > 1.0:
                    color = tuple(int(c * 0.6) for c in track.color) if hasattr(track, 'color') else (80, 80, 80)
                else:
                    color = track.color if hasattr(track, 'color') else (128, 128, 128)

                # Draw stable bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw stable track info
                info = f"ID:{track.track_id}"
                if hasattr(track, 'confidence') and track.confidence > 0:
                    info += f" {track.confidence:.2f}"

                cv2.putText(frame, info, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Draw state only after it becomes stable
                if hasattr(track, 'state') and track.stability_frames >= track.min_stability:
                    cv2.putText(frame, track.state.upper(), (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Draw stable message
                if hasattr(track, 'display_message') and track.display_message and track.stability_frames >= track.min_stability:
                    lines = track.display_message.split('\n')[:2]
                    for i, line in enumerate(lines):
                        cv2.putText(frame, line, (x1, y2 + 20 + (i * 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            return frame

        except Exception as e:
            print(f"❌ Track drawing error: {e}")
            return frame

    def update_tracks(self, detections):
        """Enhanced tracking pipeline with debug output"""
        current_time = time.time()
        
        try:
            # Match detections with existing tracks
            used_detections = set()
            for track_id, tracker in list(self.active_tracks.items()):
                best_match = self._find_closest_detection_fast(tracker.bbox, detections, used_detections)

                if best_match:
                    detection, idx = best_match
                    
                    # THE CRITICAL FIX: Update both bbox and display_bbox
                    tracker.bbox = detection['bbox']
                    tracker.display_bbox = detection['bbox'] # This ensures the box doesn't revert on coasting
                    
                    tracker.last_seen = current_time
                    tracker.embedding = detection['embedding']
                    tracker.quality_score = detection.get('quality_score', 0)
                    tracker.face_area = detection.get('face_area', 0)

                    used_detections.add(idx)

            # ... (rest of the update_tracks method is unchanged) ...
            
            # Create new tracks for unmatched detections
            for idx, detection in enumerate(detections):
                if idx not in used_detections:
                    track_id = self.next_track_id
                    self.next_track_id += 1

                    new_track = OptimizedFaceTracker(track_id, detection['bbox'], detection['embedding'])
                    new_track.quality_score = detection.get('quality_score', 0)
                    new_track.face_area = detection.get('face_area', 0)

                    self.active_tracks[track_id] = new_track

            # Cleanup old tracks
            self._cleanup_old_tracks_fast(current_time)
            
            return list(self.active_tracks.values())

        except Exception as e:
            print(f"❌ Tracking update error: {e}")
            return list(self.active_tracks.values())

    def _find_closest_detection_fast(self, tracker_bbox, detections, used_indices):
        """Ultra-fast closest detection finder from reference main.py"""
        best_detection = None
        best_distance = float('inf')
        best_idx = -1

        # Pre-calculate tracker center once
        t_center = ((tracker_bbox[0] + tracker_bbox[2]) / 2, (tracker_bbox[1] + tracker_bbox[3]) / 2)

        for idx, detection in enumerate(detections):
            if idx in used_indices:
                continue

            # Simple center distance calculation
            d_bbox = detection['bbox']
            d_center = ((d_bbox[0] + d_bbox[2]) / 2, (d_bbox[1] + d_bbox[3]) / 2)

            # Fast distance calculation (avoid sqrt for performance)
            distance_sq = (t_center[0] - d_center[0]) ** 2 + (t_center[1] - d_center[1]) ** 2

            if distance_sq < best_distance and distance_sq < self.max_distance_threshold:
                best_distance = distance_sq
                best_detection = detection
                best_idx = idx

        return (best_detection, best_idx) if best_detection else None

    def _cleanup_old_tracks_fast(self, current_time):
        """Ultra-fast track cleanup from reference main.py"""
        tracks_to_remove = [
            track_id for track_id, tracker in self.active_tracks.items()
            if current_time - tracker.last_seen > self.track_timeout
        ]

        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
        
        return tracks_to_remove

    def get_track_count(self):
        """Get number of active tracks"""
        return len(self.active_tracks)
