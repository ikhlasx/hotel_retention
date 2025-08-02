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

        # Add persistence for stable display
        self.last_seen = time.time()
        self.display_bbox = bbox  # Keep last known bbox for display
        self.display_confidence = 0.0
        self.display_duration = 2.0  # Keep showing for 2 seconds after loss

        # Stability improvements
        self.stability_frames = 0
        self.min_stability = 10  # Increased for more stable detection
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

        # Improved stability settings
        self.track_timeout = 3.0  # Longer timeout for stability
        self.display_timeout = 2.0  # Keep showing boxes for 2 seconds
        self.max_distance_threshold = 8000  # More restrictive matching

    def draw_tracks(self, frame, tracks):
        """Draw stabilized tracking boxes without blinking"""
        try:
            current_time = time.time()

            for track in tracks:
                if not hasattr(track, 'bbox'):
                    continue

                # Use display_bbox for stable rendering
                bbox = track.display_bbox if hasattr(track, 'display_bbox') else track.bbox
                x1, y1, x2, y2 = [int(coord) for coord in bbox]

                # Check if track should still be displayed
                time_since_seen = current_time - track.last_seen
                if time_since_seen > track.display_duration:
                    continue  # Don't draw expired tracks

                # Choose color based on state and age
                if time_since_seen > 1.0:
                    # Fade color for tracks not recently seen
                    color = tuple(int(c * 0.5) for c in track.color) if hasattr(track, 'color') else (64, 64, 64)
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

                # Draw state with stability
                if hasattr(track, 'state') and track.stability_frames >= track.min_stability:
                    cv2.putText(frame, track.state.upper(), (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Draw stable message
                if hasattr(track,
                           'display_message') and track.display_message and track.stability_frames >= track.min_stability:
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
        current_track_ids = set()

        try:
            print(f"🔄 Tracking update: {len(detections)} detections, {len(self.active_tracks)} active tracks")

            # Match detections with existing tracks
            used_detections = set()

            for track_id, tracker in list(self.active_tracks.items()):
                best_match = self._find_closest_detection_fast(tracker.bbox, detections, used_detections)

                if best_match:
                    detection, idx = best_match
                    tracker.bbox = detection['bbox']
                    tracker.last_seen = current_time
                    tracker.embedding = detection['embedding']

                    # Store detection quality information
                    tracker.quality_score = detection.get('quality_score', 0)
                    tracker.face_area = detection.get('face_area', 0)

                    used_detections.add(idx)
                    current_track_ids.add(track_id)

                    print(f"  ✅ Updated track {track_id} with detection {idx}")

            # Create new tracks for unmatched detections
            for idx, detection in enumerate(detections):
                if idx not in used_detections:
                    track_id = self.next_track_id
                    self.next_track_id += 1

                    new_track = OptimizedFaceTracker(track_id, detection['bbox'], detection['embedding'])
                    new_track.quality_score = detection.get('quality_score', 0)
                    new_track.face_area = detection.get('face_area', 0)

                    self.active_tracks[track_id] = new_track
                    current_track_ids.add(track_id)

                    print(f"  🆕 Created new track {track_id} from detection {idx}")

            # Cleanup old tracks
            removed_tracks = self._cleanup_old_tracks_fast(current_time)
            if removed_tracks:
                print(f"  🗑️ Removed {removed_tracks} old tracks")

            active_tracks_list = list(self.active_tracks.values())
            print(f"  📊 Final result: {len(active_tracks_list)} active tracks")

            return active_tracks_list

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

    def get_track_count(self):
        """Get number of active tracks"""
        return len(self.active_tracks)

    def draw_tracks(self, frame, tracks):
        """Draw ultra-optimized tracking boxes"""
        try:
            for track in tracks:
                if not hasattr(track, 'bbox'):
                    continue

                bbox = track.bbox
                x1, y1, x2, y2 = [int(coord) for coord in bbox]

                # Choose color based on state
                color = track.color if hasattr(track, 'color') else (128, 128, 128)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw minimal info for performance
                info = f"ID:{track.track_id}"
                if hasattr(track, 'confidence') and track.confidence > 0:
                    info += f" {track.confidence:.2f}"

                cv2.putText(frame, info, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Draw state
                if hasattr(track, 'state'):
                    cv2.putText(frame, track.state.upper(), (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Draw message if available
                if hasattr(track, 'display_message') and track.display_message:
                    lines = track.display_message.split('\n')[:2]  # Max 2 lines
                    for i, line in enumerate(lines):
                        cv2.putText(frame, line, (x1, y2 + 20 + (i * 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            return frame

        except Exception as e:
            print(f"Track drawing error: {e}")
            return frame
