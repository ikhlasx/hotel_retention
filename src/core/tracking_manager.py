# src/core/tracking_manager.py - Ultra-Optimized Tracking
import numpy as np
from collections import OrderedDict, deque
import cv2
import time
from datetime import datetime
from .deepsort_tracker import DeepSort, Detection


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
        self.message_time = time.time()
        self.color = (128, 128, 128)

        # Recognition tracking
        self.recognition_complete = False
        self.database_checked = False
        self.visit_processed = False
        # Count consecutive failed identification attempts
        self.fail_count = 0

    def set_message(self, message):
        self.display_message = message
        self.message_time = time.time()


class TrackingManager:
    def __init__(self, gpu_mode=True):
        self.gpu_mode = gpu_mode
        self.active_tracks = {}

        # DeepSort tracker instance
        self.deepsort = DeepSort()

        # Maximum allowed failed identification attempts before dropping a track
        self.max_fail_count = 3

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

                # Draw stable message within display duration
                if (
                    hasattr(track, 'display_message')
                    and track.display_message
                    and track.stability_frames >= track.min_stability
                ):
                    if current_time - getattr(track, 'message_time', 0) < track.display_duration:
                        lines = track.display_message.split('\n')[:2]
                        for i, line in enumerate(lines):
                            cv2.putText(
                                frame,
                                line,
                                (x1, y2 + 20 + (i * 15)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                color,
                                1,
                            )
                    else:
                        track.display_message = None

            return frame

        except Exception as e:
            print(f"❌ Track drawing error: {e}")
            return frame

    def update_tracks(self, detections):
        """Update tracking state using DeepSort."""
        current_time = time.time()

        try:
            ds_detections = [Detection(d['bbox'], d['embedding']) for d in detections]
            ds_tracks = self.deepsort.update(ds_detections)

            updated_tracks = []
            active_ids = set()
            for track_id, bbox, embedding in ds_tracks:
                active_ids.add(track_id)
                if track_id in self.active_tracks:
                    tracker = self.active_tracks[track_id]
                    tracker.bbox = bbox
                    tracker.display_bbox = bbox
                    tracker.last_seen = current_time
                    tracker.embedding = embedding
                else:
                    tracker = OptimizedFaceTracker(track_id, bbox, embedding)
                    self.active_tracks[track_id] = tracker
                updated_tracks.append(tracker)

            # Remove stale tracks
            for tid in list(self.active_tracks.keys()):
                if tid not in active_ids:
                    del self.active_tracks[tid]

            return updated_tracks

        except Exception as e:
            print(f"❌ Tracking update error: {e}")
            return list(self.active_tracks.values())

    def get_track_count(self):
        """Get number of active tracks"""
        return len(self.active_tracks)
