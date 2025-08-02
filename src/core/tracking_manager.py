# src/core/tracking_manager.py - Ultra-Optimized Tracking
import cv2
import time
from datetime import datetime

from deep_sort_realtime.deepsort_tracker import DeepSort


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
        self.failed_identifications = 0

    def set_message(self, message):
        self.display_message = message
        self.message_time = time.time()


class TrackingManager:
    def __init__(self, gpu_mode=True, max_unidentified_frames=10):
        self.gpu_mode = gpu_mode
        self.active_tracks = {}

        # Initialize DeepSort tracker
        self.deepsort = DeepSort(max_age=30)

        # MODIFIED: Increased timeouts for better track stability.
        self.track_timeout = 5.0  # Changed from 3.0
        self.display_timeout = 3.0  # Changed from 2.0

        # Configurable removal threshold for unidentified tracks
        self.max_unidentified_frames = max_unidentified_frames

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
        """Update tracking using DeepSort and manage active tracks."""
        current_time = time.time()

        try:
            raw_dets = []
            embeds = []

            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                w = x2 - x1
                h = y2 - y1
                conf = det.get('confidence', 1.0)
                raw_dets.append(([x1, y1, w, h], conf, 'face'))
                embeds.append(det.get('embedding'))

            ds_tracks = self.deepsort.update_tracks(raw_dets, embeds=embeds)

            active_ids = set()
            for ds_trk in ds_tracks:
                if not ds_trk.is_confirmed() or ds_trk.is_deleted():
                    continue

                track_id = ds_trk.track_id
                bbox = ds_trk.to_ltrb()
                embedding = ds_trk.features[-1] if ds_trk.features else None

                if track_id not in self.active_tracks:
                    new_track = OptimizedFaceTracker(track_id, bbox, embedding)
                    self.active_tracks[track_id] = new_track
                else:
                    tracker = self.active_tracks[track_id]
                    tracker.bbox = bbox
                    tracker.display_bbox = bbox
                    if ds_trk.time_since_update == 0:
                        tracker.last_seen = current_time
                    tracker.embedding = embedding

                active_ids.add(track_id)

            # Remove tracks not present in active DeepSort results
            for tid in list(self.active_tracks.keys()):
                if tid not in active_ids:
                    del self.active_tracks[tid]

            # Cleanup tracks that have timed out
            self._cleanup_old_tracks_fast(current_time)

            return list(self.active_tracks.values())

        except Exception as e:
            print(f"❌ Tracking update error: {e}")
            return list(self.active_tracks.values())

    def _cleanup_old_tracks_fast(self, current_time):
        """Ultra-fast track cleanup from reference main.py"""
        tracks_to_remove = [
            track_id for track_id, tracker in self.active_tracks.items()
            if current_time - tracker.last_seen > self.track_timeout
        ]

        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]

        return tracks_to_remove

    def remove_track(self, track_id):
        """Remove a track by its ID."""
        if track_id in self.active_tracks:
            del self.active_tracks[track_id]

    def get_track_count(self):
        """Get number of active tracks"""
        return len(self.active_tracks)
