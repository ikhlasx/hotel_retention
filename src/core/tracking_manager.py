import cv2
import time
from datetime import datetime, date
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .deepsort_tracker import DeepSort, Detection
from .database_manager import DatabaseManager
from .face_engine import FaceRecognitionEngine


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])


class OptimizedFaceTracker:
    def __init__(self, track_id, bbox, embedding):
        self.track_id = track_id
        self.bbox = bbox
        self.embedding = embedding

        # Tracking properties
        self.last_seen = time.time()
        self.display_bbox = bbox
        self.display_confidence = 0.0
        self.display_duration = 4.0

        # Stability and recognition
        self.stability_frames = 0
        # Lower stability requirement to trigger retention logic sooner
        self.min_stability = 8
        self.state_timer = time.time()
        self.display_message = "Analyzing customer..."
        self.message_time = time.time()
        self.state = "checking"

        # Customer retention tracking
        self.customer_id = None
        self.visit_status = None
        self.total_visits = 0
        self.visits_today = 0
        self.is_returning_customer = False
        self.customer_processed = False

        # Recognition tracking
        self.recognition_complete = False
        self.database_checked = False
        self.visit_processed = False
        self.fail_count = 0

    def update_customer_info(self, customer_id, visit_status):
        self.customer_id = customer_id
        self.visit_status = visit_status
        self.customer_processed = True

        if visit_status:
            self.total_visits = visit_status.get('total_visits', 0)
            self.visits_today = visit_status.get('visits_today', 0)
            self.is_returning_customer = visit_status.get('visited_today', False)

            if visit_status.get('visited_today'):
                self.set_retention_message(
                    f"Welcome back!\nAlready counted today\nTotal visits: {self.total_visits}"
                )
                self.state = "already_counted_today"
            else:
                new_total = self.total_visits + 1
                self.set_retention_message(
                    f"Welcome!\nVisit #{new_total} recorded\nThank you for visiting"
                )
                self.state = "new_visit_recorded"

    def set_retention_message(self, message, duration=4.0):
        self.display_message = message
        self.message_time = time.time()
        self.display_duration = duration

    # Compatibility alias
    def set_message(self, message):
        self.set_retention_message(message)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if getattr(self, '_state', None) == value:
            return
        self._state = value
        color_map = {
            'checking': (255, 255, 0),
            'verified_unknown': (0, 255, 255),
            'processing_customer': (0, 255, 0),
            'new_visit_recorded': (0, 200, 0),
            'already_counted_today': (255, 165, 0),
            'returning_customer': (0, 150, 255),
            'new_customer_registered': (255, 0, 255),
        }
        self.color = color_map.get(value, (128, 128, 128))


class TrackingManager:
    def __init__(self, gpu_mode=True):
        self.gpu_mode = gpu_mode
        self.active_tracks = {}
        self.deepsort = DeepSort()

        self.db_manager = DatabaseManager()
        self.face_engine = FaceRecognitionEngine(gpu_mode=gpu_mode)

        self.max_fail_count = 3
        self.customer_processing_timeout = 5.0

    def process_customer_retention(self, track):
        """Run customer retention logic with cosine-similarity verification."""
        try:
            if getattr(track, 'customer_processed', False):
                best_match_id = None
                best_score = 0.0
                threshold = 0.6

                embedding = track.embedding
                for cust_id, stored in self.face_engine.customer_database.items():
                    score = cosine_sim(embedding, stored)
                    if score > best_score:
                        best_score = score
                        best_match_id = cust_id
                    if best_score >= 0.85:
                        break

                if best_score >= threshold and best_match_id:
                    track.confidence = best_score
                    visit_status = self.db_manager.check_daily_visit_status(best_match_id)

                    if visit_status['visited_today']:
                        track.update_customer_info(best_match_id, visit_status)
                        track.set_retention_message(
                            f"Welcome back!\nAlready counted today\nTotal visits: {visit_status['total_visits']}"
                        )
                    else:
                        visit_result = self.db_manager.record_customer_visit(best_match_id, best_score)
                        if visit_result['success']:
                            track.update_customer_info(best_match_id, visit_result)
                            track.set_retention_message(
                                f"Welcome!\nVisit #{visit_result['total_visits']} recorded\nThank you for visiting!"
                            )
                        else:
                            track.set_retention_message("Welcome back!\nAlready processed today")
                else:
                    track.confidence = 0.0
                    if track.stability_frames >= track.min_stability and hasattr(track, 'embedding'):
                        new_customer_id = self.face_engine.register_new_customer(track.embedding)
                        if new_customer_id:
                            visit_result = self.db_manager.record_customer_visit(new_customer_id, best_score)
                            if visit_result['success']:
                                track.update_customer_info(new_customer_id, visit_result)
                                track.set_retention_message(
                                    f"Welcome new customer!\nID: {new_customer_id}\nFirst visit recorded\nThank you for choosing us!"
                                )
                            else:
                                track.set_retention_message("Welcome! Processing...")
                        else:
                            track.set_retention_message("Welcome visitor!")

                track.customer_processed = True

        except Exception as e:
            print(f"❌ Customer retention processing error: {e}")
            track.set_retention_message("Welcome!")

    def update_tracks(self, detections):
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
                    tracker.stability_frames += 1
                    # Invoke retention logic after stability update
                    self.process_customer_retention(tracker)
                else:
                    tracker = OptimizedFaceTracker(track_id, bbox, embedding)
                    tracker.set_retention_message("Analyzing customer...", duration=5.0)
                    self.active_tracks[track_id] = tracker
                    updated_tracks.append(tracker)
            for tid in list(self.active_tracks.keys()):
                if tid not in active_ids:
                    del self.active_tracks[tid]

            return updated_tracks

        except Exception as e:
            print(f"❌ Tracking update error: {e}")
            return list(self.active_tracks.values())

    def draw_retention_info(self, frame, tracks):
        try:
            current_time = time.time()
            for track in tracks:
                if current_time - track.last_seen > track.display_duration:
                    continue

                if not hasattr(track, 'display_bbox'):
                    continue

                bbox = track.display_bbox
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                color = track.color if hasattr(track, 'color') else (128, 128, 128)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                info = f"ID:{track.track_id}"
                if hasattr(track, 'confidence') and track.confidence > 0:
                    info += f" {track.confidence:.2f}"
                cv2.putText(frame, info, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if (
                    hasattr(track, 'display_message')
                    and track.display_message
                    and track.stability_frames >= track.min_stability
                ):
                    if current_time - getattr(track, 'message_time', 0) < track.display_duration:
                        lines = track.display_message.split('\n')[:4]
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

            return frame

        except Exception as e:
            print(f"❌ Retention info drawing error: {e}")
            return frame

    def get_track_count(self):
        return len(self.active_tracks)

