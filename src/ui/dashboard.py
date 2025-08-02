# src/ui/dashboard.py - Complete Ultra-Optimized Implementation with All Issues Fixed

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime, date
import numpy as np
import os

# Import optimized modules
from core.face_engine import FaceRecognitionEngine
from core.tracking_manager import TrackingManager
from utils.camera_utils import CameraManager
from core.config_manager import ConfigManager


class HotelDashboard:
    def __init__(self, root, gpu_available=False):
        self.parent = root
        self.gpu_available = gpu_available
        self.running = False
        self.config = ConfigManager()

        # Initialize fullscreen variables FIRST
        self.fullscreen_window = None
        self.fullscreen_label = None
        self.fullscreen_active = False

        # Frame capture system
        self.current_frame = None
        self.current_tracks = []  # Share track metadata with fullscreen loop
        self.capture_lock = threading.Lock()
        self.captured_photos = []
        self.auto_capture_enabled = False

        # Visit tracking system
        self.visit_counts = {
            'total_today': 0,
            'known_customers': 0,
            'new_customers': 0,
            'staff_checkins': 0
        }

        # Track customers counted for the current day
        self.current_date = date.today()
        self.customers_today = set()

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.frame_count = 0

        # Ultra-optimization settings
        self.process_every_n_frames = 2  # Process every 2nd frame for performance
        self.checking_duration = 1.0  # Faster checking
        self.display_duration = 3.0  # Shorter display
        self.auto_register_delay = 2.0  # Faster registration

        # Anti-flicker control system
        self.last_message_update = 0
        self.last_stats_update = 0
        self.message_update_interval = 0.5  # 500ms minimum between updates
        self.frame_update_interval = 0.033  # 30 FPS max
        self.update_in_progress = False

        # Message queuing system
        self.pending_welcome_messages = []
        self.pending_detection_updates = []
        self.detection_update_scheduled = False

        # Initialize detection statistics
        self.detection_stats = {
            'total_detections': 0,
            'known_customers': 0,
            'unknown_persons': 0,
            'staff_detections': 0
        }

        self.setup_gui()
        self.setup_engines()

    def setup_gui(self):
        """Setup ultra-optimized GUI with enhanced photo capture capabilities"""
        # Create main paned window
        paned = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for video and controls
        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=3)

        # Right panel for statistics and logs
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=1)

        # Enhanced Video Frame with capture capabilities
        video_frame = ttk.LabelFrame(left_panel, text="🎥 Live Camera Feed - Stable Processing", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(video_frame, bg='black', text="Camera Disconnected")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Enhanced Controls Frame
        control_frame = ttk.LabelFrame(left_panel, text="🎮 System Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        # Main control buttons
        main_buttons = ttk.Frame(control_frame)
        main_buttons.pack(fill=tk.X, pady=5)

        self.start_btn = ttk.Button(main_buttons, text="▶ Start Ultra Recognition",
                                    command=self.start_ultra_recognition)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(main_buttons, text="⏹ Stop Recognition",
                                   command=self.stop_recognition, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Photo capture controls
        capture_buttons = ttk.Frame(control_frame)
        capture_buttons.pack(fill=tk.X, pady=5)

        self.capture_btn = ttk.Button(capture_buttons, text="📸 Capture Photo",
                                      command=self.capture_current_frame, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        self.auto_capture_var = tk.BooleanVar()
        self.auto_capture_check = ttk.Checkbutton(capture_buttons, text="🔄 Auto Capture",
                                                  variable=self.auto_capture_var,
                                                  command=self.toggle_auto_capture)
        self.auto_capture_check.pack(side=tk.LEFT, padx=5)

        self.fullscreen_btn = ttk.Button(capture_buttons, text="🔳 Fullscreen",
                                         command=self.toggle_fullscreen_camera)
        self.fullscreen_btn.pack(side=tk.LEFT, padx=5)

        # Test and debug controls
        debug_buttons = ttk.Frame(control_frame)
        debug_buttons.pack(fill=tk.X, pady=5)

        # self.test_msg_btn = ttk.Button(debug_buttons, text="🧪 Test Messages",
        #                                command=self.test_messages)
        # self.test_msg_btn.pack(side=tk.LEFT, padx=5)

        self.reset_stats_btn = ttk.Button(debug_buttons, text="🔄 Reset Stats",
                                          command=self.reset_visit_stats)
        self.reset_stats_btn.pack(side=tk.LEFT, padx=5)

        # Mode selection
        mode_frame = ttk.Frame(debug_buttons)
        mode_frame.pack(side=tk.RIGHT)

        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="GPU" if self.gpu_available else "CPU")

        mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var,
                                  values=["GPU", "CPU"] if self.gpu_available else ["CPU"],
                                  state="readonly", width=8)
        mode_combo.pack(side=tk.LEFT, padx=5)

        # Enhanced status display
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, pady=5)

        self.camera_status = ttk.Label(status_frame, text="📹 Camera: Disconnected", foreground="red")
        self.camera_status.pack(side=tk.LEFT)

        self.processing_status = ttk.Label(status_frame, text="🧠 Processing: Stopped", foreground="red")
        self.processing_status.pack(side=tk.RIGHT)

        # Setup enhanced right panel
        self.setup_enhanced_right_panel(right_panel)

    def setup_enhanced_right_panel(self, right_panel):
        """Setup enhanced right panel with photo gallery and visit counts"""
        # Real-time System Statistics
        stats_frame = ttk.LabelFrame(right_panel, text="📊 Live Performance Stats", padding=10)
        stats_frame.pack(fill=tk.X, pady=5)

        self.fps_label = ttk.Label(stats_frame, text="FPS: 0", font=("Arial", 10, "bold"))
        self.fps_label.pack(anchor=tk.W)

        self.tracks_label = ttk.Label(stats_frame, text="Active Tracks: 0", font=("Arial", 10))
        self.tracks_label.pack(anchor=tk.W)

        self.customers_label = ttk.Label(stats_frame, text="Total Customers: 0", font=("Arial", 10))
        self.customers_label.pack(anchor=tk.W)

        self.staff_label = ttk.Label(stats_frame, text="Staff Members: 0", font=("Arial", 10))
        self.staff_label.pack(anchor=tk.W)

        # Enhanced Visit Counts Display
        visits_frame = ttk.LabelFrame(right_panel, text="🎯 Today's Activity", padding=10)
        visits_frame.pack(fill=tk.X, pady=5)

        self.visits_today_label = ttk.Label(visits_frame, text="Total Visits: 0",
                                            font=("Arial", 12, "bold"), foreground="blue")
        self.visits_today_label.pack(anchor=tk.W)

        self.known_customers_label = ttk.Label(visits_frame, text="Known Customers: 0", font=("Arial", 10))
        self.known_customers_label.pack(anchor=tk.W)

        self.new_customers_label = ttk.Label(visits_frame, text="New Customers: 0", font=("Arial", 10))
        self.new_customers_label.pack(anchor=tk.W)

        self.staff_checkins_label = ttk.Label(visits_frame, text="Staff Check-ins: 0", font=("Arial", 10))
        self.staff_checkins_label.pack(anchor=tk.W)

        # Enhanced Recent Detections
        detections_frame = ttk.LabelFrame(right_panel, text="🔍 Live Detections", padding=5)
        detections_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ('Time', 'Type', 'ID', 'Confidence', 'Status')
        self.recent_tree = ttk.Treeview(detections_frame, columns=columns, show='headings', height=8)

        for col in columns:
            self.recent_tree.heading(col, text=col)
            self.recent_tree.column(col, width=80)

        detection_scrollbar = ttk.Scrollbar(detections_frame, orient=tk.VERTICAL, command=self.recent_tree.yview)
        self.recent_tree.configure(yscrollcommand=detection_scrollbar.set)

        self.recent_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detection_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Photo Gallery
        photos_frame = ttk.LabelFrame(right_panel, text="📸 Captured Photos", padding=5)
        photos_frame.pack(fill=tk.X, pady=5)

        self.photos_listbox = tk.Listbox(photos_frame, height=4)
        self.photos_listbox.pack(fill=tk.X)

        photo_buttons = ttk.Frame(photos_frame)
        photo_buttons.pack(fill=tk.X, pady=2)

        ttk.Button(photo_buttons, text="🔍 View", command=self.view_captured_photo).pack(side=tk.LEFT, padx=2)
        ttk.Button(photo_buttons, text="💾 Save", command=self.save_captured_photo).pack(side=tk.LEFT, padx=2)
        ttk.Button(photo_buttons, text="🗑️ Clear", command=self.clear_captured_photos).pack(side=tk.LEFT, padx=2)

        # Welcome Messages with anti-flicker
        messages_frame = ttk.LabelFrame(right_panel, text="💬 Welcome Messages", padding=5)
        messages_frame.pack(fill=tk.X, pady=5)

        self.welcome_text = tk.Text(messages_frame, height=6, wrap=tk.WORD, font=("Arial", 9))
        welcome_scrollbar = ttk.Scrollbar(messages_frame, orient=tk.VERTICAL, command=self.welcome_text.yview)
        self.welcome_text.configure(yscrollcommand=welcome_scrollbar.set)

        self.welcome_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        welcome_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        clear_btn = ttk.Button(messages_frame, text="Clear", command=self.clear_welcome_messages)
        clear_btn.pack(side=tk.BOTTOM, pady=2)

    def setup_engines(self):
        """Initialize ultra-optimized recognition engines"""
        self.face_engine = None
        self.tracking_manager = None
        self.camera_manager = CameraManager()

        # Load initial statistics
        self.update_database_statistics()

    def start_ultra_recognition(self):
        """Start ultra-optimized face recognition system with photo capture"""
        try:
            print("🚀 Starting Ultra-High-Performance Recognition System...")

            # Initialize engines
            gpu_mode = self.mode_var.get() == "GPU" and self.gpu_available

            self.face_engine = FaceRecognitionEngine(gpu_mode=gpu_mode)
            # Set optimal detection threshold for stability
            if hasattr(self.face_engine, 'detection_threshold'):
                self.face_engine.detection_threshold = 0.6  # Balanced for visibility and accuracy

            self.tracking_manager = TrackingManager(gpu_mode=gpu_mode)

            # Start camera with stability optimizations
            if not self.camera_manager.start_camera():
                messagebox.showerror("Camera Error", "Failed to start camera")
                return

            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.capture_btn.config(state=tk.NORMAL)

            # Update status
            self.camera_status.config(text="📹 Camera: Connected", foreground="green")
            self.processing_status.config(text="🧠 Processing: Active", foreground="green")

            # Start optimized processing thread
            self.process_thread = threading.Thread(target=self.ultra_stable_video_processing, daemon=True)
            self.process_thread.start()

            # Start statistics update thread
            self.stats_thread = threading.Thread(target=self.update_statistics_loop, daemon=True)
            self.stats_thread.start()

            # Display startup message
            startup_msg = f"🎉 SYSTEM STARTED SUCCESSFULLY\n"
            startup_msg += f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
            startup_msg += f"Mode: {'GPU Accelerated' if gpu_mode else 'CPU Processing'}\n"
            startup_msg += f"Photo Capture: Enabled\n"
            startup_msg += f"Auto-Capture: {'ON' if self.auto_capture_var.get() else 'OFF'}\n"
            startup_msg += "=" * 40 + "\n\n"

            self.display_welcome_message(startup_msg)

            print("✅ Ultra-optimized recognition system started successfully")

        except Exception as e:
            print(f"❌ Start recognition error: {e}")
            messagebox.showerror("Error", f"Failed to start recognition: {e}")

    def ultra_stable_video_processing(self):
        """Ultra-stable video processing loop with all optimizations"""
        print("🎬 Starting ultra-stable video processing with frame capturing...")

        last_gui_update = 0
        last_auto_capture = 0
        frame_skip_counter = 0

        tracks = []

        while self.running:
            try:
                # Get stable frame with error handling
                frame = self.get_stable_frame()
                if frame is None:
                    time.sleep(0.02)
                    continue

                self.frame_count += 1
                current_time = time.time()

                # Process face detection every Nth frame for stability
                # Detection/tracking is centralized here for all display modes
                detections = []

                if self.frame_count % self.process_every_n_frames == 0:
                    detections = self.face_engine.detect_faces(frame)
                    tracks = self.tracking_manager.update_tracks(detections)

                    if detections:
                        print(f"🔍 Frame {self.frame_count}: Found {len(detections)} stable detections")

                        # Process each track with visit counting
                        for track in tracks:
                            self.process_track_with_visit_counting(track)

                        # Auto-capture on detection if enabled
                        if (self.auto_capture_var.get() and
                                current_time - last_auto_capture > 5.0):  # Every 5 seconds
                            self.auto_capture_detection(tracks)
                            last_auto_capture = current_time

                        # Update statistics
                        self.update_dashboard_statistics(detections, tracks)
                else:
                    tracks = list(self.tracking_manager.active_tracks.values())

                # Draw enhanced overlays with visit counts and bounding boxes
                frame = self.draw_enhanced_overlays_stable(frame, detections, tracks)

                # Update current frame for capture system (thread-safe)
                with self.capture_lock:
                    self.current_frame = frame.copy()
                    self.current_tracks = list(tracks)

                # GUI update with anti-flicker throttling
                if current_time - last_gui_update >= self.frame_update_interval:
                    self.parent.after_idle(self.update_display_ultra_stable, frame.copy())
                    last_gui_update = current_time

                # Update FPS
                self.update_fps()

                # Update dashboard statistics every 10 frames
                if self.frame_count % 10 == 0:
                    self.parent.after_idle(self.update_realtime_dashboard, detections, tracks)

                # Sleep for stability (50 FPS max)
                time.sleep(0.02)

            except Exception as e:
                print(f"❌ Ultra-stable processing error: {e}")
                time.sleep(0.1)

    def get_stable_frame(self):
        """Get stable frame with multiple attempts and error handling"""
        try:
            # Multiple attempts for frame stability
            for attempt in range(3):
                frame = self.camera_manager.get_frame()
                if frame is not None and frame.size > 0:
                    return frame
                time.sleep(0.01)
            return None
        except Exception as e:
            print(f"Frame acquisition error: {e}")
            return None

    def draw_enhanced_overlays_stable(self, frame, detections, tracks):
        """Draw enhanced overlays with visit counts, clear bounding boxes, and stability"""
        try:
            frame_height, frame_width = frame.shape[:2]

            color_map = {
                'checking': (255, 255, 0),
                'verified_unknown': (0, 255, 255),
                'processing_visit': (0, 255, 0),
                'registered': (255, 0, 255),
                'verified_known': (0, 200, 0),
                'processing_staff_attendance': (255, 0, 0),
                'verified_staff': (255, 100, 0)
            }

            # First, draw default boxes for tracks that are still being checked
            checking_tracks = [t for t in tracks if getattr(t, "state", "") == "checking"]
            if checking_tracks:
                # Temporarily suppress messages for checking tracks
                saved_messages = [getattr(t, "display_message", None) for t in checking_tracks]
                for t in checking_tracks:
                    t.display_message = None

                if hasattr(self, "tracking_manager"):
                    frame = self.tracking_manager.draw_tracks(frame, checking_tracks)
                else:
                    for t in checking_tracks:
                        if hasattr(t, "bbox"):
                            bbox = getattr(t, "display_bbox", t.bbox)
                            x1, y1, x2, y2 = [int(coord) for coord in bbox]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)

                # Restore messages (if any) for future use
                for t, msg in zip(checking_tracks, saved_messages):
                    t.display_message = msg

            # Draw tracking boxes only for recognized individuals
            for track in tracks:
                if not hasattr(track, 'bbox'):
                    continue

                # Skip tracks that are still being checked (already drawn above)
                if getattr(track, 'state', '') == 'checking':
                    continue

                # Only display tracks that have been identified
                if getattr(track, 'state', '') not in {
                    'processing_visit', 'verified_known',
                    'processing_staff_attendance', 'verified_staff'
                }:
                    continue

                bbox = getattr(track, "display_bbox", track.bbox)
                x1, y1, x2, y2 = [int(coord) for coord in bbox]

                color = color_map.get(track.state, (128, 128, 128))

                # Draw thick tracking box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

                # Enhanced track information
                track_info = f"ID: {track.track_id} | {getattr(track, 'state', 'unknown')}"
                cv2.putText(frame, track_info, (x1, y1 - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Confidence and stability information
                if hasattr(track, 'confidence') and track.confidence > 0:
                    conf_text = f"Confidence: {track.confidence:.2f}"
                    cv2.putText(frame, conf_text, (x1, y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if hasattr(track, 'stability_frames'):
                    stab_text = f"Stability: {track.stability_frames}"
                    cv2.putText(frame, stab_text, (x1, y2 + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Display status messages
                if (
                    hasattr(track, "display_message")
                    and track.display_message
                    and time.time() - getattr(track, "message_time", 0) < track.display_duration
                ):
                    lines = track.display_message.split("\n")[:2]  # Max 2 lines
                    for j, line in enumerate(lines):
                        msg_y = y2 + 75 + (j * 25)
                        if msg_y < frame_height - 10:
                            cv2.putText(
                                frame,
                                line,
                                (x1, msg_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color,
                                2,
                            )
                elif hasattr(track, "display_message") and track.display_message:
                    track.display_message = None

            # Draw visit counts overlay (top-left)
            self.draw_visit_counts_overlay(frame)

            # Draw system status overlay (top-right)
            self.draw_system_status_overlay(frame)

            return frame

        except Exception as e:
            print(f"❌ Enhanced overlay drawing error: {e}")
            return frame

    def draw_visit_counts_overlay(self, frame):
        """Draw real-time visit counts overlay on video feed"""
        try:
            frame_height, frame_width = frame.shape[:2]

            # Visit counts information
            visit_info = [
                f"📊 TODAY'S VISITS",
                f"Total: {self.visit_counts['total_today']}",
                f"Known: {self.visit_counts['known_customers']}",
                f"New: {self.visit_counts['new_customers']}",
                f"Staff: {self.visit_counts['staff_checkins']}"
            ]

            # Position at top-left
            start_x, start_y = 20, 30
            box_width, box_height = 200, len(visit_info) * 25 + 20

            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (start_x - 10, start_y - 20),
                          (start_x + box_width, start_y + box_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Draw visit count text with color coding
            for i, info in enumerate(visit_info):
                y_pos = start_y + i * 25
                color = (0, 200, 255) if i == 0 else (255, 255, 255)  # Header in orange, data in white
                thickness = 2 if i == 0 else 1
                cv2.putText(frame, info, (start_x, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

        except Exception as e:
            print(f"Visit counts overlay error: {e}")

    def draw_system_status_overlay(self, frame):
        """Draw enhanced system status overlay"""
        try:
            frame_height, frame_width = frame.shape[:2]

            # System status information
            status_info = [
                f"FPS: {self.current_fps:.1f}",
                f"Frame: {self.frame_count}",
                f"Photos: {len(self.captured_photos)}",
                f"Auto-Cap: {'ON' if self.auto_capture_var.get() else 'OFF'}",
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            ]

            # Position at top-right
            start_x = frame_width - 220
            start_y = 30

            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (start_x - 10, start_y - 20),
                          (start_x + 210, start_y + len(status_info) * 25 + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Draw status text
            for i, info in enumerate(status_info):
                y_pos = start_y + i * 25
                cv2.putText(frame, info, (start_x, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        except Exception as e:
            print(f"System status overlay error: {e}")

    def process_track_with_visit_counting(self, track):
        """Process track with integrated visit counting and state management"""
        try:
            current_time = time.time()

            # Initialize track attributes if needed
            if not hasattr(track, 'stability_frames'):
                track.stability_frames = 0
            if not hasattr(track, 'state'):
                track.state = "checking"
                track.state_timer = current_time
            if not hasattr(track, 'fail_count'):
                track.fail_count = 0

            # State: CHECKING
            if track.state == "checking":
                track.stability_frames += 1

                # MODIFIED: Increased stability frames for more reliable identification
                if track.stability_frames >= 15 and current_time - track.state_timer >= 1.5: # Changed from 12
                    # Identify person with enhanced threshold
                    person_type, person_id, confidence = self.face_engine.identify_person(track.embedding)
                    
                    if person_type == 'customer' and confidence >= 0.55:
                        track.customer_id = person_id
                        track.confidence = confidence
                        track.state = "processing_visit"
                        track.state_timer = current_time
                        track.set_message("Known customer recognized")
                        track.fail_count = 0  # Reset on success

                    elif person_type == 'staff' and confidence >= 0.65:
                        track.person_id = person_id
                        track.confidence = confidence
                        track.state = "processing_staff_attendance"
                        track.state_timer = current_time
                        track.set_message("Staff member detected")
                        track.fail_count = 0  # Reset on success

                        # Update staff count
                        self.visit_counts['staff_checkins'] += 1

                    else:
                        # No match found; keep checking without triggering messages
                        track.state = "checking"
                        track.state_timer = current_time
                        track.set_message(None)
                        track.fail_count += 1

                        # Drop track if it fails too many times
                        if (
                            self.tracking_manager
                            and track.fail_count > self.tracking_manager.max_fail_count
                        ):
                            if track.track_id in self.tracking_manager.active_tracks:
                                del self.tracking_manager.active_tracks[track.track_id]
                            return

            # State: PROCESSING_VISIT
            elif track.state == "processing_visit" and not getattr(track, 'visit_processed', False):
                self.process_customer_visit_with_counting(track)

            # State: PROCESSING_STAFF_ATTENDANCE
            elif track.state == "processing_staff_attendance" and not getattr(track, 'visit_processed', False):
                self.process_staff_attendance_with_counting(track)


        except Exception as e:
            print(f"❌ Track processing error: {e}")

    def _reset_daily_customers_if_needed(self):
        """Reset daily customer tracking when the date changes."""
        today = date.today()
        if today != self.current_date:
            self.current_date = today
            self.customers_today.clear()


    def process_customer_visit_with_counting(self, track):
        """Process customer visits with daily duplicate prevention."""
        try:
            if hasattr(self.face_engine, 'db_manager'):
                self._reset_daily_customers_if_needed()
                customer_info = self.face_engine.db_manager.get_customer_info(track.customer_id)

                if customer_info:
                    total_visits = customer_info.get('total_visits', 0)
                    last_visit = customer_info.get('last_visit')
                    last_visit_date = None
                    if last_visit:
                        try:
                            last_visit_date = datetime.fromisoformat(last_visit).date()
                        except Exception:
                            try:
                                last_visit_date = datetime.strptime(last_visit, "%Y-%m-%d %H:%M:%S").date()
                            except Exception:
                                last_visit_date = None

                    today = self.current_date

                    if track.customer_id in self.customers_today or last_visit_date == today:
                        track.set_message(f"Already counted for today\nVisit #{total_visits}")
                        info_msg = (
                            f"⚠️ CUSTOMER VISIT\n"
                            f"ID: {track.customer_id}\n"
                            f"Already counted for today\n"
                            f"Visit #{total_visits}\n"
                            f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
                            + "=" * 40 + "\n\n"
                        )
                        self.display_welcome_message(info_msg)
                    else:
                        success = self.face_engine.db_manager.record_visit(track.customer_id, track.confidence)
                        if success:
                            total_visits += 1
                            self.customers_today.add(track.customer_id)
                            self.visit_counts['known_customers'] += 1
                            self.visit_counts['total_today'] += 1
                            track.set_message(f"Welcome {track.customer_id}\nVisit #{total_visits}")
                            welcome_msg = (
                                f"🎉 CUSTOMER RECOGNIZED\n"
                                f"ID: {track.customer_id}\n"
                                f"Visit #{total_visits}\n"
                                f"Confidence: {track.confidence:.2f}\n"
                                f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
                                + "=" * 40 + "\n\n"
                            )
                            self.display_welcome_message(welcome_msg)
                            customer_name = customer_info.get('name', f"Customer {track.customer_id}")
                            self.add_recent_detection("Customer", track.customer_id, customer_name, track.confidence)
                            print(f"✅ Customer visit processed: {track.customer_id} (Visit #{total_visits})")

                    track.visit_processed = True
                    track.state = "verified_known"
                    track.state_timer = time.time()
        except Exception as e:
            print(f"❌ Customer visit processing error: {e}")

    def process_staff_attendance_with_counting(self, track):
        """Process staff attendance with counting integration"""
        try:
            if hasattr(self.face_engine, 'db_manager'):
                staff_info = self.face_engine.db_manager.get_staff_info(track.person_id)

                if staff_info:
                    staff_name = staff_info.get('name', track.person_id)

                    # Show attendance message
                    attendance_msg = f"👨‍💼 STAFF CHECK-IN\n"
                    attendance_msg += f"Name: {staff_name}\n"
                    attendance_msg += f"Department: {staff_info.get('department', 'N/A')}\n"
                    attendance_msg += f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
                    attendance_msg += f"Confidence: {track.confidence:.2f}\n"
                    attendance_msg += "=" * 40 + "\n\n"

                    self.display_welcome_message(attendance_msg)

                    # Add to recent detections
                    self.add_recent_detection("Staff", track.person_id, staff_name, track.confidence)

                    track.visit_processed = True
                    track.state = "verified_staff"
                    track.state_timer = time.time()

        except Exception as e:
            print(f"❌ Staff attendance processing error: {e}")

    def capture_current_frame(self):
        """Capture current frame as photo with timestamp"""
        try:
            with self.capture_lock:
                if self.current_frame is not None:
                    # Create timestamped capture
                    timestamp = datetime.now()
                    photo_data = {
                        'timestamp': timestamp,
                        'frame': self.current_frame.copy(),
                        'filename': f"capture_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                    }

                    self.captured_photos.append(photo_data)

                    # Update photos listbox
                    self.photos_listbox.insert(0,
                                               f"{timestamp.strftime('%H:%M:%S')} - Photo {len(self.captured_photos)}")

                    # Limit to 20 photos for memory management
                    if len(self.captured_photos) > 20:
                        self.captured_photos.pop(0)
                        self.photos_listbox.delete(tk.END)

                    # Display success message
                    success_msg = f"📸 PHOTO CAPTURED\n"
                    success_msg += f"Time: {timestamp.strftime('%H:%M:%S')}\n"
                    success_msg += f"Total Photos: {len(self.captured_photos)}\n"
                    success_msg += f"Filename: {photo_data['filename']}\n"
                    success_msg += "=" * 30 + "\n\n"

                    self.display_welcome_message(success_msg)

                    print(f"✅ Photo captured: {photo_data['filename']}")
                    return True
                else:
                    messagebox.showwarning("Capture Error", "No frame available for capture")
                    return False

        except Exception as e:
            print(f"❌ Photo capture error: {e}")
            messagebox.showerror("Capture Error", f"Failed to capture photo: {e}")
            return False

    def auto_capture_detection(self, tracks):
        """Automatically capture photos when faces are detected"""
        try:
            if tracks and len(tracks) > 0:
                # Auto-capture only for recognized tracks
                for track in tracks:
                    if (
                        getattr(track, 'state', '') in {
                            'processing_visit', 'verified_known',
                            'processing_staff_attendance', 'verified_staff'
                        }
                        and hasattr(track, 'confidence') and track.confidence > 0.6
                        and hasattr(track, 'stability_frames') and track.stability_frames > 15
                    ):

                        success = self.capture_current_frame()
                        if success:
                            auto_msg = f"📸 AUTO-CAPTURED\n"
                            auto_msg += f"Track ID: {track.track_id}\n"
                            auto_msg += f"Confidence: {track.confidence:.2f}\n"
                            auto_msg += f"State: {getattr(track, 'state', 'unknown')}\n"
                            auto_msg += "-" * 25 + "\n\n"

                            self.display_welcome_message(auto_msg)
                        break  # Only capture one per cycle

        except Exception as e:
            print(f"❌ Auto-capture error: {e}")

    def toggle_auto_capture(self):
        """Toggle auto-capture mode"""
        self.auto_capture_enabled = self.auto_capture_var.get()
        status = "enabled" if self.auto_capture_enabled else "disabled"

        toggle_msg = f"🔄 AUTO-CAPTURE {status.upper()}\n"
        toggle_msg += f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
        toggle_msg += "=" * 30 + "\n\n"

        self.display_welcome_message(toggle_msg)
        print(f"✅ Auto-capture {status}")

    def view_captured_photo(self):
        """View selected captured photo in a new window"""
        try:
            selection = self.photos_listbox.curselection()
            if not selection:
                messagebox.showwarning("Selection Error", "Please select a photo to view")
                return

            photo_index = selection[0]
            if photo_index < len(self.captured_photos):
                photo_data = self.captured_photos[photo_index]

                # Create photo viewer window
                viewer = tk.Toplevel(self.parent)
                viewer.title(f"Photo Viewer - {photo_data['filename']}")
                viewer.geometry("800x600")

                # Convert and display photo
                frame = photo_data['frame']
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                pil_image.thumbnail((780, 580))
                photo = ImageTk.PhotoImage(pil_image)

                label = tk.Label(viewer, image=photo)
                label.image = photo  # Keep reference
                label.pack(expand=True)

        except Exception as e:
            print(f"❌ Photo viewer error: {e}")
            messagebox.showerror("Viewer Error", f"Failed to view photo: {e}")

    def save_captured_photo(self):
        """Save selected captured photo to file"""
        try:
            selection = self.photos_listbox.curselection()
            if not selection:
                messagebox.showwarning("Selection Error", "Please select a photo to save")
                return

            photo_index = selection[0]
            if photo_index < len(self.captured_photos):
                photo_data = self.captured_photos[photo_index]

                # Ask for save location
                filename = filedialog.asksaveasfilename(
                    defaultextension=".jpg",
                    filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")],
                    initialname=photo_data['filename']
                )

                if filename:
                    cv2.imwrite(filename, photo_data['frame'])
                    messagebox.showinfo("Save Success", f"Photo saved to: {filename}")

                    save_msg = f"💾 PHOTO SAVED\n"
                    save_msg += f"File: {os.path.basename(filename)}\n"
                    save_msg += f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
                    save_msg += "=" * 25 + "\n\n"

                    self.display_welcome_message(save_msg)

        except Exception as e:
            print(f"❌ Photo save error: {e}")
            messagebox.showerror("Save Error", f"Failed to save photo: {e}")

    def clear_captured_photos(self):
        """Clear all captured photos"""
        try:
            if len(self.captured_photos) > 0:
                result = messagebox.askyesno("Clear Photos",
                                             f"Are you sure you want to clear all {len(self.captured_photos)} captured photos?")
                if result:
                    self.captured_photos.clear()
                    self.photos_listbox.delete(0, tk.END)

                    clear_msg = f"🗑️ PHOTOS CLEARED\n"
                    clear_msg += f"All captured photos removed\n"
                    clear_msg += f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
                    clear_msg += "=" * 25 + "\n\n"

                    self.display_welcome_message(clear_msg)
                    print("✅ All captured photos cleared")
            else:
                messagebox.showinfo("Clear Photos", "No photos to clear")

        except Exception as e:
            print(f"❌ Clear photos error: {e}")

    def display_welcome_message(self, message):
        """Anti-flicker welcome message display with stable appending."""
        try:
            # Ensure the text widget is in a normal state to be modified
            self.welcome_text.config(state='normal')
            
            # Insert the new message at the end of the text box
            self.welcome_text.insert(tk.END, message)
            
            # Automatically scroll to the end to show the latest message
            self.welcome_text.see(tk.END)
            
            # Trim the text from the top if it exceeds 100 lines to prevent memory issues
            num_lines = int(self.welcome_text.index('end-1c').split('.')[0])
            if num_lines > 100:
                self.welcome_text.delete('1.0', f'{num_lines - 100}.0')

            # Disable the widget to prevent user editing
            self.welcome_text.config(state='disabled')
            
            # Force the UI to update immediately
            self.welcome_text.update_idletasks()

            # Queue message for fullscreen mirroring
            self.pending_welcome_messages.append(message)

        except Exception as e:
            print(f"❌ Welcome message display error: {e}")

    def add_recent_detection(self, person_type, person_id, name, confidence=0.0):
        """Anti-flicker recent detection updates with proper state management"""
        try:
            current_time = time.time()

            # Batch updates to prevent rapid-fire GUI changes
            if not hasattr(self, 'last_tree_update'):
                self.last_tree_update = 0

            if current_time - self.last_tree_update < 0.5:  # 500ms minimum between updates
                return

            # Disable selection events during update
            self.recent_tree.configure(selectmode='none')

            # Single atomic insert operation
            timestamp = datetime.now().strftime('%H:%M:%S')

            # Check for duplicates in last 5 entries to prevent spam
            children = self.recent_tree.get_children()
            is_duplicate = False

            for item in children[:5]:  # Check only recent entries
                values = self.recent_tree.item(item)['values']
                if len(values) >= 3 and values[1] == person_type and values[2] == person_id:
                    is_duplicate = True
                    break

            if not is_duplicate:
                # Insert new item at top
                confidence_str = f"{confidence:.2f}" if confidence > 0 else "N/A"
                status = "Active" if person_type in ["Customer", "Staff"] else "New"

                new_item = self.recent_tree.insert('', 0, values=(
                    timestamp, person_type, person_id, confidence_str, status
                ))

                # Limit to 30 items for performance
                if len(children) >= 30:
                    for item in children[29:]:  # Keep only first 29 + new item
                        self.recent_tree.delete(item)

                # Highlight new item briefly
                self.recent_tree.selection_set(new_item)
                self.parent.after(2000, lambda: self.recent_tree.selection_remove(new_item))

            # Restore selection mode
            self.recent_tree.configure(selectmode='extended')

            # Force update to prevent batching
            self.recent_tree.update_idletasks()

            self.last_tree_update = current_time

        except Exception as e:
            print(f"❌ Anti-flicker detection update error: {e}")

    def update_display_ultra_stable(self, frame):
        """Thread-safe display update with anti-flicker double buffering"""
        try:
            if frame is None or frame.size == 0:
                return

            # Create processing copy to avoid frame reference issues
            frame_copy = frame.copy()

            # Get current label dimensions
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()

            # Skip update if label is too small (prevents flicker during resize)
            if label_width < 50 or label_height < 50:
                return

            # Calculate optimal dimensions with aspect ratio preservation
            h, w = frame_copy.shape[:2]
            camera_aspect_ratio = w / h
            label_aspect_ratio = label_width / label_height

            if label_aspect_ratio > camera_aspect_ratio:
                new_height = label_height
                new_width = int(new_height * camera_aspect_ratio)
            else:
                new_width = label_width
                new_height = int(new_width / camera_aspect_ratio)

            # High-quality resize with anti-aliasing
            frame_resized = cv2.resize(frame_copy, (new_width, new_height),
                                       interpolation=cv2.INTER_LANCZOS4)

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Create PIL image
            pil_image = Image.fromarray(frame_rgb)

            # CRITICAL: Create PhotoImage with proper reference handling
            new_photo = ImageTk.PhotoImage(pil_image)

            # Store old photo reference before updating (prevents garbage collection flicker)
            old_photo = getattr(self.video_label, 'image', None)

            # Atomic update: set both reference and display simultaneously
            self.video_label.image = new_photo  # Store reference FIRST
            self.video_label.configure(image=new_photo)  # Then configure display

            # Force immediate update to prevent batching
            self.video_label.update_idletasks()

            # Clean up old reference after new one is displayed (prevents memory leaks)
            if old_photo is not None:
                del old_photo

        except Exception as e:
            print(f"❌ Anti-flicker display update error: {e}")

    def update_dashboard_statistics(self, detections, tracks):
        """Anti-flicker statistics update with proper synchronization"""
        try:
            current_time = time.time()

            # Prevent overlapping updates
            if self.update_in_progress:
                return

            # Minimum interval between updates
            if current_time - self.last_stats_update < 0.5:
                return

            self.update_in_progress = True

            # Batch all statistics updates together
            stats_updates = {}

            # FPS update
            stats_updates['fps'] = f"FPS: {self.current_fps:.1f}"

            # Active tracks count
            active_count = len(tracks) if tracks else 0
            stats_updates['tracks'] = f"Active Tracks: {active_count}"

            # Apply all updates simultaneously to prevent multiple redraws
            self.fps_label.config(text=stats_updates['fps'])
            self.tracks_label.config(text=stats_updates['tracks'])

            # Force single update cycle
            self.fps_label.update_idletasks()
            self.tracks_label.update_idletasks()

            self.last_stats_update = current_time
            self.update_in_progress = False

        except Exception as e:
            print(f"❌ Anti-flicker statistics error: {e}")
            self.update_in_progress = False

    def update_realtime_dashboard(self, detections=None, tracks=None):
        """Update dashboard UI elements with real-time data and live detections."""
        try:
            # Update visit count displays
            self.visits_today_label.config(text=f"Total Visits: {self.visit_counts['total_today']}")
            self.known_customers_label.config(text=f"Known Customers: {self.visit_counts['known_customers']}")
            self.new_customers_label.config(text=f"New Customers: {self.visit_counts['new_customers']}")
            self.staff_checkins_label.config(text=f"Staff Check-ins: {self.visit_counts['staff_checkins']}")

            # Update system statistics if available
            if hasattr(self, 'face_engine') and self.face_engine:
                try:
                    stats = self.face_engine.get_statistics()
                    total_customers = stats.get('total_customers', 0)
                    self.customers_label.config(text=f"Total Customers: {total_customers}")

                    total_staff = stats.get('total_staff', 0)
                    self.staff_label.config(text=f"Staff Members: {total_staff}")
                except:
                    pass  # Graceful fallback if statistics not available

            # Auto-populate live detections from recognized tracks
            if tracks:
                for track in tracks:
                    if getattr(track, 'state', '') in {
                        'processing_visit', 'verified_known',
                        'processing_staff_attendance', 'verified_staff'
                    }:
                        time_str = datetime.now().strftime('%H:%M:%S')
                        self.recent_tree.insert('', 'end', values=(
                            time_str,
                            getattr(track, 'state', 'tracking'),
                            track.track_id,
                            f"{getattr(track, 'confidence', 0):.2f}",
                            "Live"
                        ))
                # Limit treeview size
                children = self.recent_tree.get_children()
                if len(children) > 50:
                    self.recent_tree.delete(children[0])


        except Exception as e:
            print(f"❌ Dashboard update error: {e}")

    def reset_visit_stats(self):
        """Reset visit statistics"""
        try:
            result = messagebox.askyesno("Reset Statistics",
                                         "Are you sure you want to reset today's visit statistics?")
            if result:
                self.visit_counts = {
                    'total_today': 0,
                    'known_customers': 0,
                    'new_customers': 0,
                    'staff_checkins': 0
                }

                # Update display
                self.update_realtime_dashboard()

                reset_msg = f"🔄 STATISTICS RESET\n"
                reset_msg += f"All visit counts cleared\n"
                reset_msg += f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
                reset_msg += "=" * 30 + "\n\n"

                self.display_welcome_message(reset_msg)
                print("✅ Visit statistics reset")

        except Exception as e:
            print(f"❌ Reset statistics error: {e}")

    # def test_messages(self):
    #     """Test message display functionality with comprehensive tests"""
    #     try:
    #         # Test welcome message
    #         test_welcome = f"🧪 TEST MESSAGE #{int(time.time() % 1000)}\n"
    #         test_welcome += f"Testing welcome message display\n"
    #         test_welcome += f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
    #         test_welcome += f"Status: All systems functional\n"
    #         test_welcome += f"Photo Capture: {'Enabled' if not self.capture_btn['state'] == 'disabled' else 'Disabled'}\n"
    #         test_welcome += "=" * 40 + "\n\n"
    #
    #         self.display_welcome_message(test_welcome)
    #
    #         # Test recent detection
    #         self.add_recent_detection("Test", f"TEST_{int(time.time())}", "Test Person", 0.95)
    #
    #         # Test visit count increment
    #         self.visit_counts['total_today'] += 1
    #         self.update_realtime_dashboard()
    #
    #         print("✅ Test messages sent to dashboard")
    #         messagebox.showinfo("Test Complete", "Test messages sent successfully!")
    #
    #     except Exception as e:
    #         print(f"❌ Test messages error: {e}")
    #         messagebox.showerror("Test Error", f"Failed to send test messages: {e}")

    # Fullscreen functionality (from original code)
    def toggle_fullscreen_camera(self):
        """Toggle fullscreen camera feed display"""
        try:
            if hasattr(self, 'fullscreen_active') and self.fullscreen_active and self.fullscreen_window:
                self.exit_fullscreen()
            else:
                self.enter_fullscreen()
        except Exception as e:
            print(f"❌ Fullscreen toggle error: {e}")
            messagebox.showerror("Fullscreen Error", f"Failed to toggle fullscreen: {e}")

    def enter_fullscreen(self):
        """Enter fullscreen camera mode"""
        try:
            if not self.running or not hasattr(self, 'camera_manager'):
                messagebox.showwarning("Warning", "Please start camera recognition first")
                return

            self.fullscreen_window = tk.Toplevel(self.parent)
            self.fullscreen_window.title("Live Camera Feed - Fullscreen")
            self.fullscreen_window.attributes('-fullscreen', True)
            self.fullscreen_window.configure(bg='black')
            self.fullscreen_window.focus_set()

            # Bind escape keys
            self.fullscreen_window.bind('<Escape>', lambda e: self.exit_fullscreen())
            self.fullscreen_window.bind('<F11>', lambda e: self.exit_fullscreen())
            self.fullscreen_window.bind('<Double-Button-1>', lambda e: self.exit_fullscreen())

            self.fullscreen_label = tk.Label(self.fullscreen_window, bg='black')
            self.fullscreen_label.pack(fill=tk.BOTH, expand=True)

            instruction_label = tk.Label(self.fullscreen_window,
                                         text="Press ESC or F11 to exit fullscreen • Double-click to exit",
                                         bg='black', fg='white', font=('Arial', 12))
            instruction_label.place(relx=0.5, rely=0.95, anchor='center')

            self.fullscreen_active = True
            self._last_fullscreen_msg = 0
            self.fullscreen_video_loop()

            print("✅ Entered fullscreen camera mode")

        except Exception as e:
            print(f"❌ Enter fullscreen error: {e}")
            messagebox.showerror("Fullscreen Error", f"Failed to enter fullscreen mode: {e}")

    def exit_fullscreen(self):
        """Exit fullscreen camera mode"""
        try:
            self.fullscreen_active = False
            if hasattr(self, 'fullscreen_window') and self.fullscreen_window:
                self.fullscreen_window.destroy()
                self.fullscreen_window = None
                self.fullscreen_label = None
            print("✅ Exited fullscreen camera mode")
        except Exception as e:
            print(f"❌ Exit fullscreen error: {e}")

    def fullscreen_video_loop(self):
        """Update fullscreen video display using frames from the main pipeline"""
        try:
            if not self.fullscreen_active or not self.fullscreen_window:
                return

            now = time.time()
            if now - getattr(self, '_last_fullscreen_msg', 0) > 1.0:  # 1 s interval
                # Only update messages once per second
                self._last_fullscreen_msg = now
                if self.pending_welcome_messages:
                    self._fullscreen_msg = self.pending_welcome_messages.pop(0)

            frame = None
            tracks = []
            with self.capture_lock:
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
                    # Copy tracks for potential fullscreen overlays
                    tracks = list(getattr(self, "current_tracks", []))

            if frame is not None:
                # Get screen dimensions
                screen_width = self.fullscreen_window.winfo_screenwidth()
                screen_height = self.fullscreen_window.winfo_screenheight()

                # Calculate scaling
                h, w = frame.shape[:2]
                aspect_ratio = w / h

                if screen_width / screen_height > aspect_ratio:
                    new_height = screen_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    new_width = screen_width
                    new_height = int(new_width / aspect_ratio)

                # Resize and process
                frame_resized = cv2.resize(
                    frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                # Detection/tracking overlays come from ultra_stable_video_processing

                if getattr(self, '_fullscreen_msg', None):
                    y0 = 40
                    for i, line in enumerate(self._fullscreen_msg.splitlines()):
                        cv2.putText(frame_resized, line, (20, y0 + i * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                                    cv2.LINE_AA)

                # Convert and display
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)

                if self.fullscreen_label:
                    self.fullscreen_label.image = photo
                    self.fullscreen_label.configure(image=photo)

            # Schedule next update
            if self.fullscreen_active and self.fullscreen_window:
                self.fullscreen_window.after(33, self.fullscreen_video_loop)

        except Exception as e:
            print(f"❌ Fullscreen video loop error: {e}")
            if self.fullscreen_active and self.fullscreen_window:
                self.fullscreen_window.after(100, self.fullscreen_video_loop)

    # Utility and cleanup methods
    def update_fps(self):
        """Update FPS counter with ultra-optimization"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time

    def update_statistics_loop(self):
        """Update statistics in a separate thread"""
        while self.running:
            try:
                self.parent.after_idle(self.update_gui_statistics)
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                print(f"❌ Statistics loop error: {e}")
                time.sleep(5)

    def update_gui_statistics(self):
        """Update GUI statistics display"""
        try:
            # Update FPS
            self.fps_label.config(text=f"FPS: {self.current_fps:.1f}")

            # Update active tracks
            if hasattr(self, 'tracking_manager') and self.tracking_manager:
                active_count = len(getattr(self.tracking_manager, 'active_tracks', {}))
                self.tracks_label.config(text=f"Active Tracks: {active_count}")

        except Exception as e:
            print(f"❌ GUI statistics update error: {e}")

    def update_database_statistics(self):
        """Update database-related statistics"""
        try:
            # Initialize with zero values
            self.customers_label.config(text="Total Customers: 0")
            self.staff_label.config(text="Staff Members: 0")
        except Exception as e:
            print(f"❌ Database statistics update error: {e}")

    def clear_welcome_messages(self):
        """Clear welcome messages"""
        try:
            self.welcome_text.config(state='normal')
            self.welcome_text.delete('1.0', tk.END)
            self.welcome_text.config(state='disabled')

            clear_msg = f"🗑️ MESSAGES CLEARED\n"
            clear_msg += f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
            clear_msg += "=" * 25 + "\n\n"

            self.display_welcome_message(clear_msg)

        except Exception as e:
            print(f"❌ Clear welcome messages error: {e}")

    def stop_recognition(self):
        """Stop the ultra-optimized face recognition system"""
        try:
            print("🛑 Stopping Ultra-High-Performance Recognition System...")

            # Exit fullscreen if active
            if hasattr(self, 'fullscreen_active') and self.fullscreen_active:
                self.exit_fullscreen()

            self.running = False

            if hasattr(self, 'camera_manager'):
                self.camera_manager.stop_camera()

            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.capture_btn.config(state=tk.DISABLED)

            # Update status
            self.camera_status.config(text="📹 Camera: Disconnected", foreground="red")
            self.processing_status.config(text="🧠 Processing: Stopped", foreground="red")

            # Clear video display
            self.video_label.config(image='', text="Camera Disconnected")

            # Display shutdown message
            shutdown_msg = f"✅ SYSTEM STOPPED SUCCESSFULLY\n"
            shutdown_msg += f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
            shutdown_msg += f"Session Duration: {time.time() - self.fps_start_time:.0f} seconds\n"
            shutdown_msg += f"Total Frames Processed: {self.frame_count}\n"
            shutdown_msg += f"Photos Captured: {len(self.captured_photos)}\n"
            shutdown_msg += "=" * 50 + "\n\n"

            self.display_welcome_message(shutdown_msg)

            print("✅ Ultra-optimized system stopped successfully")

        except Exception as e:
            print(f"❌ Stop recognition error: {e}")
