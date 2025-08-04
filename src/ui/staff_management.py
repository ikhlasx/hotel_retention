# src/ui/staff_management.py - FIXED VERSION with Proper Face Capture

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import time
from core.database_manager import DatabaseManager
from core.face_engine import FaceRecognitionEngine
from core.config_manager import ConfigManager

def find_working_camera_index():
    """Find the first working camera index with backend selection"""
    import os
    os.environ['OPENCV_VIDEOIO_PRIORITY_OBSENSOR'] = '0'
    os.environ['OPENCV_VIDEOIO_PRIORITY_DIRECTSHOW'] = '1000'
    
    backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
    
    for backend in backends_to_try:
        print(f"Testing backend: {backend}")
        for index in range(5):
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        print(f"✅ Working camera found at index: {index} with backend: {backend}")
                        return index, backend
                cap.release()
            except Exception as e:
                continue
    
    print("❌ No working camera found with any backend")
    return None, None

class AddEditStaffDialog:
    def __init__(self, parent, db_manager, staff_id=None, callback=None):
        self.db_manager = db_manager
        self.staff_id = staff_id
        self.callback = callback
        self.photo_data = None
        self.multiple_photos = []
        self.photo_embeddings = []
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Add Staff" if staff_id is None else "Edit Staff")
        
        # **FIXED: Even larger window for proper face display**
        self.dialog.geometry("900x1000")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.setup_gui()
        
        if staff_id:
            self.load_staff_data()

    def setup_gui(self):
        # **FIXED: Better scrollable layout**
        main_canvas = tk.Canvas(self.dialog)
        scrollbar = ttk.Scrollbar(self.dialog, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        frame = ttk.Frame(scrollable_frame, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        # Staff Information Section
        info_frame = ttk.LabelFrame(frame, text="Staff Information", padding=15)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        # Staff ID
        ttk.Label(info_frame, text="Staff ID:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=2)
        self.staff_id_var = tk.StringVar()
        staff_id_entry = ttk.Entry(info_frame, textvariable=self.staff_id_var, width=30, font=('Arial', 12))
        staff_id_entry.pack(fill=tk.X, pady=2)

        # Name
        ttk.Label(info_frame, text="Full Name:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(10, 2))
        self.name_var = tk.StringVar()
        name_entry = ttk.Entry(info_frame, textvariable=self.name_var, width=30, font=('Arial', 12))
        name_entry.pack(fill=tk.X, pady=2)

        # Department
        ttk.Label(info_frame, text="Department:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(10, 2))
        self.department_var = tk.StringVar()
        department_combo = ttk.Combobox(info_frame, textvariable=self.department_var,
                                       values=["Reception", "Housekeeping", "Security", "Management", 
                                              "Kitchen", "Maintenance", "Sales", "Admin"],
                                       font=('Arial', 12))
        department_combo.pack(fill=tk.X, pady=2)

        # **FIXED: Much Larger Photo Section with proper dimensions**
        photo_frame = ttk.LabelFrame(frame, text="Staff Photo - High Quality Capture", padding=15)
        photo_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # **FIXED: Very large photo display (500x500) for full face visibility**
        self.photo_label = tk.Label(photo_frame, text="No photo selected", 
                                   bg='lightgray', width=60, height=30,
                                   font=('Arial', 14), fg='gray')
        self.photo_label.pack(pady=10)
        
        # Photo info display
        self.photo_info_label = tk.Label(photo_frame, text="", 
                                        font=('Arial', 11), fg='blue')
        self.photo_info_label.pack(pady=5)

        # **FIXED: Enhanced photo buttons**
        photo_button_frame = ttk.Frame(photo_frame)
        photo_button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(photo_button_frame, text="📁 Browse Photo", 
                  command=self.browse_photo).pack(side=tk.LEFT, padx=5)
        ttk.Button(photo_button_frame, text="📸 Take High-Quality Photos", 
                  command=self.take_photo).pack(side=tk.LEFT, padx=5)
        ttk.Button(photo_button_frame, text="🔄 Clear Photo", 
                  command=self.clear_photo).pack(side=tk.LEFT, padx=5)

        # **FIXED: Enhanced photo quality indicator**
        self.quality_frame = ttk.Frame(photo_frame)
        self.quality_frame.pack(fill=tk.X, pady=5)
        
        self.quality_label = tk.Label(self.quality_frame, text="", font=('Arial', 11))
        self.quality_label.pack()

        # Face detection preview frame
        self.preview_frame = ttk.LabelFrame(photo_frame, text="Face Detection Preview", padding=10)
        self.preview_frame.pack(fill=tk.X, pady=5)
        
        self.face_preview_label = tk.Label(self.preview_frame, text="Face area will be shown here after photo capture",
                                          bg='white', height=8, font=('Arial', 10))
        self.face_preview_label.pack(pady=5)

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=20)

        ttk.Button(button_frame, text="💾 Save Staff", 
                  command=self.save_staff).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="❌ Cancel", 
                  command=self.dialog.destroy).pack(side=tk.RIGHT)

    def browse_photo(self):
        """Browse for photo file with enhanced display and validation"""
        file_path = filedialog.askopenfilename(
            title="Select Staff Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )

        if file_path:
            try:
                # **FIXED: Load and validate image properly**
                image = Image.open(file_path)
                
                # Store original photo data
                self.photo_data = cv2.imread(file_path)
                
                # **FIXED: Display much larger preview (500x500 max)**
                display_image = image.copy()
                display_image.thumbnail((500, 500), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(display_image)
                
                self.photo_label.config(image=photo, text="")
                self.photo_label.image = photo
                
                # **FIXED: Show detailed image information**
                width, height = image.size
                file_size = os.path.getsize(file_path) / 1024  # KB
                self.photo_info_label.config(
                    text=f"📸 Image: {width}x{height} pixels, {file_size:.1f} KB | Resolution: {'High' if min(width, height) > 640 else 'Medium'}"
                )
                
                # **FIXED: Enhanced face validation with preview**
                self.validate_and_preview_face()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def take_photo(self):
        """Take multiple high-quality photos using enhanced camera capture"""
        HighQualityPhotoCaptureDialog(self.dialog, callback=self.set_photo_data)

    def clear_photo(self):
        """Clear the selected photo"""
        self.photo_data = None
        self.multiple_photos = []
        self.photo_embeddings = []
        
        self.photo_label.config(image='', text="No photo selected")
        self.photo_label.image = None
        self.photo_info_label.config(text="")
        self.quality_label.config(text="")
        self.face_preview_label.config(image='', text="Face area will be shown here after photo capture")
        self.face_preview_label.image = None

    def validate_and_preview_face(self):
        """**FIXED: Enhanced face validation with proper face preview**"""
        if self.photo_data is None:
            return
        
        try:
            # Test face detection on the photo
            face_engine = FaceRecognitionEngine()
            detections = face_engine.detect_faces(self.photo_data)
            
            if not detections:
                self.quality_label.config(text="❌ No face detected. Please use a clear face photo.", 
                                        fg='red')
                self.face_preview_label.config(image='', text="No face detected in image")
                self.face_preview_label.image = None
                
            elif len(detections) > 1:
                self.quality_label.config(text="⚠️ Multiple faces detected. Please use single face photo.", 
                                        fg='orange')
                # Show all detected faces
                self.show_multiple_faces_preview(detections)
                
            else:
                detection = detections[0]
                confidence = detection['confidence']
                bbox = detection['bbox']
                
                # **FIXED: Show face area preview**
                self.show_face_preview(bbox, confidence)
                
                if confidence > 0.8:
                    self.quality_label.config(text="✅ Excellent photo quality for recognition!", 
                                            fg='green')
                elif confidence > 0.6:
                    self.quality_label.config(text="✅ Good photo quality for recognition.", 
                                            fg='blue')
                else:
                    self.quality_label.config(text="⚠️ Photo quality could be better. Consider retaking.", 
                                            fg='orange')
                    
        except Exception as e:
            self.quality_label.config(text=f"❌ Error validating photo: {e}", fg='red')

    def show_face_preview(self, bbox, confidence):
        """**FIXED: Show detected face area in preview with proper cropping**"""
        try:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # **FIXED: Add padding around face to show full face context**
            padding = 50  # pixels
            h, w = self.photo_data.shape[:2]
            
            # Expand bounding box with padding
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(w, x2 + padding)
            y2_padded = min(h, y2 + padding)
            
            # **FIXED: Extract face area with padding**
            face_area = self.photo_data[y1_padded:y2_padded, x1_padded:x2_padded]
            
            if face_area.size > 0:
                # Convert to RGB and resize for preview
                face_rgb = cv2.cvtColor(face_area, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                
                # **FIXED: Resize to good preview size**
                face_pil.thumbnail((200, 200), Image.Resampling.LANCZOS)
                face_photo = ImageTk.PhotoImage(face_pil)
                
                self.face_preview_label.config(image=face_photo, text="")
                self.face_preview_label.image = face_photo
                
                # Update info
                face_width = x2 - x1
                face_height = y2 - y1
                self.face_preview_label.config(text=f"Face: {face_width}x{face_height}px, Confidence: {confidence:.1%}")
            else:
                self.face_preview_label.config(text="Face area too small to preview")
                
        except Exception as e:
            print(f"Face preview error: {e}")
            self.face_preview_label.config(text="Error showing face preview")

    def show_multiple_faces_preview(self, detections):
        """Show preview when multiple faces are detected"""
        face_count = len(detections)
        self.face_preview_label.config(text=f"⚠️ {face_count} faces detected - please use single face photo")

    def set_photo_data(self, photo_data):
        """**FIXED: Enhanced photo data handling with proper validation**"""
        try:
            if isinstance(photo_data, dict):
                # Multiple photos captured (recommended)
                if 'photos' in photo_data and photo_data['photos']:
                    self.multiple_photos = photo_data['photos']
                    self.photo_embeddings = photo_data.get('embeddings', [])
                    
                    # Use the best quality photo for display
                    best_photo = self.select_best_photo(self.multiple_photos)
                    self.photo_data = best_photo
                    
                    # **FIXED: Display large, high-quality preview**
                    rgb_image = cv2.cvtColor(self.photo_data, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    
                    # **FIXED: Display at very good size (500x500 max)**
                    pil_image.thumbnail((500, 500), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    self.photo_label.config(image=photo, text="")
                    self.photo_label.image = photo
                    
                    # **FIXED: Show comprehensive capture information**
                    photo_count = len(self.multiple_photos)
                    h, w = self.photo_data.shape[:2]
                    avg_quality = np.mean([emb.mean() for emb in self.photo_embeddings]) if self.photo_embeddings else 0
                    
                    self.photo_info_label.config(
                        text=f"📸 {photo_count} photos captured | Resolution: {w}x{h} | Avg Quality: {avg_quality:.3f}"
                    )
                    
                    self.quality_label.config(
                        text=f"✅ {photo_count} high-quality photos captured for enhanced recognition!", 
                        fg='green'
                    )
                    
                    # **FIXED: Show face preview**
                    self.validate_and_preview_face()
                    
                    print(f"✅ Received {photo_count} high-quality photos for staff registration")
            else:
                # Single photo (legacy support)
                if photo_data is not None and isinstance(photo_data, np.ndarray):
                    self.photo_data = photo_data
                    
                    # **FIXED: Display large, high-quality preview**
                    rgb_image = cv2.cvtColor(photo_data, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    pil_image.thumbnail((500, 500), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    self.photo_label.config(image=photo, text="")
                    self.photo_label.image = photo
                    
                    # Show photo info
                    h, w = photo_data.shape[:2]
                    self.photo_info_label.config(text=f"📸 Resolution: {w}x{h} | Single photo captured")
                    
                    # **FIXED: Validate and show face preview**
                    self.validate_and_preview_face()
                else:
                    messagebox.showerror("Error", "Invalid photo data received from camera")
                    
        except Exception as e:
            print(f"❌ Photo data processing error: {e}")
            messagebox.showerror("Error", f"Failed to process photo data: {e}")

    def select_best_photo(self, photos):
        """**FIXED: Select the best quality photo from multiple captures**"""
        try:
            face_engine = FaceRecognitionEngine()
            best_photo = photos[0]
            best_score = 0
            
            for photo in photos:
                detections = face_engine.detect_faces(photo)
                if detections:
                    # Score based on confidence and face size
                    confidence = detections[0]['confidence']
                    bbox = detections[0]['bbox']
                    face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    
                    # Combined score: confidence * face_size_factor
                    score = confidence * min(1.0, face_area / 10000)  # Normalize face area
                    
                    if score > best_score:
                        best_score = score
                        best_photo = photo
            
            print(f"✅ Selected best photo with score: {best_score:.3f}")
            return best_photo
        except Exception as e:
            print(f"Error selecting best photo: {e}")
            return photos[0] if photos else None

    def save_staff(self):
        """**FIXED: Enhanced staff saving with better validation**"""
        staff_id = self.staff_id_var.get().strip()
        name = self.name_var.get().strip()
        department = self.department_var.get().strip()

        if not staff_id or not name:
            messagebox.showerror("Error", "Please fill in Staff ID and Name")
            return

        if self.photo_data is None:
            messagebox.showerror("Error", "Please provide a staff photo")
            return

        try:
            # **FIXED: Enhanced face processing with multiple photos**
            face_engine = FaceRecognitionEngine()
            
            if self.multiple_photos and len(self.multiple_photos) >= 3:
                # Use multiple photos for better embedding
                all_embeddings = []
                valid_photos = 0
                
                for photo in self.multiple_photos:
                    detections = face_engine.detect_faces(photo)
                    if detections and len(detections) == 1:  # Ensure single face
                        all_embeddings.append(detections[0]['embedding'])
                        valid_photos += 1
                
                if not all_embeddings:
                    messagebox.showerror("Error", "No suitable faces detected in captured photos")
                    return
                
                # **FIXED: Create averaged embedding for better recognition**
                embedding = np.mean(all_embeddings, axis=0)
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
                
                print(f"✅ Created averaged embedding from {valid_photos} photos")
                
            else:
                # Single photo processing
                detections = face_engine.detect_faces(self.photo_data)
                
                if not detections:
                    messagebox.showerror("Error", 
                                       "❌ No face detected in the photo.\n\n"
                                       "Please ensure:\n"
                                       "• Face is clearly visible and not cropped\n"
                                       "• Good lighting conditions\n"
                                       "• Camera is properly focused\n"
                                       "• Face takes up significant portion of image")
                    return

                if len(detections) > 1:
                    messagebox.showerror("Error", 
                                       "❌ Multiple faces detected in photo.\n\n"
                                       "Please use a photo with only one face.")
                    return

                embedding = detections[0]['embedding']

            # **FIXED: Enhanced embedding quality validation**
            embedding_quality = np.linalg.norm(embedding)
            
            if embedding_quality < 0.5:  # Very low quality threshold
                messagebox.showwarning("Low Quality Warning",
                                     f"⚠️ Face embedding quality is low ({embedding_quality:.3f}).\n\n"
                                     f"This might affect recognition accuracy.\n"
                                     f"Consider retaking photos with better lighting.")

            # Save to database
            success = self.db_manager.add_staff_member(staff_id, name, department, embedding, self.photo_data)

            if success:
                # **FIXED: Add to face engine database immediately**
                face_engine.staff_database[staff_id] = embedding
                
                messagebox.showinfo("Success", 
                                  f"✅ Staff member saved successfully!\n\n"
                                  f"Staff ID: {staff_id}\n"
                                  f"Name: {name}\n"
                                  f"Department: {department}\n"
                                  f"Embedding Quality: {embedding_quality:.3f}\n"
                                  f"Photos Used: {len(self.multiple_photos) if self.multiple_photos else 1}")
                
                if self.callback:
                    self.callback()
                self.dialog.destroy()
            else:
                messagebox.showerror("Error", "Failed to save staff member to database")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process staff photo: {e}")

    def load_staff_data(self):
        """Load existing staff data for editing"""
        staff_info = self.db_manager.get_staff_info(self.staff_id)
        if staff_info:
            self.staff_id_var.set(staff_info['staff_id'])
            self.name_var.set(staff_info['name'])
            self.department_var.set(staff_info['department'] or '')


class HighQualityPhotoCaptureDialog:
    """**FIXED: High-quality photo capture dialog with proper face detection**"""
    
    def __init__(self, parent, callback=None):
        self.callback = callback
        self.cap = None
        self.running = False
        
        # Initialize enhanced photo storage
        self.captured_photos = []
        self.photo_embeddings = []
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("📸 High-Quality Staff Photo Capture")
        
        # **FIXED: Very large dialog for proper photo capture**
        self.dialog.geometry("1200x900")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.setup_enhanced_gui()
        self.start_camera()

    def setup_enhanced_gui(self):
        """**FIXED: Setup enhanced GUI with very large video display**"""
        
        # **FIXED: Very large video frame**
        self.video_frame = tk.Frame(self.dialog, bg='black', relief='solid', bd=2)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # **FIXED: Very large video label for full face visibility**
        self.video_label = tk.Label(self.video_frame, bg='black', 
                                   text="🔌 Connecting to camera...\n\nPlease wait for high-quality video...",
                                   font=('Arial', 18), fg='white')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # **FIXED: Enhanced info panel**
        info_frame = tk.Frame(self.dialog, bg='lightgreen', relief='raised', bd=3)
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 15))

        # Photo count and instructions
        self.photo_count_label = tk.Label(info_frame, text="📸 Photos Captured: 0/5",
                                         font=('Arial', 16, 'bold'), bg='lightgreen')
        self.photo_count_label.pack(side=tk.LEFT, padx=15, pady=15)

        instruction_label = tk.Label(info_frame,
                                   text="💡 Keep your full face visible in the camera preview",
                                   font=('Arial', 13, 'bold'), bg='lightgreen', fg='darkgreen')
        instruction_label.pack(side=tk.RIGHT, padx=15, pady=15)

        # **FIXED: Enhanced face detection status with better visibility**
        self.detection_status = tk.Label(info_frame, text="🔍 Face Detection: Initializing...",
                                        font=('Arial', 12, 'bold'), bg='lightgreen', fg='darkblue')
        self.detection_status.pack(pady=8)

        # **FIXED: Large control buttons**
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=15)

        # Very large capture button
        self.capture_btn = tk.Button(button_frame, text="📸 CAPTURE HIGH-QUALITY PHOTO",
                                    command=self.capture_multiple_photos,
                                    font=('Arial', 18, 'bold'), bg='green', fg='white',
                                    height=3, width=25)
        self.capture_btn.pack(side=tk.LEFT, padx=15)

        # Clear photos button
        self.retake_btn = tk.Button(button_frame, text="🔄 Clear All Photos",
                                   command=self.clear_photos,
                                   font=('Arial', 14), bg='orange', fg='white',
                                   height=3, width=18, state=tk.DISABLED)
        self.retake_btn.pack(side=tk.LEFT, padx=15)

        # Finish button
        self.finish_btn = tk.Button(button_frame, text="✅ FINISH & SAVE",
                                   command=self.finish_capture,
                                   font=('Arial', 18, 'bold'), bg='blue', fg='white',
                                   height=3, width=20, state=tk.DISABLED)
        self.finish_btn.pack(side=tk.LEFT, padx=15)

        # Cancel button
        tk.Button(button_frame, text="❌ Cancel", command=self.close_dialog,
                 font=('Arial', 14), height=3, width=15).pack(side=tk.RIGHT, padx=15)

    def start_camera(self):
        """**FIXED: Start camera with highest quality settings**"""
        try:
            config = ConfigManager()
            camera_settings = config.get_camera_settings()
            
            # **FIXED: Ultra-high quality camera settings**
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|rtsp_flags;prefer_tcp|fflags;nobuffer|'
                'flags;low_delay|fflags;flush_packets'
            )

            if camera_settings.get('source_type') in ['rtsp', 'ip']:
                camera_source = camera_settings.get('rtsp_url')
                print(f"📹 Using RTSP camera for high-quality capture: {camera_source}")
                
                self.cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
                
                if self.cap.isOpened():
                    # **FIXED: High-quality camera properties**
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # **FIXED: Higher resolution**
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # **FIXED: Higher resolution**
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)        # **FIXED: Enable autofocus**
                    
                    # Test connection
                    for attempt in range(3):
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            print(f"✅ High-quality RTSP camera connected successfully")
                            self.running = True
                            self.update_video()
                            return
                        time.sleep(0.5)
                    
                    self.cap.release()
                    print("❌ RTSP camera: Cannot read frames")

            # **FIXED: USB fallback with high quality**
            print("📷 Trying USB cameras with high-quality settings...")
            camera_index, backend = find_working_camera_index()
            
            if camera_index is not None:
                self.cap = cv2.VideoCapture(camera_index, backend)
                
                if self.cap.isOpened():
                    # **FIXED: High-quality USB camera settings**
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)    # **FIXED: Higher resolution**
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)   # **FIXED: Higher resolution**
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)         # **FIXED: Enable autofocus**
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # **FIXED: Better exposure**
                    
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"✅ High-quality USB camera connected: index {camera_index}")
                        self.running = True
                        self.update_video()
                        return

            # Enhanced error message
            messagebox.showerror("Camera Error",
                               "❌ Cannot access camera for high-quality photo capture.\n\n"
                               "Troubleshooting:\n"
                               "• Ensure camera supports high resolution (1080p+)\n"
                               "• Close other applications using the camera\n"
                               "• Check camera permissions\n"
                               "• Verify RTSP URL and credentials\n"
                               "• Try different USB ports for webcams\n"
                               "• Ensure good lighting conditions")
            self.dialog.destroy()

        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            messagebox.showerror("Camera Error", f"Failed to initialize camera: {e}")
            self.dialog.destroy()

    def update_video(self):
        """**FIXED: Enhanced video display with face detection feedback**"""
        if not self.running or not self.cap or not self.cap.isOpened():
            return

        try:
            # Skip frames for reduced latency
            for _ in range(1):  # **FIXED: Skip fewer frames for better quality**
                ret, _ = self.cap.read()
                if not ret:
                    break

            # Get the latest frame
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                # **FIXED: Enhanced face detection feedback with proper bounding boxes**
                try:
                    face_engine = FaceRecognitionEngine()
                    detections = face_engine.detect_faces(frame)
                    
                    # **FIXED: Draw face detection boxes on the frame**
                    display_frame = frame.copy()
                    
                    if detections:
                        face_count = len(detections)
                        
                        for detection in detections:
                            bbox = detection['bbox']
                            confidence = detection['confidence']
                            
                            # **FIXED: Draw bounding box with padding visualization**
                            x1, y1, x2, y2 = [int(coord) for coord in bbox]
                            
                            # Draw main face box
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            
                            # **FIXED: Draw padding box to show full capture area**
                            padding = 50
                            h, w = frame.shape[:2]
                            x1_padded = max(0, x1 - padding)
                            y1_padded = max(0, y1 - padding)
                            x2_padded = min(w, x2 + padding)
                            y2_padded = min(h, y2 + padding)
                            
                            cv2.rectangle(display_frame, (x1_padded, y1_padded), (x2_padded, y2_padded), (255, 255, 0), 2)
                            
                            # **FIXED: Add confidence text**
                            cv2.putText(display_frame, f'Face: {confidence:.1%}', (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        if face_count == 1:
                            confidence = detections[0]['confidence']
                            if confidence > 0.8:
                                self.detection_status.config(text="✅ Excellent face detected - Ready to capture!",
                                                            fg='darkgreen')
                            elif confidence > 0.6:
                                self.detection_status.config(text="✅ Good face detected - Ready to capture",
                                                            fg='green')
                            else:
                                self.detection_status.config(text="⚠️ Face detected but quality could be better",
                                                            fg='orange')
                        else:
                            self.detection_status.config(text=f"⚠️ {face_count} faces detected - Use single face only",
                                                        fg='orange')
                    else:
                        self.detection_status.config(text="🔍 No face detected - Please position face in center of camera",
                                                    fg='red')
                except Exception as face_error:
                    print(f"Face detection error: {face_error}")
                    display_frame = frame
                    self.detection_status.config(text="🔍 Face detection active", fg='blue')

                # **FIXED: Very large video display**
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # **FIXED: Get dialog size for optimal large display**
                dialog_width = self.dialog.winfo_width()
                dialog_height = self.dialog.winfo_height()
                
                # **FIXED: Calculate optimal display size (use most of the dialog space)**
                max_width = min(dialog_width - 60, 1100)    # **FIXED: Larger display**
                max_height = min(dialog_height - 300, 700)   # **FIXED: Larger display**
                
                h, w = frame_rgb.shape[:2]
                scale = min(max_width / w, max_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                
                # Convert to PhotoImage
                image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image)
                
                # Update display
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo
                
                # Store current frame
                self.current_frame = frame

        except Exception as e:
            print(f"❌ Video update error: {e}")

        # Schedule next update with optimal frequency
        if self.running:
            self.dialog.after(33, self.update_video)  # ~30 FPS

    def capture_multiple_photos(self):
        """**FIXED: Enhanced multi-photo capture with proper validation**"""
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            
            # Validate frame
            if not isinstance(self.current_frame, np.ndarray) or self.current_frame.size == 0:
                messagebox.showerror("Error", "Invalid camera frame")
                return

            # **FIXED: Enhanced face validation with detailed feedback**
            try:
                face_engine = FaceRecognitionEngine()
                detections = face_engine.detect_faces(self.current_frame)
                
                if not detections:
                    messagebox.showwarning("No Face Detected",
                                         "❌ No face detected in current frame.\n\n"
                                         "Please ensure:\n"
                                         "• Your full face is clearly visible\n"
                                         "• Good lighting conditions\n"
                                         "• Face is centered in camera view\n"
                                         "• Camera is properly focused")
                    return
                
                if len(detections) > 1:
                    messagebox.showwarning("Multiple Faces",
                                         "⚠️ Multiple faces detected.\n\n"
                                         "Please ensure only one person is visible in the camera.")
                    return

                # **FIXED: Validate face quality and size**
                detection = detections[0]
                confidence = detection['confidence']
                bbox = detection['bbox']
                
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                face_area = face_width * face_height
                
                if face_area < 10000:  # **FIXED: Minimum face size validation**
                    messagebox.showwarning("Face Too Small",
                                         f"⚠️ Face appears too small in the image.\n\n"
                                         f"Current size: {face_width:.0f}x{face_height:.0f} pixels\n"
                                         f"Please move closer to the camera.")
                    return
                
                if confidence < 0.5:  # **FIXED: Minimum confidence validation**
                    messagebox.showwarning("Low Quality",
                                         f"⚠️ Face detection confidence is low ({confidence:.1%}).\n\n"
                                         f"Please improve lighting or camera focus.")
                    return

                # **FIXED: Store the high-quality photo**
                photo_copy = self.current_frame.copy()
                self.captured_photos.append(photo_copy)
                
                # Store embedding
                self.photo_embeddings.append(detection['embedding'])
                
                # **FIXED: Update UI with detailed progress**
                photo_count = len(self.captured_photos)
                self.photo_count_label.config(text=f"📸 Photos Captured: {photo_count}/5")
                
                # **FIXED: Enhanced capture instructions**
                instructions = [
                    "Perfect! Now turn your head slightly RIGHT → (keep full face visible)",
                    "Excellent! Now turn your head slightly LEFT ← (keep full face visible)", 
                    "Great! Now tilt your head slightly UP ↑ (keep full face visible)",
                    "Amazing! Now look slightly DOWN ↓ (keep full face visible)",
                    "Perfect! All high-quality photos captured successfully! ✅"
                ]
                
                if photo_count <= 5:
                    instruction = instructions[min(photo_count - 1, 4)]
                    messagebox.showinfo("High-Quality Photo Captured",
                                      f"✅ Photo {photo_count}/5 captured successfully!\n\n"
                                      f"Quality: {confidence:.1%}\n"
                                      f"Face Size: {face_width:.0f}x{face_height:.0f} pixels\n\n"
                                      f"Next: {instruction}")

                # **FIXED: Enable/disable buttons based on progress**
                if photo_count >= 3:  # Minimum for good recognition
                    self.finish_btn.config(state=tk.NORMAL, bg='blue')
                    
                if photo_count >= 5:  # Maximum photos
                    self.capture_btn.config(state=tk.DISABLED, bg='gray', 
                                          text="📸 Max High-Quality Photos Reached")
                    messagebox.showinfo("Capture Complete",
                                      "🎉 All 5 high-quality photos captured successfully!\n\n"
                                      f"Average quality: {np.mean([d['confidence'] for d in detections]):.1%}\n\n"
                                      "Click 'FINISH & SAVE' to continue with staff registration.")

                self.retake_btn.config(state=tk.NORMAL, bg='orange')

            except Exception as e:
                print(f"❌ Face processing error: {e}")
                messagebox.showerror("Error", f"Failed to process photo: {e}")
        else:
            messagebox.showwarning("Warning", "No camera frame available. Please wait for camera to initialize.")

    def clear_photos(self):
        """Clear all captured photos"""
        if len(self.captured_photos) > 0:
            if messagebox.askyesno("Clear Photos",
                                 f"🗑️ Clear all {len(self.captured_photos)} high-quality photos?\n\n"
                                 f"You'll need to capture them again."):
                
                # Reset everything
                self.captured_photos.clear()
                self.photo_embeddings.clear()
                
                # Reset UI
                self.photo_count_label.config(text="📸 Photos Captured: 0/5")
                self.capture_btn.config(state=tk.NORMAL, bg='green', text="📸 CAPTURE HIGH-QUALITY PHOTO")
                self.retake_btn.config(state=tk.DISABLED, bg='gray')
                self.finish_btn.config(state=tk.DISABLED, bg='gray')
                
                messagebox.showinfo("Photos Cleared",
                                  "🔄 All photos cleared successfully!\n\n"
                                  "Ready to capture new high-quality photos.")
        else:
            messagebox.showinfo("No Photos", "No photos to clear.")

    def finish_capture(self):
        """**FIXED: Finish photo capture with enhanced validation**"""
        if len(self.captured_photos) >= 3:
            # **FIXED: Comprehensive photo validation**
            valid_photos = []
            valid_embeddings = []
            quality_scores = []
            
            for i, photo in enumerate(self.captured_photos):
                if isinstance(photo, np.ndarray) and photo.size > 0:
                    # **FIXED: Additional quality check**
                    h, w = photo.shape[:2]
                    if h >= 480 and w >= 640:  # Minimum resolution check
                        valid_photos.append(photo)
                        if i < len(self.photo_embeddings):
                            embedding = self.photo_embeddings[i]
                            valid_embeddings.append(embedding)
                            quality_scores.append(np.linalg.norm(embedding))

            if valid_photos:
                if self.callback:
                    # **FIXED: Return enhanced data structure with quality metrics**
                    callback_data = {
                        'photos': valid_photos,
                        'embeddings': valid_embeddings,
                        'count': len(valid_photos),
                        'quality_scores': quality_scores,
                        'average_quality': np.mean(quality_scores) if quality_scores else 0,
                        'resolution': f"{valid_photos[0].shape[1]}x{valid_photos[0].shape[0]}"
                    }
                    self.callback(callback_data)
                    
                self.close_dialog()
            else:
                messagebox.showerror("Error", "No valid high-quality photos found")
        else:
            messagebox.showwarning("Insufficient Photos",
                                 f"❌ Please capture at least 3 high-quality photos for reliable recognition.\n\n"
                                 f"Currently captured: {len(self.captured_photos)} photos\n"
                                 f"Recommended: 5 photos from different angles\n\n"
                                 f"High-quality photos ensure better staff recognition accuracy.")

    def close_dialog(self):
        """Close the dialog and cleanup"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.dialog.destroy()


# **FIXED: Staff Management Window with enhanced layout**
class StaffManagementWindow:
    def __init__(self, parent):
        self.parent = parent
        self.db_manager = DatabaseManager()
        
        self.window = tk.Toplevel(parent)
        self.window.title("👥 Staff Management System - Enhanced Quality")
        self.window.geometry("1100x800")  # **FIXED: Even larger window**
        self.window.transient(parent)
        
        self.setup_gui()
        self.load_staff_list()

    def setup_gui(self):
        # **FIXED: Enhanced main frame layout**
        main_frame = ttk.Frame(self.window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # **FIXED: Enhanced header with better statistics**
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(header_frame, text="👥 Staff Management System", 
                              font=('Arial', 18, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        self.stats_label = tk.Label(header_frame, text="", font=('Arial', 12))
        self.stats_label.pack(side=tk.RIGHT)

        # **FIXED: Enhanced staff list with better columns**
        list_frame = ttk.LabelFrame(main_frame, text="📋 Staff Members Database", padding=15)
        list_frame.pack(fill=tk.BOTH, expand=True)

        # **FIXED: Enhanced treeview with more information**
        columns = ('ID', 'Name', 'Department', 'Added Date', 'Status', 'Quality')
        self.staff_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=25)

        # **FIXED: Better column configuration**
        column_widths = {'ID': 120, 'Name': 200, 'Department': 150, 
                        'Added Date': 150, 'Status': 100, 'Quality': 100}
        
        for col in columns:
            self.staff_tree.heading(col, text=col)
            self.staff_tree.column(col, width=column_widths.get(col, 100))

        # **FIXED: Enhanced scrollbars**
        v_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.staff_tree.yview)
        h_scrollbar = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.staff_tree.xview)
        
        self.staff_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.staff_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # **FIXED: Enhanced action buttons**
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)

        ttk.Button(button_frame, text="➕ Add New Staff", 
                  command=self.add_staff).pack(side=tk.LEFT, padx=8)
        ttk.Button(button_frame, text="✏️ Edit Staff", 
                  command=self.edit_staff).pack(side=tk.LEFT, padx=8)
        ttk.Button(button_frame, text="🗑️ Delete Staff", 
                  command=self.delete_staff).pack(side=tk.LEFT, padx=8)
        ttk.Button(button_frame, text="🔄 Refresh List", 
                  command=self.load_staff_list).pack(side=tk.LEFT, padx=8)
        
        # Spacer
        tk.Frame(button_frame, width=30).pack(side=tk.LEFT)
        
        ttk.Button(button_frame, text="📊 View Details", 
                  command=self.view_staff_details).pack(side=tk.LEFT, padx=8)
        ttk.Button(button_frame, text="🔍 Test Recognition", 
                  command=self.test_staff_recognition).pack(side=tk.LEFT, padx=8)
        
        ttk.Button(button_frame, text="❌ Close", 
                  command=self.window.destroy).pack(side=tk.RIGHT, padx=8)

    def load_staff_list(self):
        """**FIXED: Load staff list with enhanced information**"""
        # Clear existing items
        for item in self.staff_tree.get_children():
            self.staff_tree.delete(item)

        # Load staff from database
        staff_members = self.db_manager.get_all_staff()
        
        for staff in staff_members:
            status = "✅ Active" if staff.get('is_active', True) else "❌ Inactive"
            
            # **FIXED: Estimate quality based on embedding if available**
            quality = "Unknown"
            if staff.get('embedding'):
                try:
                    import pickle
                    embedding = pickle.loads(staff['embedding'])
                    embedding_norm = np.linalg.norm(embedding)
                    if embedding_norm > 0.8:
                        quality = "✅ High"
                    elif embedding_norm > 0.6:
                        quality = "🟡 Medium"
                    else:
                        quality = "🟠 Low"
                except:
                    quality = "❓ Error"
            
            self.staff_tree.insert('', tk.END, values=(
                staff['staff_id'],
                staff['name'],
                staff['department'] or 'N/A',
                staff['added_date'],
                status,
                quality
            ))
        
        # **FIXED: Enhanced statistics**
        total_staff = len(staff_members)
        active_staff = sum(1 for s in staff_members if s.get('is_active', True))
        with_photos = sum(1 for s in staff_members if s.get('embedding'))
        
        self.stats_label.config(text=f"Total: {total_staff} | Active: {active_staff} | With Photos: {with_photos}")

    def add_staff(self):
        """Add new staff member"""
        AddEditStaffDialog(self.window, self.db_manager, callback=self.load_staff_list)

    def edit_staff(self):
        """Edit selected staff member"""
        selection = self.staff_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a staff member to edit")
            return

        item = self.staff_tree.item(selection[0])
        staff_id = item['values'][0]
        AddEditStaffDialog(self.window, self.db_manager, staff_id=staff_id, callback=self.load_staff_list)

    def delete_staff(self):
        """Delete selected staff member"""
        selection = self.staff_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a staff member to delete")
            return

        item = self.staff_tree.item(selection[0])
        staff_id = item['values'][0]
        staff_name = item['values'][1]

        result = messagebox.askyesno("Confirm Delete",
                                   f"🗑️ Delete staff member?\n\n"
                                   f"Staff ID: {staff_id}\n"
                                   f"Name: {staff_name}\n\n"
                                   f"⚠️ This action cannot be undone!")

        if result:
            success = self.db_manager.delete_staff_member(staff_id)
            if success:
                messagebox.showinfo("Success", f"✅ Staff member '{staff_name}' deleted successfully")
                self.load_staff_list()
            else:
                messagebox.showerror("Error", "❌ Failed to delete staff member")

    def view_staff_details(self):
        """**FIXED: Enhanced staff details view**"""
        selection = self.staff_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a staff member to view details")
            return

        item = self.staff_tree.item(selection[0])
        staff_id = item['values'][0]
        
        # Get detailed staff info
        staff_info = self.db_manager.get_staff_info(staff_id)
        if staff_info:
            # **FIXED: Enhanced details with quality information**
            embedding_info = "Not available"
            try:
                staff_data = self.db_manager.get_all_staff()
                staff_record = next((s for s in staff_data if s['staff_id'] == staff_id), None)
                if staff_record and staff_record.get('embedding'):
                    import pickle
                    embedding = pickle.loads(staff_record['embedding'])
                    embedding_norm = np.linalg.norm(embedding)
                    embedding_info = f"Quality: {embedding_norm:.3f} (Norm)"
            except:
                embedding_info = "Error reading embedding"
            
            details = f"""
📋 DETAILED STAFF INFORMATION

Staff ID: {staff_info['staff_id']}
Name: {staff_info['name']}
Department: {staff_info['department'] or 'N/A'}

🔍 RECOGNITION SYSTEM INFO
Database Status: ✅ Found
Photo Quality: {embedding_info}
Recognition Ready: {'✅ Yes' if staff_record and staff_record.get('embedding') else '❌ No photo'}

💡 NOTES
• High-quality photos improve recognition accuracy
• Multiple angle photos provide better results
• Good lighting conditions are essential
"""
            messagebox.showinfo("Staff Details", details)

    def test_staff_recognition(self):
        """**FIXED: Test staff recognition functionality**"""
        selection = self.staff_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a staff member to test recognition")
            return

        item = self.staff_tree.item(selection[0])
        staff_id = item['values'][0]
        staff_name = item['values'][1]
        
        messagebox.showinfo("Recognition Test",
                          f"🔍 Recognition Test for {staff_name}\n\n"
                          f"Staff ID: {staff_id}\n\n"
                          f"To test recognition:\n"
                          f"1. Start the main recognition system\n"
                          f"2. Position {staff_name} in front of camera\n"
                          f"3. System should identify them as staff\n\n"
                          f"If not recognized, consider retaking photos with:\n"
                          f"• Better lighting conditions\n"
                          f"• Multiple angles (front, slight left/right)\n"
                          f"• Higher resolution camera\n"
                          f"• Clear, unobstructed face view")
