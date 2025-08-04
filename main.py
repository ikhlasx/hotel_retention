# main.py - Complete Hotel Face Recognition System (Root Directory)

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import json
from datetime import datetime, time
import logging

# CRITICAL: Set environment variables BEFORE any imports
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;tcp|'
    'rtsp_flags;prefer_tcp|'
    'fflags;nobuffer|'
    'flags;low_delay|'
    'fflags;flush_packets|'
    'max_delay;500000|'
    'reorder_queue_size;0|'
    'buffer_size;32768'
)

# Disable problematic camera backends
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_OBSENSOR'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
# Force optimal backends for performance
os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '1000'
os.environ['OPENCV_VIDEOIO_PRIORITY_DIRECTSHOW'] = '900'

# Setup Python path and ensure __init__.py files exist
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')

# Add both directories to Python path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def ensure_init_files():
    """Ensure __init__.py files exist in all necessary directories"""
    init_dirs = [
        src_dir,
        os.path.join(src_dir, 'core'),
        os.path.join(src_dir, 'ui'),
        os.path.join(src_dir, 'utils')
    ]
    
    for dir_path in init_dirs:
        if os.path.exists(dir_path):
            init_file = os.path.join(dir_path, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('# Auto-generated __init__.py\n')
                print(f"Created {init_file}")

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "data/reports", 
        "data/backups",
        "config",
        "assets",
        "assets/icons",
        "logs",
        "src",
        "src/core",
        "src/ui",
        "src/utils"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'cv2', 'numpy', 'sklearn', 'scipy', 'PIL', 'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        return False
    
    # Check optional packages
    optional_packages = ['insightface', 'onnxruntime']
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"⚠️ {package} not available (optional)")
    
    return True

class HotelRecognitionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hotel Face Recognition System v2.0 - Enhanced Detection")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Set icon if available
        try:
            icon_path = os.path.join("assets", "icons", "hotel.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except:
            pass
        
        # Maximize window
        try:
            self.root.state('zoomed') if os.name == 'nt' else self.root.attributes('-zoomed', True)
        except:
            pass
        
        # Initialize components
        self.config = None
        self.report_generator = None
        self.gpu_available = False
        self.dashboard = None
        
        # Setup main interface first
        self.setup_main_interface()
        
        # Initialize system after GUI is created
        self.initialize_system()

    def setup_main_interface(self):
        """Setup the main application interface"""
        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="🔧 Camera Setup", command=self.open_camera_setup)
        file_menu.add_command(label="⚙️ Settings", command=self.open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="🔄 Reset Database", command=self.reset_database)
        file_menu.add_separator()
        file_menu.add_command(label="❌ Exit", command=self.on_closing)
        
        # Staff menu
        staff_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Staff", menu=staff_menu)
        staff_menu.add_command(label="👥 Manage Staff", command=self.open_staff_management)
        staff_menu.add_command(label="📋 Staff Attendance", command=self.open_staff_attendance)
        staff_menu.add_separator()
        staff_menu.add_command(label="📥 Import Staff", command=self.import_staff)
        
        # Reports menu
        reports_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Reports", menu=reports_menu)
        reports_menu.add_command(label="📊 View Reports", command=self.open_reports)
        reports_menu.add_command(label="📅 Generate Daily Report", command=self.generate_daily_report)
        reports_menu.add_command(label="📈 Export Monthly Report", command=self.export_monthly_report)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="🔍 Detection Test", command=self.test_detection)
        tools_menu.add_command(label="📷 Camera Test", command=self.test_camera_connection)
        tools_menu.add_command(label="📊 System Statistics", command=self.show_system_stats)
        tools_menu.add_command(label="📸 Captured Photos", command=self.open_captured_photos)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="📖 User Guide", command=self.show_user_guide)
        help_menu.add_command(label="🔧 Troubleshooting", command=self.show_troubleshooting)
        help_menu.add_command(label="ℹ️ About", command=self.show_about)
        
        # Status bar
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status labels
        self.status_label = ttk.Label(self.status_bar, text="Initializing system...", font=('Arial', 9))
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(self.status_bar, orient='vertical').pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # GPU status
        self.gpu_status_label = ttk.Label(self.status_bar, text="Detecting GPU...", font=('Arial', 9))
        self.gpu_status_label.pack(side=tk.RIGHT, padx=5)
        
        # System time
        self.time_label = ttk.Label(self.status_bar, text="", font=('Arial', 9))
        self.time_label.pack(side=tk.RIGHT, padx=5)
        
        # Start time update
        self.update_time()

    def initialize_system(self):
        """Initialize system components safely"""
        try:
            print("🚀 Initializing Hotel Face Recognition System...")
            
            # Import modules after GUI is created
            try:
                from core.config_manager import ConfigManager
                from utils.gpu_utils import detect_gpu_capability
                from utils.report_generator import ReportGenerator
                from utils.installer import check_and_install_requirements
                
                self.config = ConfigManager()
                self.gpu_available = detect_gpu_capability()
                self.report_generator = ReportGenerator()
                
                # Update GPU status
                gpu_status = "GPU Available" if self.gpu_available else "CPU Only"
                self.gpu_status_label.config(text=gpu_status)
                
                # Check requirements
                print("📦 Checking system requirements...")
                if not check_and_install_requirements():
                    messagebox.showwarning("Warning", "Some packages may not be installed correctly")
                
                # Import UI modules
                from ui.dashboard import HotelDashboard
                from ui.camera_setup import CameraSetupWindow
                from ui.staff_management import StaffManagementWindow
                from ui.reports import ReportsWindow
                from ui.staff_attendance import StaffAttendanceWindow
                from ui.captured_photos import CapturedPhotosWindow
                
                # Store classes for later use
                self.HotelDashboard = HotelDashboard
                self.CameraSetupWindow = CameraSetupWindow
                self.StaffManagementWindow = StaffManagementWindow
                self.ReportsWindow = ReportsWindow
                self.StaffAttendanceWindow = StaffAttendanceWindow
                self.CapturedPhotosWindow = CapturedPhotosWindow
                
                # Initialize dashboard
                print("🎯 Initializing Enhanced Face Detection Dashboard...")
                self.dashboard = self.HotelDashboard(self.root, gpu_available=self.gpu_available)
                
                # Schedule automatic reports
                self.schedule_daily_report()
                
                # Update status
                self.status_label.config(text="System Ready - Enhanced Detection Enabled")
                print("✅ Hotel Face Recognition System initialized successfully!")
                
            except ImportError as e:
                error_msg = f"Failed to import system modules: {e}"
                print(f"❌ {error_msg}")
                self.show_import_error(error_msg)
                
            except Exception as e:
                error_msg = f"System initialization failed: {e}"
                print(f"❌ {error_msg}")
                self.show_initialization_error(error_msg)
                
        except Exception as e:
            print(f"❌ Critical initialization error: {e}")
            self.show_critical_error(str(e))

    def show_import_error(self, error_msg):
        """Show import error with helpful message"""
        error_dialog = tk.Toplevel(self.root)
        error_dialog.title("Import Error")
        error_dialog.geometry("600x400")
        error_dialog.transient(self.root)
        error_dialog.grab_set()
        
        frame = ttk.Frame(error_dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="❌ Import Error", font=('Arial', 16, 'bold'), foreground='red').pack(pady=10)
        
        error_text = tk.Text(frame, wrap=tk.WORD, height=15)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=error_text.yview)
        error_text.configure(yscrollcommand=scrollbar.set)
        
        error_content = f"""System modules could not be imported properly.

Error Details:
{error_msg}

Possible Solutions:
1. Install missing packages:
   pip install opencv-python numpy scikit-learn scipy insightface Pillow pandas

2. Check if all source files exist:
   - src/core/config_manager.py
   - src/core/database_manager.py
   - src/core/face_engine.py
   - src/ui/dashboard.py
   - src/utils/gpu_utils.py

3. Verify Python path configuration

4. Run the installer script:
   python install_requirements.py

The system will continue to run with limited functionality."""
        
        error_text.insert('1.0', error_content)
        error_text.config(state=tk.DISABLED)
        
        error_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Button(frame, text="Continue Anyway", command=error_dialog.destroy).pack(pady=10)

    def show_initialization_error(self, error_msg):
        """Show initialization error"""
        messagebox.showerror("Initialization Error", 
                           f"System initialization failed:\n\n{error_msg}\n\n"
                           f"The system will run with limited functionality.")

    def show_critical_error(self, error_msg):
        """Show critical error"""
        messagebox.showerror("Critical Error",
                           f"A critical error occurred:\n\n{error_msg}\n\n"
                           f"Please check the console for more details.")

    def update_time(self):
        """Update the status bar time"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)

    def open_camera_setup(self):
        """Open camera setup window"""
        try:
            if hasattr(self, 'CameraSetupWindow'):
                self.CameraSetupWindow(self.root)
            else:
                messagebox.showerror("Error", "Camera setup module not available")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open camera setup: {e}")

    def open_staff_management(self):
        """Open staff management window"""
        try:
            if hasattr(self, 'StaffManagementWindow'):
                self.StaffManagementWindow(self.root)
            else:
                messagebox.showerror("Error", "Staff management module not available")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open staff management: {e}")

    def open_staff_attendance(self):
        """Open staff attendance window"""
        try:
            if hasattr(self, 'StaffAttendanceWindow'):
                self.StaffAttendanceWindow(self.root)
            else:
                messagebox.showerror("Error", "Staff attendance module not available")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open staff attendance: {e}")

    def open_reports(self):
        """Open reports window"""
        try:
            if hasattr(self, 'ReportsWindow'):
                self.ReportsWindow(self.root)
            else:
                messagebox.showerror("Error", "Reports module not available")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open reports: {e}")

    def open_captured_photos(self):
        """Open captured photos window"""
        try:
            if hasattr(self, 'CapturedPhotosWindow'):
                self.CapturedPhotosWindow(self.root, self.dashboard.captured_photos)
            else:
                messagebox.showerror("Error", "Captured photos module not available")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open captured photos: {e}")

    def open_settings(self):
        """Open system settings"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("System Settings")
        settings_window.geometry("500x400")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        frame = ttk.Frame(settings_window, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="System Settings", font=('Arial', 16, 'bold')).pack(pady=10)
        
        # Detection settings
        detection_frame = ttk.LabelFrame(frame, text="Face Detection Settings", padding=10)
        detection_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(detection_frame, text="Detection Threshold (0.1-0.9):").pack(anchor=tk.W)
        threshold_var = tk.StringVar(value="0.6")
        threshold_entry = ttk.Entry(detection_frame, textvariable=threshold_var, width=10)
        threshold_entry.pack(anchor=tk.W, pady=5)
        
        # Performance settings
        perf_frame = ttk.LabelFrame(frame, text="Performance Settings", padding=10)
        perf_frame.pack(fill=tk.X, pady=10)
        
        gpu_var = tk.BooleanVar(value=self.gpu_available)
        ttk.Checkbutton(perf_frame, text="Use GPU Acceleration (if available)",
                       variable=gpu_var).pack(anchor=tk.W)
        
        ttk.Button(frame, text="Close", command=settings_window.destroy).pack(pady=20)

    def reset_database(self):
        """Reset recognition database"""
        try:
            if not hasattr(self, 'config') or self.config is None:
                messagebox.showerror("Error", "System not properly initialized")
                return
                
            result = messagebox.askyesno(
                "Confirm Database Reset",
                "🚨 WARNING: This will permanently delete ALL recognition data!\n\n"
                "This action CANNOT be undone!\n"
                "A backup will be created automatically.\n\n"
                "Do you want to continue?",
                icon='warning'
            )
            
            if result:
                # Show progress
                progress_window = tk.Toplevel(self.root)
                progress_window.title("Resetting Database...")
                progress_window.geometry("400x150")
                progress_window.transient(self.root)
                progress_window.grab_set()
                
                ttk.Label(progress_window, text="Resetting database, please wait...",
                         font=('Arial', 12)).pack(pady=20)
                
                progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
                progress_bar.pack(pady=10, padx=20, fill=tk.X)
                progress_bar.start()
                
                self.root.update()
                
                # Perform reset (simplified version)
                try:
                    from core.database_manager import DatabaseManager
                    db_manager = DatabaseManager()
                    success = db_manager.reset_recognition_data()
                    
                    progress_window.destroy()
                    
                    if success:
                        messagebox.showinfo("Reset Complete",
                                          "✅ Database reset completed successfully!")
                        self.status_label.config(text="Database reset complete")
                    else:
                        messagebox.showerror("Reset Failed", "❌ Database reset failed")
                        
                except Exception as e:
                    progress_window.destroy()
                    messagebox.showerror("Error", f"Database reset error: {e}")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Reset operation failed: {e}")

    def test_detection(self):
        """Test face detection functionality"""
        messagebox.showinfo("Detection Test",
                          "🔍 Detection Test Feature\n\n"
                          "This will test face detection with your current camera setup.")

    def test_camera_connection(self):
        """Test camera connection"""
        try:
            from utils.camera_utils import find_working_camera_index
            
            messagebox.showinfo("Testing Camera", "Testing camera connection...")
            
            camera_index, backend = find_working_camera_index()
            
            if camera_index is not None:
                messagebox.showinfo("Camera Test Result",
                                  f"✅ Camera connection successful!\n\n"
                                  f"Camera Index: {camera_index}\n"
                                  f"Backend: {backend}")
            else:
                messagebox.showwarning("Camera Test Result",
                                     "❌ No camera detected\n\n"
                                     "Please check camera connections and permissions.")
        except Exception as e:
            messagebox.showerror("Camera Test Error", f"Camera test failed: {e}")

    def show_system_stats(self):
        """Show system statistics"""
        try:
            stats_window = tk.Toplevel(self.root)
            stats_window.title("System Statistics")
            stats_window.geometry("600x500")
            stats_window.transient(self.root)
            
            text_widget = tk.Text(stats_window, wrap=tk.WORD, font=('Courier', 10))
            scrollbar = ttk.Scrollbar(stats_window, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            stats_content = f"""
HOTEL FACE RECOGNITION SYSTEM STATISTICS
========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM INFORMATION:
------------------
GPU Available: {self.gpu_available}
Python Version: {sys.version.split()[0]}
Operating System: {os.name}

CONFIGURATION:
--------------
Config Available: {self.config is not None}
Dashboard Available: {self.dashboard is not None}

STATUS:
-------
System Status: {self.status_label.cget('text')}
"""
            
            text_widget.insert('1.0', stats_content)
            text_widget.config(state=tk.DISABLED)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate statistics: {e}")

    def generate_daily_report(self):
        """Generate daily report"""
        try:
            if hasattr(self, 'report_generator') and self.report_generator:
                report_path = self.report_generator.generate_daily_report()
                messagebox.showinfo("Success", f"✅ Daily report generated!\n\nSaved to: {report_path}")
            else:
                messagebox.showerror("Error", "Report generator not available")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate daily report: {e}")

    def export_monthly_report(self):
        """Export monthly report"""
        try:
            if hasattr(self, 'report_generator') and self.report_generator:
                report_path = self.report_generator.generate_monthly_report()
                messagebox.showinfo("Success", f"✅ Monthly report exported!\n\nSaved to: {report_path}")
            else:
                messagebox.showerror("Error", "Report generator not available")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export monthly report: {e}")

    def import_staff(self):
        """Import staff from file"""
        messagebox.showinfo("Import Staff",
                          "📥 Staff Import Feature\n\n"
                          "Import staff from CSV/Excel files\n"
                          "This feature will be implemented soon.")

    def show_user_guide(self):
        """Show user guide"""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("User Guide - Hotel Face Recognition System")
        guide_window.geometry("800x600")
        guide_window.transient(self.root)
        
        text_widget = tk.Text(guide_window, wrap=tk.WORD, font=('Arial', 11))
        scrollbar = ttk.Scrollbar(guide_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        guide_content = """
HOTEL FACE RECOGNITION SYSTEM - USER GUIDE
==========================================

QUICK START:
-----------
1. 🔧 Configure your camera: File → Camera Setup
2. 👥 Add staff members: Staff → Manage Staff
3. ▶️ Start face recognition: Click "Start Ultra Recognition"
4. 📊 Monitor detections in real-time

CAMERA SETUP:
------------
• For IP/RTSP cameras: Enter your camera URL
• For USB webcams: Select camera index
• Test connection before saving

FACE DETECTION:
--------------
• System detects and tracks faces automatically
• Identifies staff vs customers
• Records visits and attendance

TROUBLESHOOTING:
---------------
• Check camera permissions
• Verify network connection for IP cameras
• Ensure good lighting conditions
• Install required packages if needed
"""
        
        text_widget.insert('1.0', guide_content)
        text_widget.config(state=tk.DISABLED)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def show_troubleshooting(self):
        """Show troubleshooting guide"""
        messagebox.showinfo("Troubleshooting",
                          "🔧 TROUBLESHOOTING GUIDE\n\n"
                          "Detection Issues:\n"
                          "• Check camera permissions\n"
                          "• Ensure good lighting\n"
                          "• Verify camera focus\n\n"
                          "Performance Issues:\n"
                          "• Use GPU if available\n"
                          "• Close other camera apps\n"
                          "• Check system resources")

    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About",
                          "🏨 HOTEL FACE RECOGNITION SYSTEM v2.0\n\n"
                          "🎯 Enhanced Detection & Tracking\n"
                          "🚀 AI-Powered Customer Recognition\n"
                          "📊 Real-time Analytics & Reports\n"
                          "👥 Staff Attendance Management\n\n"
                          "Built with Python, OpenCV, and InsightFace")

    def schedule_daily_report(self):
        """Schedule automatic daily report generation"""
        def check_time():
            now = datetime.now().time()
            end_of_day = time(23, 55)  # 11:55 PM
            
            if now >= end_of_day:
                try:
                    if hasattr(self, 'report_generator') and self.report_generator:
                        self.report_generator.generate_end_of_day_report()
                        self.status_label.config(text="Auto daily report generated")
                except Exception as e:
                    print(f"Auto-report error: {e}")
            
            # Schedule next check in 5 minutes
            self.root.after(300000, check_time)
        
        # Start scheduling after 1 minute
        self.root.after(60000, check_time)

    def on_closing(self):
        """Handle application closing"""
        try:
            # Check if recognition is running
            if (hasattr(self, 'dashboard') and self.dashboard and 
                hasattr(self.dashboard, 'running') and self.dashboard.running):
                result = messagebox.askyesno("Confirm Exit",
                                           "Face recognition is running.\n\n"
                                           "Stop recognition and exit?")
                if not result:
                    return
                
                # Stop recognition
                self.dashboard.stop_recognition()
            
            print("🏁 Shutting down Hotel Face Recognition System...")
            self.status_label.config(text="Shutting down...")
            self.root.update()
            
            # Close application
            self.root.quit()
            self.root.destroy()
            print("✅ Application closed successfully")
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
            self.root.destroy()

    def run(self):
        """Run the application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Center window
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f'{width}x{height}+{x}+{y}')
            
            print("🚀 Starting Hotel Face Recognition System GUI...")
            self.root.mainloop()
            
        except KeyboardInterrupt:
            print("\n🛑 Application interrupted by user")
            self.on_closing()
        except Exception as e:
            print(f"❌ Application error: {e}")
            messagebox.showerror("Application Error", f"Fatal error: {e}")


def main():
    """Main function to run the application"""
    print("=" * 60)
    print("🏨 HOTEL FACE RECOGNITION SYSTEM v2.0")
    print("🎯 Enhanced Detection & Tracking System")
    print("=" * 60)
    
    try:
        # Create directories and init files
        create_directories()
        ensure_init_files()
        
        # Check requirements
        if not check_requirements():
            print("⚠️ Some requirements are missing, but continuing...")
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/hotel_recognition.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create and run application
        app = HotelRecognitionApp()
        app.run()
        
    except Exception as e:
        print(f"❌ Fatal error starting application: {e}")
        try:
            messagebox.showerror("Fatal Error",
                               f"Could not start Hotel Face Recognition System:\n\n{e}")
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
