# main.py (Root Directory) - Complete Corrected Version
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import json
from datetime import datetime, time

# CRITICAL: Set environment variables for optimal camera performance BEFORE any imports
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

# Add src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, current_dir)
sys.path.insert(0, src_dir)


class HotelRecognitionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hotel Face Recognition System v2.0 - Enhanced Detection")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # Set icon
        try:
            self.root.iconbitmap("assets/icons/hotel.ico")
        except:
            pass

        # Set window state
        self.root.state('zoomed') if os.name == 'nt' else self.root.attributes('-zoomed', True)

        # Initialize core components
        self.setup_main_interface()

        # Import modules after GUI is created to avoid import errors
        try:
            from ui.dashboard import HotelDashboard
            from ui.camera_setup import CameraSetupWindow
            from ui.staff_management import StaffManagementWindow
            from ui.reports import ReportsWindow
            from ui.staff_attendance import StaffAttendanceWindow
            from utils.installer import check_and_install_requirements
            from utils.gpu_utils import detect_gpu_capability
            from core.config_manager import ConfigManager
            from utils.report_generator import ReportGenerator

            # Store imported classes as instance variables
            self.HotelDashboard = HotelDashboard
            self.CameraSetupWindow = CameraSetupWindow
            self.StaffManagementWindow = StaffManagementWindow
            self.ReportsWindow = ReportsWindow
            self.StaffAttendanceWindow = StaffAttendanceWindow

            # Initialize system components
            print("🚀 Initializing Hotel Face Recognition System...")

            self.config = ConfigManager()
            self.report_generator = ReportGenerator()
            self.gpu_available = detect_gpu_capability()

            # Update GPU status in UI
            gpu_status = "GPU Available" if self.gpu_available else "CPU Only"
            self.gpu_status_label.config(text=gpu_status)

            # Check and install requirements
            print("📦 Checking system requirements...")
            if not check_and_install_requirements():
                messagebox.showerror("Error", "Failed to install required packages")
                sys.exit(1)

            # Initialize enhanced dashboard with detection visibility
            print("🎯 Initializing Enhanced Face Detection Dashboard...")
            self.dashboard = self.HotelDashboard(self.root, gpu_available=self.gpu_available)

            # Schedule automatic reports
            self.schedule_daily_report()

            # Update status
            self.status_label.config(text="System Ready - Enhanced Detection Enabled")

            print("✅ Hotel Face Recognition System initialized successfully!")
            print("🔍 Enhanced face detection with visibility improvements active")

        except ImportError as e:
            error_msg = f"Failed to import system modules: {e}"
            print(f"❌ {error_msg}")
            messagebox.showerror("Import Error", error_msg)
            sys.exit(1)
        except Exception as e:
            error_msg = f"System initialization failed: {e}"
            print(f"❌ {error_msg}")
            messagebox.showerror("Initialization Error", error_msg)
            sys.exit(1)

    def setup_main_interface(self):
        """Setup the main application interface"""
        # Menu bar
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

        # Tools menu (new)
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="🔍 Detection Test", command=self.test_detection)
        tools_menu.add_command(label="📷 Camera Test", command=self.test_camera_connection)
        tools_menu.add_command(label="📊 System Statistics", command=self.show_system_stats)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="📖 User Guide", command=self.show_user_guide)
        help_menu.add_command(label="🔧 Troubleshooting", command=self.show_troubleshooting)
        help_menu.add_command(label="ℹ️ About", command=self.show_about)

        # Enhanced status bar
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Status labels with enhanced information
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

    def update_time(self):
        """Update the status bar time"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)

    def open_camera_setup(self):
        """Open camera setup window"""
        try:
            self.CameraSetupWindow(self.root)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open camera setup: {e}")

    def open_staff_management(self):
        """Open staff management window"""
        try:
            self.StaffManagementWindow(self.root)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open staff management: {e}")

    def open_staff_attendance(self):
        """Open staff attendance window"""
        try:
            self.StaffAttendanceWindow(self.root)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open staff attendance: {e}")

    def open_reports(self):
        """Open reports window"""
        try:
            self.ReportsWindow(self.root)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open reports: {e}")

    def open_settings(self):
        """Open system settings"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("System Settings")
        settings_window.geometry("500x400")
        settings_window.transient(self.root)
        settings_window.grab_set()

        # Settings content
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

        # Close button
        ttk.Button(frame, text="Close", command=settings_window.destroy).pack(pady=20)

    def reset_database(self):
        """Reset recognition database with enhanced confirmation"""
        try:
            from core.database_manager import DatabaseManager

            # Get current database statistics
            db_manager = DatabaseManager()
            stats = db_manager.get_database_stats()

            # Create enhanced confirmation message
            message = "🚨 DATABASE RESET WARNING 🚨\n\n"
            message += "This will permanently delete ALL recognition data:\n\n"

            total_records = 0
            for table, count in stats.items():
                if isinstance(count, int) and count > 0:
                    message += f"• {count:,} records from {table}\n"
                    total_records += count

            if total_records == 0:
                message += "• Database is already empty\n"

            message += f"\nTotal records to be deleted: {total_records:,}\n"
            message += "\n⚠️ This action CANNOT be undone!\n"
            message += "✅ A backup will be created automatically\n\n"
            message += "Do you want to continue?"

            # Show confirmation dialog
            result = messagebox.askyesno(
                "Confirm Database Reset",
                message,
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

                status_label = ttk.Label(progress_window, text="Creating backup...")
                status_label.pack(pady=5)

                self.root.update()

                # Perform reset
                success = db_manager.reset_recognition_data()

                # Close progress window
                progress_window.destroy()

                if success:
                    messagebox.showinfo("Reset Complete",
                                        "✅ Database reset completed successfully!\n\n"
                                        "🔄 All recognition data has been cleared\n"
                                        "💾 Backup created automatically\n"
                                        "🚀 System ready for fresh detection")

                    # Update status
                    self.status_label.config(text="Database reset complete - Ready for fresh detection")

                    # Restart recognition if running
                    if hasattr(self, 'dashboard') and hasattr(self.dashboard, 'running'):
                        if self.dashboard.running:
                            self.dashboard.stop_recognition()
                            messagebox.showinfo("Recognition Restarted",
                                                "Face recognition stopped. Please restart when ready to begin fresh detection.")
                else:
                    messagebox.showerror("Reset Failed",
                                         "❌ Database reset failed\n\n"
                                         "Please check console for error details")

        except Exception as e:
            messagebox.showerror("Error", f"Database reset error: {e}")

    def test_detection(self):
        """Test face detection functionality"""
        messagebox.showinfo("Detection Test",
                            "🔍 Detection Test Feature\n\n"
                            "This will be implemented to test face detection\n"
                            "with your current camera setup and thresholds.")

    def test_camera_connection(self):
        """Test camera connection"""
        try:
            from utils.camera_utils import find_working_camera_index

            messagebox.showinfo("Testing Camera", "Testing camera connection...")

            # Test camera connection
            camera_index, backend = find_working_camera_index()

            if camera_index is not None:
                messagebox.showinfo("Camera Test Result",
                                    f"✅ Camera connection successful!\n\n"
                                    f"Camera Index: {camera_index}\n"
                                    f"Backend: {backend}\n"
                                    f"Ready for face detection")
            else:
                messagebox.showwarning("Camera Test Result",
                                       "❌ No camera detected\n\n"
                                       "Please check:\n"
                                       "• Camera is connected\n"
                                       "• Camera permissions granted\n"
                                       "• IP camera settings if using RTSP")
        except Exception as e:
            messagebox.showerror("Camera Test Error", f"Camera test failed: {e}")

    def show_system_stats(self):
        """Show comprehensive system statistics"""
        try:
            stats_window = tk.Toplevel(self.root)
            stats_window.title("System Statistics")
            stats_window.geometry("600x500")
            stats_window.transient(self.root)

            text_widget = tk.Text(stats_window, wrap=tk.WORD, font=('Courier', 10))
            scrollbar = ttk.Scrollbar(stats_window, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)

            # Generate comprehensive statistics
            stats_content = f"""
HOTEL FACE RECOGNITION SYSTEM STATISTICS
========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM INFORMATION:
------------------
GPU Available: {self.gpu_available}
Python Version: {sys.version.split()[0]}
Operating System: {os.name}

CAMERA CONFIGURATION:
--------------------
"""

            try:
                camera_settings = self.config.get_camera_settings()
                stats_content += f"Camera Type: {camera_settings.get('source_type', 'Unknown')}\n"
                if camera_settings.get('source_type') == 'rtsp':
                    stats_content += f"RTSP URL: {camera_settings.get('rtsp_url', 'Not configured')}\n"
                stats_content += f"Resolution: {camera_settings.get('resolution', 'Unknown')}\n"
                stats_content += f"FPS: {camera_settings.get('fps', 'Unknown')}\n"
            except:
                stats_content += "Camera configuration not available\n"

            stats_content += f"""
DATABASE STATISTICS:
-------------------
"""

            try:
                from core.database_manager import DatabaseManager
                db_manager = DatabaseManager()
                db_stats = db_manager.get_database_stats()

                for table, count in db_stats.items():
                    stats_content += f"{table}: {count}\n"
            except:
                stats_content += "Database statistics not available\n"

            stats_content += f"""
PERFORMANCE SETTINGS:
--------------------
Detection Threshold: 0.6 (Enhanced for visibility)
Processing Resolution: 640x480
Buffer Size: Minimized for low latency
Transport Protocol: TCP (for RTSP cameras)
            """

            text_widget.insert('1.0', stats_content)
            text_widget.config(state=tk.DISABLED)

            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate system statistics: {e}")

    def generate_daily_report(self):
        """Generate daily report"""
        try:
            report_path = self.report_generator.generate_daily_report()
            messagebox.showinfo("Success", f"✅ Daily report generated!\n\nSaved to: {report_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate daily report: {e}")

    def export_monthly_report(self):
        """Export monthly report"""
        try:
            report_path = self.report_generator.generate_monthly_report()
            messagebox.showinfo("Success", f"✅ Monthly report exported!\n\nSaved to: {report_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export monthly report: {e}")

    def import_staff(self):
        """Import staff from file"""
        messagebox.showinfo("Import Staff",
                            "📥 Staff Import Feature\n\n"
                            "Import staff from CSV/Excel files\n"
                            "This feature will be implemented soon.")

    def show_user_guide(self):
        """Show comprehensive user guide"""
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
4. 📊 Monitor detections in real-time on the live feed

CAMERA SETUP:
------------
• For IP/RTSP cameras: Enter your camera URL
  Example: rtsp://admin:password@192.168.1.100:554/stream
• For USB webcams: Select the appropriate camera index
• Test connection before saving settings

FACE DETECTION:
--------------
• Green boxes: Raw face detections from camera
• Colored boxes: Tracking states
  - Yellow: Analyzing new face
  - Cyan: Unknown person detected
  - Green: Known customer recognized
  - Red: Staff member detected
  - Magenta: Newly registered customer

STAFF MANAGEMENT:
----------------
• Add staff with multiple photos for better accuracy
• Capture 5 photos from different angles
• Edit or delete staff records as needed
• Monitor staff attendance automatically

TROUBLESHOOTING:
---------------
• No detections visible: Lower detection threshold in settings
• Camera not connecting: Check camera permissions and network
• Poor recognition: Ensure good lighting and clear face visibility
• High latency: Use TCP transport for IP cameras

REPORTS:
--------
• Daily reports: Customer visits and staff attendance
• Monthly summaries: Comprehensive statistics and trends
• Export to Excel: For further analysis and record keeping
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
                            "• Lower detection threshold to 0.5-0.6\n"
                            "• Ensure good lighting conditions\n"
                            "• Check camera focus and positioning\n\n"
                            "Camera Issues:\n"
                            "• Verify camera permissions\n"
                            "• Test network connection for IP cameras\n"
                            "• Try different camera backends\n\n"
                            "Performance Issues:\n"
                            "• Use GPU acceleration if available\n"
                            "• Reduce processing resolution\n"
                            "• Close other camera applications")

    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About",
                            "🏨 HOTEL FACE RECOGNITION SYSTEM v2.0\n\n"
                            "🎯 Enhanced Detection & Tracking\n"
                            "🚀 AI-Powered Customer Recognition\n"
                            "📊 Real-time Analytics & Reports\n"
                            "👥 Staff Attendance Management\n\n"
                            "🔧 Built with:\n"
                            "• InsightFace for face recognition\n"
                            "• OpenCV for computer vision\n"
                            "• Tkinter for user interface\n"
                            "• SQLite for data storage\n\n"
                            "© 2025 Hotel Management Solutions")

    def schedule_daily_report(self):
        """Schedule automatic daily report generation"""

        def check_time():
            now = datetime.now().time()
            end_of_day = time(23, 55)  # 11:55 PM

            if now >= end_of_day:
                try:
                    self.report_generator.generate_end_of_day_report()
                    self.status_label.config(text="Automatic daily report generated")
                    print("📊 Automatic daily report generated")
                except Exception as e:
                    print(f"Auto-report error: {e}")

            # Schedule next check in 5 minutes
            self.root.after(300000, check_time)  # 5 minutes = 300000 ms

        # Start scheduling after 1 minute
        self.root.after(60000, check_time)

    def on_closing(self):
        """Handle application closing with proper cleanup"""
        try:
            # Confirm exit if recognition is running
            if hasattr(self, 'dashboard') and hasattr(self.dashboard, 'running') and self.dashboard.running:
                result = messagebox.askyesno("Confirm Exit",
                                             "Face recognition is currently running.\n\n"
                                             "Do you want to stop recognition and exit?")
                if not result:
                    return

                # Stop recognition system
                self.dashboard.stop_recognition()
                print("🛑 Face recognition system stopped")

            # Final cleanup
            print("🏁 Shutting down Hotel Face Recognition System...")
            self.status_label.config(text="Shutting down...")

            # Update GUI one last time
            self.root.update()

            # Destroy application
            self.root.quit()
            self.root.destroy()

            print("✅ Application closed successfully")

        except Exception as e:
            print(f"Error during shutdown: {e}")
            self.root.destroy()

    def run(self):
        """Run the application with proper error handling"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

            # Center window on screen
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


if __name__ == "__main__":
    # Create necessary directories
    directories = [
        "data",
        "data/reports",
        "data/backups",
        "config",
        "assets",
        "assets/icons",
        "logs"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Initialize logging
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/hotel_recognition.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    print("=" * 60)
    print("🏨 HOTEL FACE RECOGNITION SYSTEM v2.0")
    print("🎯 Enhanced Detection & Tracking System")
    print("=" * 60)

    try:
        app = HotelRecognitionApp()
        app.run()
    except Exception as e:
        print(f"❌ Fatal error starting application: {e}")
        messagebox.showerror("Fatal Error",
                             f"Could not start Hotel Face Recognition System:\n\n{e}")
        sys.exit(1)
