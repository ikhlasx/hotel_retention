# src/core/config_manager.py
import json
import os
import shutil
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_dir="config"):
        self.config_dir = config_dir
        self.settings_file = os.path.join(config_dir, "settings.json")
        self.camera_file = os.path.join(config_dir, "camera_settings.json")
        self.network_file = os.path.join(config_dir, "network_settings.json")
        
        # Ensure config directory exists
        os.makedirs(config_dir, exist_ok=True)
        
        # Load or create default settings
        self.settings = self.load_settings()
        print(f"ConfigManager initialized with camera file: {self.camera_file}")
        
    def load_settings(self) -> Dict[str, Any]:
        """Load application settings"""
        default_settings = {
            "app_version": "2.0",
            "auto_save_reports": True,
            "report_generation_time": "23:55",
            "max_detection_distance": 0.6,
            "min_face_size": 50,
            "gpu_enabled": True,
            "language": "english",
            "theme": "default",
            # Adjustable face recognition match threshold. Lower values increase
            # matches but may introduce false positives. Recommended range:
            # 0.55-0.7 based on environment.
            "confidence_threshold": 0.55,
        }
        
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    # Merge with defaults
                    default_settings.update(loaded_settings)
            except Exception as e:
                print(f"Error loading settings: {e}")
        
        return default_settings
    
    def save_settings(self, settings: Dict[str, Any] = None):
        """Save application settings"""
        if settings:
            self.settings.update(settings)
        
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def get_setting(self, key: str, default=None):
        """Get a specific setting"""
        return self.settings.get(key, default)
    
    def set_setting(self, key: str, value):
        """Set a specific setting"""
        self.settings[key] = value
        self.save_settings()
    
    def get_camera_settings(self) -> Dict[str, Any]:
        """Get camera settings - CRITICAL METHOD"""
        print(f"Loading camera settings from: {self.camera_file}")
        
        if os.path.exists(self.camera_file):
            try:
                with open(self.camera_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    print(f"Loaded camera settings: {settings}")
                    return settings
            except Exception as e:
                print(f"Error loading camera settings: {e}")
        
        # Return default camera settings if file doesn't exist
        default_settings = {
            "source_type": "usb",
            "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream1",
            "usb_index": 0,
            "resolution": "1920x1080",
            "fps": 25,
            "buffer_size": 5,
            "transport": "TCP"
        }
        print(f"Using default camera settings: {default_settings}")
        return default_settings
    
    def save_camera_settings(self, settings: Dict[str, Any]):
        """Save camera settings"""
        print(f"Saving camera settings to: {self.camera_file}")
        print(f"Settings to save: {settings}")
        
        try:
            with open(self.camera_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
            print("Camera settings saved successfully!")
        except Exception as e:
            print(f"Error saving camera settings: {e}")

    def get_network_settings(self) -> Dict[str, Any]:
        """Load network configuration"""
        print(f"Loading network settings from: {self.network_file}")

        default_settings = {
            "ethernet_ip": "192.168.1.100",
            "wifi_ip": "192.168.1.100",
            "username": "admin",
            "password": "admin",
            "port": 554,
            "auto_switch_enabled": False,
            "preferred_connection": "ethernet",
            "monitor_interval": 5,
            "failover_timeout": 5,
        }

        if os.path.exists(self.network_file):
            try:
                with open(self.network_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    if isinstance(loaded_settings, dict):
                        default_settings.update(loaded_settings)
                    else:
                        print("Network settings file is malformed; using defaults")
            except Exception as e:
                print(f"Error loading network settings: {e}")

        print(f"Using network settings: {default_settings}")
        return default_settings

    def save_network_settings(self, settings: Dict[str, Any]):
        """Persist network configuration"""
        print(f"Saving network settings to: {self.network_file}")
        print(f"Settings to save: {settings}")

        try:
            with open(self.network_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
            print("Network settings saved successfully!")
        except Exception as e:
            print(f"Error saving network settings: {e}")

    def export_configuration(self, export_path: str):
        """Export configuration files to a specified directory"""
        print(f"Exporting configuration to: {export_path}")
        os.makedirs(export_path, exist_ok=True)

        files = [self.settings_file, self.camera_file, self.network_file]

        for file_path in files:
            if os.path.exists(file_path):
                try:
                    shutil.copy(file_path, export_path)
                    print(f"Copied {file_path} to {export_path}")
                except Exception as e:
                    print(f"Error copying {file_path}: {e}")
            else:
                print(f"File {file_path} does not exist, skipping")
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        self.settings = self.load_settings()
        self.save_settings()
        
        # Remove camera settings file to reset to defaults
        if os.path.exists(self.camera_file):
            try:
                os.remove(self.camera_file)
                print("Camera settings reset to defaults")
            except Exception as e:
                print(f"Error resetting camera settings: {e}")
