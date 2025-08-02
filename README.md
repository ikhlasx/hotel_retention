# Hotel Face Recognition System v2.0

## Features
- ✅ GPU/CPU automatic detection and switching
- ✅ Staff and customer identification
- ✅ Real-time visit counting
- ✅ Automatic daily reports
- ✅ Easy camera setup with presets
- ✅ DeepSORT tracking
- ✅ Welcome messages and notifications

## Quick Installation

### Method 1: Executable (Recommended)
1. Download `HotelFaceRecognition.exe`
2. Run the executable
3. Follow the setup wizard
4. Configure your camera in Settings > Camera Setup

### Method 2: From Source
Install Python 3.7-3.9
Clone repository
```
git clone <repository-url>
cd HotelFaceRecognition
```

Install requirements
```
pip install -r requirements.txt
```

Run application
```
python main.py
```

## Camera Setup Guide

### Supported Cameras
- EZVIZ CS-H6c Pro
- Hikvision IP cameras
- Dahua IP cameras
- Any RTSP-compatible camera

### RTSP URL Format
`rtsp://username:password@ip_address:port/stream_path`

### Common Examples
- EZVIZ: `rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101`
- Hikvision: `rtsp://admin:password@192.168.1.100:554/Streaming/Channels/1`
- Dahua: `rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0`

## First Time Setup

1. **Launch Application**
2. **Camera Setup**: Go to File > Camera Setup
   - Select your camera brand from presets
   - Enter IP address, username, password
   - Test connection
   - Save settings
3. **Staff Management**: Go to Staff > Manage Staff
   - Add staff members with photos
   - Set departments and roles
4. **Start Recognition**: Click "Start Recognition"

## Usage

### Daily Operation
1. Start the application
2. Click "Start Recognition"
3. System will automatically:
   - Detect and track faces
   - Identify staff vs customers
   - Count visits
   - Display welcome messages
   - Generate reports at day end

### Reports
- **Daily Reports**: Automatic CSV generation
- **Monthly Reports**: Summary statistics
- **Real-time Dashboard**: Live visitor information

## Recognition Tuning

### Confidence Threshold
Adjust how strictly embeddings must match to identify a person. Lower values
such as `0.55` increase matches but risk false positives, while higher values
around `0.7` improve precision at the cost of missed identifications. The
default can be modified via the `confidence_threshold` setting in
`config/settings.json` to reflect real-world performance.

## Troubleshooting

### Camera Issues
- Check IP address and credentials
- Ensure camera is on same network
- Try different stream channels (101, 102)
- Test with VLC media player first

### Performance Issues
- Use GPU mode if available
- Reduce camera resolution for CPU mode
- Check network bandwidth
- Close unnecessary applications

## Support
For technical support, contact: support@yourcompany.com
