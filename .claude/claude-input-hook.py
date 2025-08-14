#!/usr/bin/env python3
"""Audio + Visual notification for Claude Code in Cursor"""

import subprocess
import sys
import os
from datetime import datetime

def notify():
    # Visual notification - always shown
    timestamp = datetime.now().strftime("%H:%M:%S")
    visual_alert = f"\n🔔 [{timestamp}] Claude Code is ready for input 🔔"
    print(visual_alert, flush=True)
    
    # Audio notification attempts
    try:
        # Method 1: Try to play a system beep sound directly
        if os.path.exists("/usr/bin/paplay"):
            sound_files = [
                "/usr/share/sounds/alsa/Front_Left.wav",
                "/usr/share/sounds/ubuntu/stereo/bell.ogg", 
                "/usr/share/sounds/sound-icons/bell.wav",
                "/usr/share/sounds/purple/alert.wav"
            ]
            for sound_file in sound_files:
                if os.path.exists(sound_file):
                    subprocess.run(["paplay", sound_file], 
                                 capture_output=True, timeout=2)
                    break
        
        # Method 2: Terminal bell (multiple attempts)
        for _ in range(3):
            sys.stdout.write("\a")
            sys.stdout.flush()
        
    except Exception:
        # Fallback: just bell
        print("\a", end="", flush=True)
    
    # Add a separator line to make it more visible
    print("─" * 50, flush=True)

if __name__ == "__main__":
    notify()