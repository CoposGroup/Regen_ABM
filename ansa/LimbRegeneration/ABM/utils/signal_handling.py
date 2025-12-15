"""
Signal handling utilities for clean simulation shutdown
"""
import signal
import sys
import matplotlib.pyplot as plt

def signal_handler(animation_manager=None):
    """Handle Ctrl+C interruption"""
    def handler(sig, frame):
        print("\nSimulation interrupted! Cleaning up...")
        if animation_manager:
            animation_manager.close()
        sys.exit(0)
    return handler

def setup_signal_handler(animation_manager=None):
    """Set up signal handler for clean shutdown"""
    signal.signal(signal.SIGINT, signal_handler(animation_manager))
