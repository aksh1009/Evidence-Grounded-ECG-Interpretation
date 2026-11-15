"""Small runtime helpers."""
import sys


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (no change to power settings; released on exit)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
