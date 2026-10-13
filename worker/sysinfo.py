"""Process and system memory figures, using only the standard library."""

from __future__ import annotations

import os
import sys


def _windows_process_memory() -> tuple[int, int]:
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return 0, 0
    return counters.WorkingSetSize, counters.PeakWorkingSetSize


def _windows_available_memory() -> int | None:
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPhys)


def process_memory() -> dict:
    """Current and peak resident memory of this worker, in bytes (0 when unknown)."""
    try:
        if sys.platform == "win32":
            current, peak = _windows_process_memory()
            return {"rssBytes": int(current), "peakRssBytes": int(peak)}
        if sys.platform.startswith("linux"):
            values = {}
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith(("VmRSS:", "VmHWM:")):
                        key, amount, _unit = line.split()
                        values[key] = int(amount) * 1024
            return {"rssBytes": values.get("VmRSS:", 0), "peakRssBytes": values.get("VmHWM:", 0)}
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
        return {"rssBytes": 0, "peakRssBytes": int(peak)}
    except (OSError, ValueError, AttributeError):
        return {"rssBytes": 0, "peakRssBytes": 0}


def available_memory() -> int | None:
    """Physical memory currently available to new allocations, in bytes, or None when unknown."""
    try:
        if sys.platform == "win32":
            return _windows_available_memory()
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        return None
    except (OSError, ValueError, AttributeError):
        return None


def cpu_count() -> int:
    return os.cpu_count() or 1
