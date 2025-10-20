import os
import sys
import subprocess
import contextlib
import importlib

def ignore_stderr(func, *args, **kwargs):
    original_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(original_stderr_fd)

    # Open null device for writing
    if os.name == 'nt':  # Windows
        null_fd = os.open('nul', os.O_WRONLY)
    else:
        null_fd = os.open('/dev/null', os.O_WRONLY)

    try:
        # Redirect stderr to null device
        os.dup2(null_fd, original_stderr_fd)
        os.close(null_fd)

        result = func(*args, **kwargs)

    finally:
        # Restore stderr
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stderr_fd)

    return result


@contextlib.contextmanager
def ignore_os_stderr_contextmanager():
    """Suppresses low-level C stderr (e.g., from Objective-C runtime)."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)  # Duplicate stderr fd
        os.dup2(devnull, 2)  # Redirect stderr to /dev/null
        yield
    finally:
        os.dup2(saved, 2)  # Restore original stderr
        os.close(saved)
        os.close(devnull)

def filter_ffmpeg_warnings_from_stderr(func, *args, **kwargs):
    original_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(original_stderr_fd)

    r_fd, w_fd = os.pipe()

    try:
        # Redirect stderr to the write end of the pipe
        os.dup2(w_fd, original_stderr_fd)

        # Call the function while stderr is redirected
        result = func(*args, **kwargs)

        # Now safe to close write end so the reader gets EOF
        os.close(w_fd)

    finally:
        # Restore stderr
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stderr_fd)

    # Read from the read end of the pipe
    output = b''
    try:
        with os.fdopen(r_fd, 'rb') as pipe_reader:
            output = pipe_reader.read()
    except Exception:
        pass  # Fail-safe

    # Decode and filter
    lines = output.decode(errors='ignore').splitlines()
    filtered = []
    for line in lines:
        if "Changing video frame properties on the fly is not supported by all filters." in line:
            continue
        if "filter context -" in line:
            continue
        filtered.append(line)

    if filtered:
        print("FFmpeg stderr (filtered):\n" + "\n".join(filtered))

    return result

def safe_import_av():
    
    with ignore_os_stderr_contextmanager():
        import av# delayed import inside the context
    check_system_ffmpeg_version_matches_av(av,should_print_output=False)

    return av

def silent_import(module_name: str):
    """
    Import a module while suppressing native stderr warnings.
    Returns the imported module object.
    """
    with ignore_os_stderr_contextmanager():
        module = importlib.import_module(module_name)
    return module

def check_system_ffmpeg_version_matches_av(av,raise_on_mismatch=True,should_print_output=True):
    # Get FFmpeg version from system
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed or not in PATH.")

    # Parse the system version from ffmpeg output
    system_versions = {}
    for line in result.stdout.splitlines():
        if line.startswith("lib"):
            parts = line.split()
            lib = parts[0]
            version = tuple(int(part) for part in parts[1].split(".") if part.strip().isdigit())
            system_versions[lib] = version

    # Get PyAV-linked library versions
    pyav_versions = av.library_versions

    # Compare keys and versions
    mismatches = []
    for lib, pyav_ver in pyav_versions.items():
        libname = f"lib{lib}"
        system_ver = system_versions.get(libname)
        if system_ver and system_ver != pyav_ver:
            mismatches.append((libname, pyav_ver, system_ver))

    if mismatches:
        msg = "\n".join(
            f"{lib}: PyAV={pv}, System={sv}" for lib, pv, sv in mismatches
        )
        full_msg = f"[Version Mismatch] FFmpeg libraries differ:\n{msg}"
        if raise_on_mismatch:
            raise RuntimeError(full_msg)
        else:
            if should_print_output:
                print(full_msg)
            return False
    else:
        if should_print_output:
            print("[PyAV check] System ffmpeg and PyAV ffmpeg versions match.")
        return True