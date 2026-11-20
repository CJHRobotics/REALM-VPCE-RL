import os
import platform

# The virtual environment's directory name, defined here and imported by
# realm_install.py so the two can never disagree. Anything else that needs it
# (job scripts, docs) should match this.
VENV_NAME = "venv-vpce-rl"


def venv_python(project_path):
    """Absolute path to the venv interpreter Webots should launch."""
    if platform.system() == "Windows":
        return os.path.join(project_path, VENV_NAME, "Scripts", "python.exe")
    return os.path.join(project_path, VENV_NAME, "bin", "python3")


def add_runtime_ini_to_dirs(controllers_dir, python_venv_path):
    if not os.path.isdir(controllers_dir):
        # os.walk yields nothing for a missing path, so without this check a
        # wrong directory name writes zero files and still reports success.
        # That is exactly how the old hardcoded "Simulation" (capital S) went
        # unnoticed: it resolved on case-insensitive macOS and silently did
        # nothing on Linux.
        raise FileNotFoundError(f"controllers directory not found: {controllers_dir}")

    written = 0
    # Immediate children only, not os.walk. Webots reads runtime.ini from the
    # controller's own directory, and a recursive walk wrote one into every
    # nested directory too -- including __pycache__, as soon as anything had
    # imported the controller. Those files are junk at best.
    for entry in sorted(os.scandir(controllers_dir), key=lambda e: e.name):
        if not entry.is_dir() or entry.name.startswith(('.', '__')):
            continue
        runtime_ini_file = os.path.join(entry.path, "runtime.ini")

        with open(runtime_ini_file, "w") as out_file:
            out_file.write("[python]\n")
            out_file.write(f"COMMAND = {python_venv_path}")

        print(f"Added runtime.ini to {entry.path}")
        written += 1

    if written == 0:
        raise RuntimeError(f"no controller directories under {controllers_dir}")
    return written


def main():
    project_path = os.getcwd()
    controllers_dir = os.path.join(project_path, "simulation", "controllers")
    add_runtime_ini_to_dirs(controllers_dir, venv_python(project_path))


if __name__ == "__main__":
    main()
