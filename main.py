"""Landmark dataset tool: one menu for recording and looking at clips.

    .venv\\Scripts\\python.exe main.py

Options given on the command line (for example --show or --user u01) are
passed on to the recorder when you choose "Record".
"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
import sys

import view_clip

RECORDER_ARGS = sys.argv[1:]


def choose_file():
    """Let the user pick one of the recorded files by number."""
    files = view_clip.list_files()
    if not files:
        print("No recordings yet. Choose 'Record' first.")
        return None
    for i, name in enumerate(files, 1):
        n = len(view_clip.read_file(name)["clips"])
        print(f"  {i}) {name}  ({n} clips)")
    answer = input("File number: ").strip()
    if not answer.isdigit() or not 1 <= int(answer) <= len(files):
        print("Not a valid number.")
        return None
    return files[int(answer) - 1]


def ask_int(prompt, default):
    answer = input(f"{prompt} [{default}]: ").strip()
    return int(answer) if answer.isdigit() else default


def record():
    # Imported here because it loads MediaPipe, which takes a moment.
    import record_landmarks
    sys.argv = [sys.argv[0]] + RECORDER_ARGS
    try:
        record_landmarks.main()
    except SystemExit as e:
        print(e)


def main():
    while True:
        print("\n1) Record clips")
        print("2) List recorded clips")
        print("3) Replay a clip")
        print("4) Print one frame as a 21x3 table")
        print("q) Quit")
        choice = input("Choice: ").strip().lower()

        if choice == "1":
            record()
        elif choice == "2":
            view_clip.list_clips()
        elif choice in ("3", "4"):
            name = choose_file()
            if name is None:
                continue
            clip = ask_int("Clip number", 1)
            try:
                if choice == "3":
                    view_clip.replay(name, clip)
                else:
                    frame = ask_int("Frame number", 0)
                    view_clip.show_array(name, clip, frame)
            except SystemExit as e:
                print(e)
        elif choice in ("q", ""):
            break
        else:
            print("Please type 1, 2, 3, 4 or q.")


if __name__ == "__main__":
    main()
