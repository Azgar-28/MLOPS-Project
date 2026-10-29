
import time
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Lock

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

PROJECT_DIR = Path(__file__).resolve().parent
INCOMING_DIR = PROJECT_DIR / "incoming_data"
NEW_DATA_FILE = PROJECT_DIR / "data" / "new_data.csv"
RETRAIN_SCRIPT = PROJECT_DIR / "src" / "retrain.py"
STATUS_FILE = PROJECT_DIR / "watcher_status.txt"

INCOMING_DIR.mkdir(exist_ok=True)

# Track files currently being processed and already processed.
processing_files = set()
processed_files = set()
processing_lock = Lock()


def update_status(message):
    with open(STATUS_FILE, "w", encoding="utf-8") as file:
        file.write(message)


class CSVHandler(FileSystemEventHandler):

    def on_created(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self.process_file(event.dest_path)

    def on_modified(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)

    def process_file(self, file_path):
        path = Path(file_path)

        # Ignore non-CSV files.
        if path.suffix.lower() != ".csv":
            return

        file_key = str(path.resolve())

        # Ignore duplicate events for the same filename.
        with processing_lock:
            if (
                file_key in processing_files
                or file_key in processed_files
            ):
                return

            processing_files.add(file_key)

        try:
            # Wait for the file copy/write to finish.
            previous_size = -1
            stable = False

            for _ in range(10):
                try:
                    current_size = path.stat().st_size
                except OSError:
                    time.sleep(1)
                    continue

                if current_size == previous_size:
                    stable = True
                    break

                previous_size = current_size
                time.sleep(1)

            if not stable:
                print("File may still be changing:", path.name)
                update_status(
                    "ERROR: File is still changing: " + path.name
                )
                return

            print("\nNew CSV detected:", path.name)
            update_status("New CSV detected: " + path.name)

            # Copy the incoming CSV to the drift detection input.
            shutil.copy2(path, NEW_DATA_FILE)

            update_status(
                "CSV copied. Checking drift and retraining..."
            )

            print("Updated data/new_data.csv")
            print("Starting drift detection and retraining...")

            result = subprocess.run(
                [sys.executable, str(RETRAIN_SCRIPT)],
                cwd=PROJECT_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            print(result.stdout)

            if result.returncode != 0:
                print("Workflow failed:")
                print(result.stderr)

                update_status(
                    "ERROR: Workflow failed. Check terminal."
                )
            else:
                print("Workflow completed.")

                update_status(
                    "SUCCESS: Drift check and retraining completed."
                )

        except Exception as error:
            print("Error processing CSV:", error)
            update_status("ERROR: " + str(error))

        finally:
            # Release the processing lock and remember this file.
            with processing_lock:
                processing_files.discard(file_key)
                processed_files.add(file_key)


if __name__ == "__main__":
    observer = Observer()

    observer.schedule(
        CSVHandler(),
        str(INCOMING_DIR),
        recursive=False
    )

    observer.start()

    print("Watching folder:", INCOMING_DIR)
    update_status("WATCHING: Waiting for a new CSV file.")

    print("Add a new CSV file to trigger the workflow.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        observer.stop()

    observer.join()
    print("Watcher stopped.")
