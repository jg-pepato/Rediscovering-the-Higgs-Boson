import os
import argparse
import subprocess
from skimming_Mu import skim_file_doublemu
from skimming_EG import skim_file_doubleeg

# ===== CONFIG =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
BASE_SKIM_DIR = os.path.join(PROJECT_ROOT, "data", "skimmed")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")

SKIM_LOGIC = {"DoubleMuon": skim_file_doublemu, "DoubleEG": skim_file_doubleeg}


# ===== HELPERS =====
def load_progress(stream, dataset):
    """Load the file index to continue from."""
    index_file = os.path.join(LOG_DIR, f"index_{stream}_{dataset}.txt")
    if os.path.exists(index_file):
        try:
            with open(index_file, "r") as f:
                return int(f.read().strip())
        except:
            return 0
    return 0


def save_progress(stream, dataset, index):
    """Save current progress."""
    index_file = os.path.join(LOG_DIR, f"index_{stream}_{dataset}.txt")
    with open(index_file, "w") as f:
        f.write(str(index))


def log_failed_file(url):
    """Record a failed file URL."""
    with open(os.path.join(LOG_DIR, "failed_files.txt"), "a") as f:
        f.write(f"{url}\n")


def download_file(url, local_path):
    """Download file via xrdcp with retry logic. Returns True if successful."""
    cmd = ["xrdcp", "--force", "--posc", "--retry", "5", url, local_path]
    env = os.environ.copy()
    env["X509_USER_PROXY"] = "/dev/null"
    
    for attempt in range(5):
        print(f"  Connection Attempt {attempt+1}...")
        try:
            result = subprocess.run(cmd, env=env, timeout=600)
            if result.returncode == 0:
                print(f"  [SUCCESS] Downloaded.")
                return True
        except subprocess.TimeoutExpired:
            print(f"  [WARNING] Attempt {attempt+1} timed out.")
        print(f"  [WARNING] Attempt {attempt+1} failed.")
    
    return False


def skim_file(stream, local_path, file_index, output_dir):
    """Skim file using appropriate logic. Returns True if successful."""
    try:
        SKIM_LOGIC[stream](local_path, file_index, output_dir=output_dir)
        print(f"  [DONE] Skimmed.")
        return True
    except Exception as e:
        print(f"  [SKIM ERROR] {e}")
        return False


def cleanup_file(local_path):
    """Delete local file to save space."""
    if os.path.exists(local_path):
        os.remove(local_path)


# ===== MAIN PIPELINE =====
def run_pipeline(input_file, stream, dataset):
    """Main pipeline: download and skim ROOT files."""
    
    # Setup directories
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    output_dir = os.path.join(BASE_SKIM_DIR, stream, dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== STARTING PIPELINE  ===\n")
    
    # Load URLs and resume progress
    with open(input_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    
    current_index = load_progress(stream, dataset)
    
    # Process files
    for i in range(current_index, len(urls)):
        url = urls[i]
        filename = os.path.basename(url)
        local_path = os.path.join(RAW_DIR, filename)
        
        print(f"[File {i+1}/{len(urls)}] Target: {filename}")
        
        # Download
        downloaded_file = download_file(url, local_path) #Downloads file and returns a boolean
        if not downloaded_file:
            print(f"  [ERROR] Skipping {filename}.")
            log_failed_file(url)
            save_progress(stream, dataset, i + 1)
            os.remove(local_path) if os.path.exists(local_path) else None
            continue
        
        # Skim
        skimed_file = skim_file(stream, local_path, i+1, output_dir) #Skims file and returns a boolean
        if not skimed_file:
            log_failed_file(url)
        
        # Cleanup
        cleanup_file(local_path)
        save_progress(stream, dataset, i + 1)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    run_pipeline(args.input, args.stream, args.dataset)