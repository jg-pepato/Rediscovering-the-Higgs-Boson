import ROOT
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
    """Download file via xrdcp with parallel streams and extended timeout."""
    cmd = ["xrdcp", "--force", "--nopbar", "-S", "4", "--retry", "5", url, local_path]
    
    env = os.environ.copy()
    env["X509_USER_PROXY"] = "/dev/null"

    timeout_seconds = 3600 
    
    print(f"  [START] Downloading via xrdcp (Timeout: {timeout_seconds}s)...")
    try:
        result = subprocess.run(cmd, env=env, timeout=timeout_seconds)
        if result.returncode == 0:
            print(f"  [SUCCESS] Downloaded.")
            return True
        else:
            print(f"  [ERROR] xrdcp failed with exit code {result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Download took longer than {timeout_seconds}s.")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    return False


def skim_file(stream, local_path, file_index, dataset):
    """Skim file using appropriate logic. Returns True if successful."""
    mu_dir = os.path.join(BASE_SKIM_DIR, "DoubleMuon", dataset)
    eg_dir = os.path.join(BASE_SKIM_DIR, "DoubleEG", dataset)
    success = True

    # Process DoubleMuon if requested
    if stream in ["DoubleMuon", "both"]:
        try:
            os.makedirs(mu_dir, exist_ok=True)
            print(f"  -> Skimming DoubleMuon to {mu_dir}...")
            SKIM_LOGIC["DoubleMuon"](local_path, file_index, output_dir=mu_dir)
        except Exception as e:
            print(f"  [MU SKIM ERROR] {e}")
            success = False

    # Process DoubleEG if requested
    if stream in ["DoubleEG", "both"]:
        try:
            os.makedirs(eg_dir, exist_ok=True)
            print(f"  -> Skimming DoubleEG to {eg_dir}...")
            SKIM_LOGIC["DoubleEG"](local_path, file_index, output_dir=eg_dir)
        except Exception as e:
            print(f"  [EG SKIM ERROR] {e}")
            success = False
            
    return success


def cleanup_file(local_path):
    """Delete local file to save space."""
    if os.path.exists(local_path):
        os.remove(local_path)


# ===== MAIN PIPELINE =====
def run_pipeline(input_file, stream, dataset):
    """Main pipeline: download and skim ROOT files."""
    
    # 1. Setup base directories only
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"=== STARTING PIPELINE (Stream: {stream}) ===\n")
    
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
        if not download_file(url, local_path):
            log_failed_file(url)
            save_progress(stream, dataset, i + 1)
            continue
    
        skimmed_ok = skim_file(stream, local_path, i+1, dataset) 
        
        if not skimmed_ok:
            log_failed_file(url)
        
        # Cleanup and save
        cleanup_file(local_path)
        save_progress(stream, dataset, i + 1)
        print()

# ===== CHAINS SKIMMED FILES =====
def chain_trees(stream, dataset, tree_name="Events"):
    """Chain skimmed ROOT files into a single file."""

    skim_dir = os.path.join(BASE_SKIM_DIR, stream, dataset)
    output_file = os.path.join(BASE_SKIM_DIR, f"{stream}_{dataset}_chained.root")
    
    chain = ROOT.TChain(tree_name)
    
    for root_file in os.listdir(skim_dir):
        if root_file.endswith(".root"):
            chain.Add(os.path.join(skim_dir, root_file))
    
    print(f"Chaining {chain.GetEntries()} entries into {output_file}...")
    out_f = ROOT.TFile(output_file, "RECREATE")
    out_tree = chain.CloneTree(-1)
    out_tree.Write()
    out_f.Close()
    print("Chaining complete.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    run_pipeline(args.input, args.stream, args.dataset)