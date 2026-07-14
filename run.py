import sys
import subprocess
import time

def run_step(script_path):
    print("="*60)
    print(f"Starting execution of: {script_path}")
    print("="*60)
    start_time = time.time()
    
    # Run script using current Python interpreter
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    
    elapsed = time.time() - start_time
    if result.returncode == 0:
        print(f"\nSuccessfully finished {script_path} in {elapsed:.2f} seconds.\n")
        return True
    else:
        print(f"\nError occurred while executing {script_path}. Return code: {result.returncode}\n")
        return False

def main():
    print("Starting ML Pipeline for 30-Day Hospital Readmission Prediction\n")
    
    steps = [
        "src/download_dataset.py",
        "src/preprocess.py",
        "src/train.py",
        "src/explain.py"
    ]
    
    for step in steps:
        success = run_step(step)
        if not success:
            print("Pipeline execution aborted due to an error in the previous step.")
            sys.exit(1)
            
    print("="*60)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("The trained models, preprocessor, plots, and summaries are ready.")
    print("="*60)

if __name__ == "__main__":
    main()
