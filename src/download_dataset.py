import os
import zipfile
import requests

def download_and_extract():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
    raw_dir = os.path.join("dataset", "raw")
    zip_path = os.path.join(raw_dir, "dataset_diabetes.zip")

    # Create directories if they do not exist
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading the dataset: {e}")
        return False

    print(f"Extracting zip file to {raw_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        print("Extraction complete!")
        
        # Check files extracted
        extracted_dir = os.path.join(raw_dir, "dataset_diabetes")
        if os.path.exists(extracted_dir):
            files = os.listdir(extracted_dir)
            print(f"Extracted files inside dataset_diabetes folder: {files}")
        else:
            files = os.listdir(raw_dir)
            print(f"Extracted files in raw folder: {files}")
            
        return True
    except Exception as e:
        print(f"Error extracting zip: {e}")
        return False

if __name__ == "__main__":
    download_and_extract()
