import json
import os
import math
from tqdm import tqdm

def split_json_file(json_path, num_splits=4):
    """
    Split a JSON file into multiple parts.
    
    Args:
        json_path (str): Path to the JSON file to split
        num_splits (int): Number of parts to split the file into
    """
    print(f"Splitting {json_path} into {num_splits} parts...")
    
    # Load the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)
    
    total_items = len(data)
    print(f"Total items in the JSON file: {total_items}")
    
    # Calculate items per split
    items_per_split = math.ceil(total_items / num_splits)
    
    # Create output directory if it doesn't exist
    base_dir = "data/navi/metainfo/"
    file_name = os.path.basename(json_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    
    # Split the data and save to separate files
    for i in range(num_splits):
        start_idx = i * items_per_split
        end_idx = min((i + 1) * items_per_split, total_items)
        
        split_data = data[start_idx:end_idx]
        
        output_path = os.path.join(base_dir, f"{file_name_without_ext}_part{i+1}.json")
        
        with open(output_path, "w") as f:
            json.dump(split_data, f, indent=2)
        
        print(f"Part {i+1}: {len(split_data)} items saved to {output_path}")

if __name__ == "__main__":
    json_path = "data/navi/metainfo/navi_v1.5_metainfo.json"
    split_json_file(json_path, num_splits=12)
