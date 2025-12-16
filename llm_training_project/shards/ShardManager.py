import json
import os
from typing import List, Optional

class ShardManager:
    """ Manages shard files for large datasets."""

    def __init__(self, json_path: str):
        """
        Args:
            json_path (str): Path to the JSON file containing shard information.
        """
        self.json_path = json_path
        self.shard_files = self._load_shard_files()

    def _load_shard_files(self) -> List[str]:

        """ 
        Load shard file paths from the JSON file.
        Returns:
            List[str]: List of shard file paths.
        """
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON file not found at {self.json_path}")
        
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        if "remaining_shards" not in data:
            raise KeyError(f"'remaining_shards' key not found in JSON file {self.json_path}")   
        
        return data["remaining_shards"]
        
    def save_shard_files(self, shard_files: List[str]) -> None:
        """
        Save the list of shard file paths to the JSON file.
        Args:
            shard_files (List[str]): List of shard file paths to save.
        """
        # create a temporary file to write the updated shard list
        tmp_path = self.json_path + ".tmp"

        # Write the updated shard list to the temporary file
        with open(tmp_path, 'w') as f:
            json.dump({"remaining_shards": shard_files}, f, indent=4)

        # Replace the original file with the temporary file
        os.replace(tmp_path, self.json_path)   

    def has_next(self) -> bool:
        """
        Check if there are remaining shards to process.
        Returns:
            bool: True if there are remaining shards, False otherwise.
        """
        return len(self.shard_files) > 0

    def get_next_shard(self) -> Optional[str]:
        """
        Get the next shard file path and remove it from the list.
        Returns:
            Optional[str]: The next shard file path, or None if no shards remain.
        """
        if not self.shard_files:
            return None
        return self.shard_files[0]
    
    def remove_shard(self, shard_file: str) -> str:
        """
        Remove a shard file path from the list and update the JSON file.
        Args:
            shard_file (str): The shard file path to remove.
        """
        if not self.shard_files:
            raise ValueError("No shards to remove.")
        
        shard = self.shard_files.pop(0)
        self.save_shard_files(self.shard_files)
        return shard
    


