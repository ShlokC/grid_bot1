"""
Data storage module for simplified grid bot that persists data in JSON format.
"""
import json
import os
import logging
import shutil
from threading import Lock
from typing import Dict, List, Any, Optional
from datetime import datetime

class DataStore:
    def __init__(self, data_dir: str = 'data'):
        """Initialize the data store with file locking support."""
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        
        # Thread safety for file operations
        self.file_locks = {}
        self.global_lock = Lock()
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # File paths
        self.grids_file = os.path.join(data_dir, 'grids.json')
        self.orders_file = os.path.join(data_dir, 'orders.json')
        
        # Initialize data structures - simplified
        self.grids = self._load_json(self.grids_file, {})
        self.orders = self._load_json(self.orders_file, {})
        
    def _get_file_lock(self, file_path: str) -> Lock:
        """Get or create file-specific lock"""
        with self.global_lock:
            if file_path not in self.file_locks:
                self.file_locks[file_path] = Lock()
            return self.file_locks[file_path]
            
    def _load_json(self, file_path: str, default_value: Any) -> Any:
        """Load data from a JSON file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return default_value
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return default_value
    
    def _save_json(self, file_path: str, data: Any) -> bool:
        """Save data to a JSON file with thread safety."""
        file_lock = self._get_file_lock(file_path)
        
        with file_lock:
            try:
                # Create a backup of the existing file if it exists
                if os.path.exists(file_path):
                    backup_path = f"{file_path}.bak"
                    try:
                        shutil.copy2(file_path, backup_path)
                        self.logger.debug(f"Created backup of {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to create backup of {file_path}: {e}")
                
                # Write to a temporary file first to prevent corruption
                temp_path = f"{file_path}.tmp"
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Replace the original file with the temporary file
                if os.path.exists(temp_path):
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    os.rename(temp_path, file_path)
                    
                return True
            except Exception as e:
                self.logger.error(f"Error saving {file_path}: {e}")
                return False
    
    def save_grid(self, grid_id: str, grid_data: Dict) -> bool:
        """Save grid configuration and status."""
        # Add timestamp
        grid_data['last_updated'] = datetime.now().isoformat()
        
        # Ensure numeric values are properly serializable
        for key, value in grid_data.items():
            if isinstance(value, float):
                grid_data[key] = float(value)
            elif isinstance(value, int):
                grid_data[key] = int(value)
        
        # Update grids dictionary
        self.grids[grid_id] = grid_data
        
        # Save to file
        return self._save_json(self.grids_file, self.grids)
    
    def get_grid(self, grid_id: str) -> Optional[Dict]:
        """Get grid data by ID."""
        return self.grids.get(grid_id)
    
    def get_all_grids(self) -> Dict:
        """Get all grid data."""
        return self.grids
    
    def delete_grid(self, grid_id: str) -> bool:
        """Delete a grid by ID."""
        if grid_id in self.grids:
            del self.grids[grid_id]
            return self._save_json(self.grids_file, self.grids)
        return False
    
    def save_orders(self, grid_id: str, orders: Dict) -> bool:
        """Save orders for a grid."""
        # Add timestamp
        timestamp = datetime.now().isoformat()
        
        # Update orders dictionary
        self.orders[grid_id] = {
            'last_updated': timestamp,
            'orders': orders
        }
        
        # Save to file
        return self._save_json(self.orders_file, self.orders)
    
    def get_orders(self, grid_id: str) -> Optional[Dict]:
        """Get orders for a grid."""
        if grid_id in self.orders:
            return self.orders[grid_id]['orders']
        return None
    
    def get_all_orders(self) -> Dict:
        """Get all orders data."""
        return self.orders
    
    def delete_orders(self, grid_id: str) -> bool:
        """Delete orders for a grid."""
        if grid_id in self.orders:
            del self.orders[grid_id]
            return self._save_json(self.orders_file, self.orders)
        return False
    
    def backup_data(self) -> str:
        """Create a backup of all data files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join(self.data_dir, f'backup_{timestamp}')
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copy all JSON files
            for file_name in ['grids.json', 'orders.json']:
                src_path = os.path.join(self.data_dir, file_name)
                if os.path.exists(src_path):
                    with open(src_path, 'r') as src_file:
                        data = json.load(src_file)
                    
                    dst_path = os.path.join(backup_dir, file_name)
                    with open(dst_path, 'w') as dst_file:
                        json.dump(data, dst_file, indent=2)
            
            self.logger.info(f"Data backup created: {backup_dir}")
            return backup_dir
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return ""