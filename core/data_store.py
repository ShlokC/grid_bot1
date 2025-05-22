"""
Data storage module for grid bot that persists data in JSON format.
"""
import json
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import shutil

class DataStore:
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the data store.
        
        Args:
            data_dir: Directory to store data files
        """
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # File paths
        self.grids_file = os.path.join(data_dir, 'grids.json')
        self.orders_file = os.path.join(data_dir, 'orders.json')
        self.positions_file = os.path.join(data_dir, 'positions.json')
        
        # Initialize data structures
        self.grids = self._load_json(self.grids_file, {})
        self.orders = self._load_json(self.orders_file, {})
        self.positions = self._load_json(self.positions_file, {})
    
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
        """Save data to a JSON file with enhanced error handling."""
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
        """
        Save grid configuration and status.
        
        Args:
            grid_id: Unique grid identifier
            grid_data: Grid configuration and status data
                
        Returns:
            bool: Success status
        """
        # Add timestamp
        grid_data['last_updated'] = datetime.now().isoformat()
        
        # Ensure numeric values are properly serializable
        # (convert any numeric values that might cause JSON serialization issues)
        for key, value in grid_data.items():
            if isinstance(value, float) or isinstance(value, int):
                grid_data[key] = float(value) if isinstance(value, float) else int(value)
        
        # Update grids dictionary
        self.grids[grid_id] = grid_data
        
        # Save to file
        success = self._save_json(self.grids_file, self.grids)
        if success:
            self.logger.info(f"Saved grid data for {grid_id}")
        else:
            self.logger.error(f"Failed to save grid data for {grid_id}")
        
        return success
    
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
        """
        Save orders for a grid.
        
        Args:
            grid_id: Grid identifier
            orders: Dictionary of orders (order_id -> order_info)
            
        Returns:
            bool: Success status
        """
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
    
    def save_positions(self, grid_id: str, positions: Dict) -> bool:
        """
        Save positions for a grid.
        
        Args:
            grid_id: Grid identifier
            positions: Dictionary of positions (position_id -> position_info)
            
        Returns:
            bool: Success status
        """
        # Add timestamp
        timestamp = datetime.now().isoformat()
        
        # Update positions dictionary
        self.positions[grid_id] = {
            'last_updated': timestamp,
            'positions': positions
        }
        
        # Save to file
        return self._save_json(self.positions_file, self.positions)
    
    def get_positions(self, grid_id: str) -> Optional[Dict]:
        """Get positions for a grid."""
        if grid_id in self.positions:
            return self.positions[grid_id]['positions']
        return None
    
    def get_all_positions(self) -> Dict:
        """Get all positions data."""
        return self.positions
    
    def delete_positions(self, grid_id: str) -> bool:
        """Delete positions for a grid."""
        if grid_id in self.positions:
            del self.positions[grid_id]
            return self._save_json(self.positions_file, self.positions)
        return False
    
    def backup_data(self) -> str:
        """Create a backup of all data files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join(self.data_dir, f'backup_{timestamp}')
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copy all JSON files
            for file_name in ['grids.json', 'orders.json', 'positions.json']:
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