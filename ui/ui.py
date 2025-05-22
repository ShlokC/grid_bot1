"""
Tkinter UI for Grid Bot.
"""
import logging
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Dict, List, Any, Optional, Callable
import threading
import time

from core.grid_manager import GridManager

class GridBotUI:
    def __init__(self, grid_manager: GridManager):
        """
        Initialize the Grid Bot UI.
        
        Args:
            grid_manager: Grid manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.grid_manager = grid_manager
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Grid Trading Bot")
        self.root.geometry("1100x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Create UI components
        self._create_ui()
        
        # Update timer
        self.update_timer = None
        
    def _create_ui(self):
        """Create UI components."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel: Grids list
        left_frame = ttk.LabelFrame(main_frame, text="Active Grids")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Grids Treeview
        columns = ('grid_id', 'symbol', 'price_range', 'grids', 'investment', 'leverage', 'pnl', 'status')
        self.grids_tree = ttk.Treeview(left_frame, columns=columns, show='headings')
        
        # Configure columns
        self.grids_tree.heading('grid_id', text='ID')
        self.grids_tree.heading('symbol', text='Symbol')
        self.grids_tree.heading('price_range', text='Price Range')
        self.grids_tree.heading('grids', text='Grids')
        self.grids_tree.heading('investment', text='Investment')
        self.grids_tree.heading('leverage', text='Leverage')
        self.grids_tree.heading('pnl', text='PnL')
        self.grids_tree.heading('status', text='Status')
        
        self.grids_tree.column('grid_id', width=60)
        self.grids_tree.column('symbol', width=80)
        self.grids_tree.column('price_range', width=120)
        self.grids_tree.column('grids', width=60)
        self.grids_tree.column('investment', width=80)
        self.grids_tree.column('leverage', width=60)
        self.grids_tree.column('pnl', width=80)
        self.grids_tree.column('status', width=80)
        
        # Scrollbar for grids
        grids_scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.grids_tree.yview)
        self.grids_tree.configure(yscrollcommand=grids_scrollbar.set)
        
        # Pack grids tree and scrollbar
        self.grids_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        grids_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel: Controls and grid details
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons
        controls_frame = ttk.LabelFrame(right_frame, text="Controls")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        btn_new = ttk.Button(controls_frame, text="New Grid", command=self.create_new_grid)
        btn_new.pack(side=tk.LEFT, padx=5, pady=5)
        
        btn_start = ttk.Button(controls_frame, text="Start", command=self.start_selected_grid)
        btn_start.pack(side=tk.LEFT, padx=5, pady=5)
        
        btn_stop = ttk.Button(controls_frame, text="Stop", command=self.stop_selected_grid)
        btn_stop.pack(side=tk.LEFT, padx=5, pady=5)
        
        btn_delete = ttk.Button(controls_frame, text="Delete", command=self.delete_selected_grid)
        btn_delete.pack(side=tk.LEFT, padx=5, pady=5)
        
        btn_update = ttk.Button(controls_frame, text="Update Config", command=self.update_grid_config)
        btn_update.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Grid details frame
        details_frame = ttk.LabelFrame(right_frame, text="Grid Details")
        details_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Grid details text
        self.details_text = tk.Text(details_frame, height=20, width=40)
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        details_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind events
        self.grids_tree.bind('<<TreeviewSelect>>', self.on_grid_selected)
        
    def run(self):
        """Run the UI main loop."""
        # Start update timer
        self.start_update_timer()
        
        # Run main loop
        self.root.mainloop()
    
    def on_close(self):
        """Handle window close event."""
        if messagebox.askokcancel("Quit", "Do you want to quit? All running grids will continue in the background."):
            # Stop update timer
            self.stop_update_timer()
            
            # Close window
            self.root.destroy()
    
    def update_grids_list(self):
        """Update the grids list."""
        # Clear existing items
        for item in self.grids_tree.get_children():
            self.grids_tree.delete(item)
        
        # Get all grid statuses
        grid_statuses = self.grid_manager.get_all_grids_status()
        
        # Add grids to tree
        for grid in grid_statuses:
            grid_id = grid['grid_id']
            symbol = grid['symbol']
            price_range = f"{grid['price_lower']} - {grid['price_upper']}"
            grid_number = str(grid['grid_number'])
            investment = f"{grid['investment']:.2f}"
            leverage = f"{grid.get('leverage', 1.0):.1f}x"
            pnl = f"{grid['pnl']:.2f} ({grid.get('pnl_percentage', 0):.2f}%)"
            status = "Running" if grid['running'] else "Stopped"
            
            self.grids_tree.insert('', tk.END, values=(
                grid_id, symbol, price_range, grid_number, investment, leverage, pnl, status
            ))
    
    def on_grid_selected(self, event):
        """Handle grid selection event."""
        # Get selected item
        selection = self.grids_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        grid_id = self.grids_tree.item(item, 'values')[0]
        
        # Get grid status
        grid_status = self.grid_manager.get_grid_status(grid_id)
        if grid_status:
            # Display grid details
            self.display_grid_details(grid_status)
    
    def display_grid_details(self, grid_status: Dict):
        """Display grid details in the text area."""
        # Clear existing text
        self.details_text.delete(1.0, tk.END)
        
        # Format grid details
        details = f"Grid ID: {grid_status['grid_id']}\n"
        details += f"Symbol: {grid_status['symbol']}\n"
        details += f"Price Range: {grid_status['price_lower']} - {grid_status['price_upper']}\n"
        details += f"Grid Number: {grid_status['grid_number']}\n"
        details += f"Investment: {grid_status['investment']:.2f}\n"
        details += f"Leverage: {grid_status.get('leverage', 1.0):.1f}x\n"
        details += f"Take Profit: {grid_status['take_profit_pnl']:.2f}%\n"
        details += f"Stop Loss: {grid_status['stop_loss_pnl']:.2f}%\n"
        details += f"PnL: {grid_status['pnl']:.2f}\n"
        details += f"PnL Percentage: {grid_status.get('pnl_percentage', 0):.2f}%\n"
        details += f"Trades Count: {grid_status['trades_count']}\n"
        details += f"Open Orders: {grid_status.get('orders_count', 0)}\n"
        details += f"Status: {'Running' if grid_status['running'] else 'Stopped'}\n"
        
        # Insert details
        self.details_text.insert(tk.END, details)
    
    def create_new_grid(self):
        """Create a new grid dialog."""
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Create New Grid")
        dialog.geometry("400x480")  # Increased height to accommodate leverage slider
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Leverage slider - first control on the form
        ttk.Label(dialog, text="Leverage:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        leverage_frame = ttk.Frame(dialog)
        leverage_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        leverage_var = tk.DoubleVar()
        leverage_var.set(10.0)  # Default value
        
        leverage_slider = ttk.Scale(leverage_frame, from_=1.0, to=75.0, variable=leverage_var, orient=tk.HORIZONTAL)
        leverage_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        leverage_label = ttk.Label(leverage_frame, text="10.0x")
        leverage_label.pack(side=tk.RIGHT, padx=5)
        
        # Update label when slider changes
        def update_leverage_label(*args):
            leverage_label.config(text=f"{leverage_var.get():.1f}x")
        
        leverage_var.trace_add("write", update_leverage_label)
        
        # Symbol
        ttk.Label(dialog, text="Symbol:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        symbol_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=symbol_var).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Price Range
        ttk.Label(dialog, text="Lower Price:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        price_lower_var = tk.DoubleVar()
        ttk.Entry(dialog, textvariable=price_lower_var).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(dialog, text="Upper Price:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        price_upper_var = tk.DoubleVar()
        ttk.Entry(dialog, textvariable=price_upper_var).grid(row=3, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Grid Number
        ttk.Label(dialog, text="Number of Grids:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        grid_number_var = tk.IntVar()
        ttk.Entry(dialog, textvariable=grid_number_var).grid(row=4, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Investment
        ttk.Label(dialog, text="Investment Amount:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        investment_var = tk.DoubleVar()
        ttk.Entry(dialog, textvariable=investment_var).grid(row=5, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Take Profit
        ttk.Label(dialog, text="Take Profit (%):").grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        take_profit_var = tk.DoubleVar()
        take_profit_var.set(5.0)  # Default value
        ttk.Entry(dialog, textvariable=take_profit_var).grid(row=6, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Stop Loss
        ttk.Label(dialog, text="Stop Loss (%):").grid(row=7, column=0, padx=5, pady=5, sticky=tk.W)
        stop_loss_var = tk.DoubleVar()
        stop_loss_var.set(3.0)  # Default value
        ttk.Entry(dialog, textvariable=stop_loss_var).grid(row=7, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Start automatically
        auto_start_var = tk.BooleanVar()
        auto_start_var.set(False)
        ttk.Checkbutton(dialog, text="Start Grid Automatically", variable=auto_start_var).grid(
            row=8, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        samig_var = tk.BooleanVar()
        samig_var.set(True)
        ttk.Checkbutton(dialog, text="Enable SAMIG (Self-Adaptive Market Intelligence)", 
                    variable=samig_var).grid(row=9, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        # Buttons
        def on_create():
            try:
                symbol = symbol_var.get()
                price_lower = price_lower_var.get()
                price_upper = price_upper_var.get()
                grid_number = grid_number_var.get()
                investment = investment_var.get()
                take_profit = take_profit_var.get()
                stop_loss = stop_loss_var.get()
                leverage = leverage_var.get()
                auto_start = auto_start_var.get()
                enable_samig = samig_var.get()
                
                # Validate inputs
                if not symbol:
                    messagebox.showerror("Error", "Symbol cannot be empty")
                    return
                    
                if price_lower >= price_upper:
                    messagebox.showerror("Error", "Lower price must be less than upper price")
                    return
                    
                if grid_number <= 0:
                    messagebox.showerror("Error", "Number of grids must be positive")
                    return
                    
                if investment <= 0:
                    messagebox.showerror("Error", "Investment amount must be positive")
                    return
                
                if leverage < 1 or leverage > 75:
                    messagebox.showerror("Error", "Leverage must be between 1x and 75x")
                    return
                
                # Create grid
                grid_id = self.grid_manager.create_grid(
                    symbol=symbol,
                    price_lower=price_lower,
                    price_upper=price_upper,
                    grid_number=grid_number,
                    investment=investment,
                    take_profit_pnl=take_profit,
                    stop_loss_pnl=stop_loss,
                    leverage=leverage,
                    enable_samig=enable_samig
                )
                
                # Start grid if requested
                if auto_start:
                    self.grid_manager.start_grid(grid_id)
                
                # Update UI
                self.update_grids_list()
                
                # Close dialog
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create grid: {e}")
        
        def on_cancel():
            dialog.destroy()
        
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=9, column=0, columnspan=2, padx=5, pady=10)
        
        ttk.Button(button_frame, text="Create", command=on_create).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        dialog.columnconfigure(1, weight=1)
        
        # Wait for dialog to close
        dialog.wait_window()
    
    def start_selected_grid(self):
        """Start the selected grid."""
        selection = self.grids_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "No grid selected")
            return
        
        item = selection[0]
        grid_id = self.grids_tree.item(item, 'values')[0]
        
        # Start grid
        if self.grid_manager.start_grid(grid_id):
            messagebox.showinfo("Success", f"Grid {grid_id} started")
            self.update_grids_list()
        else:
            messagebox.showerror("Error", f"Failed to start grid {grid_id}")
    
    def stop_selected_grid(self):
        """Stop the selected grid."""
        selection = self.grids_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "No grid selected")
            return
        
        item = selection[0]
        grid_id = self.grids_tree.item(item, 'values')[0]
        
        # Confirm stop
        if messagebox.askyesno("Confirm", f"Are you sure you want to stop grid {grid_id}?"):
            # Stop grid
            if self.grid_manager.stop_grid(grid_id):
                messagebox.showinfo("Success", f"Grid {grid_id} stopped")
                self.update_grids_list()
            else:
                messagebox.showerror("Error", f"Failed to stop grid {grid_id}")
    
    def delete_selected_grid(self):
        """Delete the selected grid."""
        selection = self.grids_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "No grid selected")
            return
        
        item = selection[0]
        grid_id = self.grids_tree.item(item, 'values')[0]
        
        # Confirm delete
        if messagebox.askyesno("Confirm", f"Are you sure you want to delete grid {grid_id}?"):
            # Delete grid
            if self.grid_manager.delete_grid(grid_id):
                messagebox.showinfo("Success", f"Grid {grid_id} deleted")
                self.update_grids_list()
            else:
                messagebox.showerror("Error", f"Failed to delete grid {grid_id}")
    
    def update_grid_config(self):
        """Update configuration of the selected grid."""
        selection = self.grids_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "No grid selected")
            return
        
        item = selection[0]
        grid_id = self.grids_tree.item(item, 'values')[0]
        
        # Get current grid status
        grid_status = self.grid_manager.get_grid_status(grid_id)
        if not grid_status:
            messagebox.showerror("Error", f"Grid {grid_id} not found")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Update Grid {grid_id}")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Take Profit
        ttk.Label(dialog, text="Take Profit (%):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        take_profit_var = tk.DoubleVar()
        take_profit_var.set(grid_status['take_profit_pnl'])
        ttk.Entry(dialog, textvariable=take_profit_var).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Stop Loss
        ttk.Label(dialog, text="Stop Loss (%):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        stop_loss_var = tk.DoubleVar()
        stop_loss_var.set(grid_status['stop_loss_pnl'])
        ttk.Entry(dialog, textvariable=stop_loss_var).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # SAMIG Enable
        ttk.Label(dialog, text="Enable SAMIG:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        samig_var = tk.BooleanVar()
        samig_var.set(grid_status.get('enable_samig', False))
        ttk.Checkbutton(dialog, variable=samig_var).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
    
        # Buttons
        def on_update():
            try:
                take_profit = take_profit_var.get()
                stop_loss = stop_loss_var.get()
                enable_samig = samig_var.get() 
                # Update grid config
                if self.grid_manager.update_grid_config(
                    grid_id=grid_id,
                    take_profit_pnl=take_profit,
                    stop_loss_pnl=stop_loss,
                    enable_samig=enable_samig
                ):
                    messagebox.showinfo("Success", f"Grid {grid_id} configuration updated")
                    self.update_grids_list()
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", f"Failed to update grid {grid_id} configuration")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update grid configuration: {e}")
        
        def on_cancel():
            dialog.destroy()
        
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=10)
        
        ttk.Button(button_frame, text="Update", command=on_update).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        dialog.columnconfigure(1, weight=1)
        
        # Wait for dialog to close
        dialog.wait_window()
    
    def start_update_timer(self):
        """Start the update timer."""
        def update_loop():
            while True:
                try:
                    # Update UI from main thread
                    self.root.after(0, self.update_grids_list)
                    
                    # Wait for next update
                    time.sleep(5)
                    
                except Exception as e:
                    self.logger.error(f"Error in update loop: {e}")
                    time.sleep(5)  # Wait and retry
        
        # Start update thread
        self.update_timer = threading.Thread(target=update_loop)
        self.update_timer.daemon = True
        self.update_timer.start()
    
    def stop_update_timer(self):
        """Stop the update timer."""
        # The timer thread is a daemon, so it will exit when the main thread exits
        self.update_timer = None