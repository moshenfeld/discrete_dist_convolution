"""
Data utilities for saving and loading experiment results.

This module provides functions for:
- Saving experiment results with timestamps
- Loading the latest results for each experiment type
- Managing data directory structure
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path

# Import DiscreteDist and DistKind from the parent directory
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from implementation.types import DiscreteDist, DistKind

# Configuration
DATA_BASE_DIR = Path(__file__).parent / "data"
PLOTS_DIR = Path(__file__).parent / "plots"

def get_timestamp() -> str:
    """Get current timestamp in format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_experiment_dir(experiment_type: str) -> Path:
    """Get the data directory for a specific experiment type."""
    return DATA_BASE_DIR / experiment_type

def save_results(results: Dict[str, Any], experiment_type: str, filename_prefix: str = "results") -> str:
    """
    Save results to JSON file with timestamp.
    
    Args:
        results: Dictionary containing experiment results
        experiment_type: Type of experiment (e.g., 'gaussian', 'lognormal')
        filename_prefix: Prefix for the filename (default: 'results')
    
    Returns:
        Path to the saved file
    """
    # Create experiment directory
    exp_dir = get_experiment_dir(experiment_type)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = get_timestamp()
    filename = f"{filename_prefix}_{timestamp}.json"
    filepath = exp_dir / filename
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, DiscreteDist):
                    serializable_results[key][subkey] = {
                        'x': subvalue.x.tolist(),
                        'vals': subvalue.vals.tolist(),
                        'kind': subvalue.kind.value,
                        'p_neg_inf': float(subvalue.p_neg_inf),
                        'p_pos_inf': float(subvalue.p_pos_inf)
                    }
                elif isinstance(subvalue, (np.ndarray, np.floating, np.integer)):
                    serializable_results[key][subkey] = subvalue.tolist() if hasattr(subvalue, 'tolist') else float(subvalue)
                else:
                    serializable_results[key][subkey] = subvalue
        else:
            serializable_results[key] = value
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"  Saved results to: {filepath}")
    return str(filepath)

def load_latest_results(experiment_type: str, filename_prefix: str = "results") -> Optional[Dict[str, Any]]:
    """
    Load the latest results for a specific experiment type.
    
    Args:
        experiment_type: Type of experiment (e.g., 'gaussian', 'lognormal')
        filename_prefix: Prefix for the filename (default: 'results')
    
    Returns:
        Dictionary containing the latest results, or None if no files found
    """
    exp_dir = get_experiment_dir(experiment_type)
    
    if not exp_dir.exists():
        return None
    
    # Find all files matching the pattern
    pattern = f"{filename_prefix}_*.json"
    files = list(exp_dir.glob(pattern))
    
    if not files:
        return None
    
    # Sort by modification time and get the latest
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    
    # Load the file
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to DiscreteDist objects
    for key, value in data.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict) and 'x' in subvalue:
                    data[key][subkey] = DiscreteDist(
                        x=np.array(subvalue['x']),
                        kind=DistKind(subvalue['kind']),
                        vals=np.array(subvalue['vals']),
                        p_neg_inf=subvalue['p_neg_inf'],
                        p_pos_inf=subvalue['p_pos_inf']
                    )
    
    print(f"  Loaded latest results from: {latest_file}")
    return data

def list_available_results(experiment_type: str, filename_prefix: str = "results") -> List[str]:
    """
    List all available result files for an experiment type.
    
    Args:
        experiment_type: Type of experiment
        filename_prefix: Prefix for the filename
    
    Returns:
        List of filenames sorted by modification time (newest first)
    """
    exp_dir = get_experiment_dir(experiment_type)
    
    if not exp_dir.exists():
        return []
    
    pattern = f"{filename_prefix}_*.json"
    files = list(exp_dir.glob(pattern))
    
    # Sort by modification time (newest first)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    return [f.name for f in files]

def save_plot(fig, filename_prefix: str) -> str:
    """
    Save a matplotlib figure (overwrites existing files).
    
    Args:
        fig: Matplotlib figure object
        filename_prefix: Prefix for the filename
    
    Returns:
        Path to the saved file
    """
    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate filename without timestamp (overwrites existing files)
    filename = f"{filename_prefix}.png"
    filepath = PLOTS_DIR / filename
    
    # Save the figure
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {filepath}")
    
    return str(filepath)

def cleanup_old_files(experiment_type: str, filename_prefix: str = "results", keep_count: int = 5) -> None:
    """
    Clean up old result files, keeping only the most recent ones.
    
    Args:
        experiment_type: Type of experiment
        filename_prefix: Prefix for the filename
        keep_count: Number of recent files to keep
    """
    exp_dir = get_experiment_dir(experiment_type)
    
    if not exp_dir.exists():
        return
    
    pattern = f"{filename_prefix}_*.json"
    files = list(exp_dir.glob(pattern))
    
    if len(files) <= keep_count:
        return
    
    # Sort by modification time (newest first)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    # Delete old files
    files_to_delete = files[keep_count:]
    for file_path in files_to_delete:
        file_path.unlink()
        print(f"  Deleted old file: {file_path}")
    
    print(f"  Kept {keep_count} most recent files, deleted {len(files_to_delete)} old files")
