import torch
import os
import json
import traceback
from typing import Dict, Optional, Tuple, Any, Union

def inspect_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Inspect a checkpoint file and return its structure
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dict containing checkpoint information
    """
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        result = {
            "format": "unknown",
            "keys": [],
            "episodes": None,
            "valid": False
        }
        
        if isinstance(checkpoint_data, dict):
            result["keys"] = list(checkpoint_data.keys())
            
            if "model_state_dict" in checkpoint_data:
                result["format"] = "structured"
                result["valid"] = True
                
                if "episodes_completed" in checkpoint_data:
                    result["episodes"] = checkpoint_data["episodes_completed"]
            else:
                # Check if it's a direct state dict
                try:
                    # Attempt to validate if it's a state dict
                    test_keys = [k for k in checkpoint_data.keys() if isinstance(k, str)]
                    if len(test_keys) > 0 and all(isinstance(checkpoint_data[k], torch.Tensor) for k in test_keys):
                        result["format"] = "state_dict"
                        result["valid"] = True
                except Exception:
                    pass
        else:
            # Some other format
            result["format"] = "unrecognized"
            
        return result
    except Exception as e:
        return {
            "format": "error",
            "error": str(e),
            "valid": False
        }

def convert_checkpoint(src_path: str, dest_path: str, episodes: int = 0) -> bool:
    """
    Convert a legacy checkpoint to the new format
    
    Args:
        src_path: Source checkpoint path
        dest_path: Destination path for converted checkpoint
        episodes: Episode count to set
        
    Returns:
        bool: True if successful
    """
    try:
        # Load the source checkpoint
        checkpoint_data = torch.load(src_path, map_location='cpu')
        
        # Create a new structured checkpoint
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            # Already in the right format, just update episodes
            new_checkpoint = checkpoint_data
            new_checkpoint["episodes_completed"] = episodes
        else:
            # Convert direct state dict to structured format
            new_checkpoint = {
                "episodes_completed": episodes,
                "model_state_dict": checkpoint_data,
                "results": {
                    "episode_rewards": [],
                    "episode_steps": [],
                    "workers": []
                }
            }
        
        # Save the new checkpoint
        torch.save(new_checkpoint, dest_path)
        return True
    except Exception as e:
        print(f"Error converting checkpoint: {e}")
        traceback.print_exc()
        return False

def validate_checkpoint(checkpoint_path: str) -> Tuple[bool, str]:
    """
    Validate a checkpoint file
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Tuple of (valid, message)
    """
    if not os.path.exists(checkpoint_path):
        return False, f"Checkpoint file not found: {checkpoint_path}"
    
    info = inspect_checkpoint(checkpoint_path)
    
    if info["valid"]:
        format_type = info["format"]
        episodes = info["episodes"]
        
        if format_type == "structured":
            return True, f"Valid structured checkpoint with {episodes} episodes"
        elif format_type == "state_dict":
            return True, "Valid state dictionary checkpoint (legacy format)"
    
    return False, f"Invalid checkpoint. Format: {info['format']}"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Checkpoint utilities")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a checkpoint")
    inspect_parser.add_argument("checkpoint", help="Path to checkpoint file")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert a checkpoint")
    convert_parser.add_argument("src", help="Source checkpoint path")
    convert_parser.add_argument("dest", help="Destination path")
    convert_parser.add_argument("--episodes", type=int, default=0, help="Episode count")
    
    # Parse args and run commands
    args = parser.parse_args()
    
    if args.command == "inspect":
        info = inspect_checkpoint(args.checkpoint)
        print(json.dumps(info, indent=2))
    elif args.command == "convert":
        success = convert_checkpoint(args.src, args.dest, args.episodes)
        if success:
            print(f"Successfully converted checkpoint to {args.dest}")
        else:
            print("Conversion failed")
    else:
        parser.print_help()
