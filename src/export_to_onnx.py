"""
Export YOLOv5s PyTorch model (.pt) to ONNX FP32 format.
Returns ModelProto object representing the graph model.
"""

import torch
import onnx
import subprocess
import shutil
from pathlib import Path
from config import YOLOV5S_PT_PATH, ONNX_FP32_PATH, ONNX_EXPORT_CONFIG
from validate_onnx import validate_model_proto


def load_and_export_model():
    """
    Load YOLOv5s PyTorch model and export to ONNX FP32.
    
    Returns:
        onnx.ModelProto: The exported ONNX model object
        
    Raises:
        FileNotFoundError: If the PyTorch model file doesn't exist
        RuntimeError: If export fails
    """
    # Check if PyTorch model exists
    if not YOLOV5S_PT_PATH.exists():
        raise FileNotFoundError(f"PyTorch model not found at: {YOLOV5S_PT_PATH}")
    
    print(f"[INFO] Loading YOLOv5s model from: {YOLOV5S_PT_PATH}")
    
    # Export to ONNX using YOLOv5's export script
    print(f"[INFO] Exporting model to ONNX FP32...")
    print(f"[INFO] Using YOLOv5 export script")
    print(f"[INFO] Opset version: {ONNX_EXPORT_CONFIG['opset_version']}")
    
    try:
        # Use YOLOv5's official export script
        result = subprocess.run(
            [
                "python",
                "yolov5/export.py",
                "--weights", str(YOLOV5S_PT_PATH),
                "--include", "onnx",
                "--opset", str(ONNX_EXPORT_CONFIG['opset_version'])
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"[INFO] Export script output:")
        print(result.stdout)
        
        if result.stderr:
            print(f"[WARNING] Export script warnings:")
            print(result.stderr)
        
        # YOLOv5 export saves as yolov5s.onnx, move to yolov5s_fp32.onnx
        exported_path = YOLOV5S_PT_PATH.parent / "yolov5s.onnx"
        if exported_path.exists() and exported_path != ONNX_FP32_PATH:
            # Use shutil.move to handle existing files (overwrites if exists)
            shutil.move(str(exported_path), str(ONNX_FP32_PATH))
            print(f"[INFO] Moved {exported_path} to {ONNX_FP32_PATH}")
        
        print(f"[SUCCESS] Model exported to: {ONNX_FP32_PATH}")
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"YOLOv5 export failed: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Failed to export model to ONNX: {str(e)}")
    
    # Load and return the ONNX ModelProto object
    print("[INFO] Loading ONNX ModelProto object...")
    try:
        model_proto = onnx.load(str(ONNX_FP32_PATH))
        print(f"[SUCCESS] ModelProto loaded successfully")
        print(f"[INFO] Model graph name: {model_proto.graph.name}")
        print(f"[INFO] Inputs: {[inp.name for inp in model_proto.graph.input]}")
        print(f"[INFO] Outputs: {[out.name for out in model_proto.graph.output]}")
        return model_proto
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX ModelProto: {str(e)}")


def get_model_info(model_proto):
    """
    Extract and display model information.
    
    Args:
        model_proto: ONNX ModelProto object
        
    Returns:
        dict: Model information
    """
    info = {
        'ir_version': model_proto.ir_version,
        'opset_version': model_proto.opset_import[0].version if model_proto.opset_import else None,
        'producer_name': model_proto.producer_name,
        'graph_name': model_proto.graph.name,
        'num_inputs': len(model_proto.graph.input),
        'num_outputs': len(model_proto.graph.output),
        'num_nodes': len(model_proto.graph.node),
    }
    
    print("\n[INFO] Model Information:")
    print(f"  - IR Version: {info['ir_version']}")
    print(f"  - Opset Version: {info['opset_version']}")
    print(f"  - Producer: {info['producer_name']}")
    print(f"  - Graph Name: {info['graph_name']}")
    print(f"  - Inputs: {info['num_inputs']}")
    print(f"  - Outputs: {info['num_outputs']}")
    print(f"  - Nodes: {info['num_nodes']}")
    
    return info


def main():
    """Main execution function."""
    print("=" * 70)
    print("YOLOv5s FP32 ONNX Export Pipeline")
    print("=" * 70)
    
    # Export model and get ModelProto
    model_proto = load_and_export_model()
    
    # Validate using validate_onnx module
    is_valid, message = validate_model_proto(model_proto, "Exported FP32 Model")
    if not is_valid:
        raise RuntimeError(f"Model validation failed: {message}")
    
    # Get model info
    model_info = get_model_info(model_proto)
    
    print("\n" + "=" * 70)
    print("Export completed successfully!")
    print(f"FP32 ONNX model saved to: {ONNX_FP32_PATH}")
    print("=" * 70)
    
    return model_proto, is_valid, model_info


if __name__ == "__main__":
    model_proto, is_valid, model_info = main()