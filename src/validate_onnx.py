"""
ONNX model validation module.
Provides functions to validate ONNX models according to ONNX standard.
"""

import onnx
from pathlib import Path
from typing import Union, Tuple


def validate_onnx_model(model_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Validate an ONNX model file according to ONNX standard.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at: {model_path}")
    
    print(f"\n[INFO] Validating ONNX model: {model_path.name}")
    print(f"[INFO] File size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        # Load the ONNX model
        print("[INFO] Loading model...")
        model = onnx.load(str(model_path))
        
        # Validate using ONNX checker
        print("[INFO] Running ONNX checker validation...")
        onnx.checker.check_model(model)
        
        # Additional structural checks
        print("[INFO] Performing structural validation...")
        
        # Check graph structure
        graph = model.graph
        
        # Validate inputs
        if len(graph.input) == 0:
            return False, "Model has no inputs"
        
        # Validate outputs
        if len(graph.output) == 0:
            return False, "Model has no outputs"
        
        # Validate nodes
        if len(graph.node) == 0:
            return False, "Model has no nodes"
        
        # Check for disconnected nodes
        node_names = {node.name for node in graph.node}
        if None in node_names:
            print("[WARNING] Some nodes have no names (this is normal)")
        
        # Validate tensor shapes
        print("[INFO] Validating tensor shapes...")
        for input_tensor in graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                    for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"  Input '{input_tensor.name}': shape={shape}, type={input_tensor.type.tensor_type.elem_type}")
        
        for output_tensor in graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                    for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"  Output '{output_tensor.name}': shape={shape}, type={output_tensor.type.tensor_type.elem_type}")
        
        # Check opset version
        if model.opset_import:
            opset_version = model.opset_import[0].version
            print(f"[INFO] Opset version: {opset_version}")
        
        # Check IR version
        print(f"[INFO] IR version: {model.ir_version}")
        
        print("[SUCCESS] Model validation passed!")
        return True, "Model is valid according to ONNX standard"
        
    except onnx.checker.ValidationError as e:
        error_msg = f"ONNX validation error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return False, error_msg
    
    except Exception as e:
        error_msg = f"Unexpected error during validation: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return False, error_msg


def validate_model_proto(model_proto: onnx.ModelProto, model_name: str = "Model") -> Tuple[bool, str]:
    """
    Validate an ONNX ModelProto object in memory.
    
    This function is useful for validating a model after export but before saving,
    or for validating a model that's already loaded in memory.
    
    Args:
        model_proto: ONNX ModelProto object to validate
        model_name: Name of the model for logging purposes
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    print(f"\n[INFO] Validating ModelProto object: {model_name}")
    
    try:
        # Validate using ONNX checker
        print("[INFO] Running ONNX checker validation...")
        onnx.checker.check_model(model_proto)
        
        # Extract and display model information
        graph = model_proto.graph
        
        print(f"[INFO] Model: {model_name}")
        print(f"  - Graph name: {graph.name}")
        print(f"  - IR version: {model_proto.ir_version}")
        
        if model_proto.opset_import:
            print(f"  - Opset version: {model_proto.opset_import[0].version}")
        
        print(f"  - Inputs: {len(graph.input)}")
        for inp in graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                    for dim in inp.type.tensor_type.shape.dim]
            print(f"    * {inp.name}: {shape}")
        
        print(f"  - Outputs: {len(graph.output)}")
        for out in graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                    for dim in out.type.tensor_type.shape.dim]
            print(f"    * {out.name}: {shape}")
        
        print(f"  - Nodes: {len(graph.node)}")
        
        print("[SUCCESS] ModelProto validation passed!")
        return True, f"{model_name} is valid"
        
    except onnx.checker.ValidationError as e:
        error_msg = f"ModelProto validation error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return False, error_msg
    
    except Exception as e:
        error_msg = f"Unexpected error during ModelProto validation: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return False, error_msg


def main():
    """Main execution function for testing."""
    from src.config import ONNX_FP32_PATH, ONNX_FP16_PATH
    
    print("=" * 70)
    print("ONNX Model Validation Module")
    print("=" * 70)
    
    # Test validation functions
    if ONNX_FP32_PATH.exists():
        is_valid, msg = validate_onnx_model(ONNX_FP32_PATH)
        print(f"\nFP32 Model Valid: {is_valid}")
    else:
        print(f"[INFO] FP32 model not found at {ONNX_FP32_PATH}")
    
    if ONNX_FP16_PATH.exists():
        is_valid, msg = validate_onnx_model(ONNX_FP16_PATH)
        print(f"\nFP16 Model Valid: {is_valid}")
    else:
        print(f"[INFO] FP16 model not found at {ONNX_FP16_PATH}")


if __name__ == "__main__":
    main()