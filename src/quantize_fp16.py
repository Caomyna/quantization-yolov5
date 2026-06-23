"""
Quantize ONNX FP32 model to FP16 using onnxconverter_common.
"""

import onnx
import onnxconverter_common
from pathlib import Path
from typing import Union, Optional
from config import ONNX_FP32_PATH, ONNX_FP16_PATH, QUANTIZATION_CONFIG


def quantize_fp32_to_fp16(
    model_path: Union[str, Path] = None,
    output_path: Union[str, Path] = None,
    min_positive_val: float = None,
    max_finite_val: float = None,
    keep_io_types: bool = None,
    disable_shape_infer: bool = None
) -> onnx.ModelProto:
    """
    Quantize ONNX FP32 model to FP16.
    
    Args:
        model_path: Path to input FP32 ONNX model (default: from config)
        output_path: Path to save FP16 ONNX model (default: from config)
        min_positive_val: Minimum positive value threshold (default: from config)
        max_finite_val: Maximum finite value threshold (default: from config)
        keep_io_types: Keep input/output as FP32 if True (default: from config)
        disable_shape_infer: Disable shape inference if True (default: from config)
        
    Returns:
        onnx.ModelProto: The quantized FP16 model
        
    Raises:
        FileNotFoundError: If input model doesn't exist
        RuntimeError: If quantization fails
    """
    # Use config defaults if not provided
    model_path = Path(model_path) if model_path else ONNX_FP32_PATH
    output_path = Path(output_path) if output_path else ONNX_FP16_PATH
    
    min_positive_val = min_positive_val if min_positive_val is not None else QUANTIZATION_CONFIG['min_positive_val']
    max_finite_val = max_finite_val if max_finite_val is not None else QUANTIZATION_CONFIG['max_finite_val']
    keep_io_types = keep_io_types if keep_io_types is not None else QUANTIZATION_CONFIG['keep_io_types']
    disable_shape_infer = disable_shape_infer if disable_shape_infer is not None else QUANTIZATION_CONFIG['disable_shape_infer']
    
    # Check if input model exists
    if not model_path.exists():
        raise FileNotFoundError(f"FP32 ONNX model not found at: {model_path}")
    
    print("\n" + "=" * 70)
    print("FP32 to FP16 Quantization")
    print("=" * 70)
    print(f"[INFO] Input model: {model_path}")
    print(f"[INFO] Output model: {output_path}")
    print(f"[INFO] Input model size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Load the FP32 model
    print("\n[INFO] Loading FP32 ONNX model...")
    try:
        model_proto = onnx.load(str(model_path))
        print(f"[SUCCESS] Model loaded successfully")
        print(f"[INFO] Graph name: {model_proto.graph.name}")
        print(f"[INFO] Number of nodes: {len(model_proto.graph.node)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load FP32 model: {str(e)}")
    
    # Perform FP16 quantization
    print("\n[INFO] Starting FP16 quantization...")
    print(f"[INFO] Quantization parameters:")
    print(f"  - min_positive_val: {min_positive_val}")
    print(f"  - max_finite_val: {max_finite_val}")
    print(f"  - keep_io_types: {keep_io_types}")
    print(f"  - disable_shape_infer: {disable_shape_infer}")
    
    try:
        # Use onnxconverter_common to convert to FP16
        # The convert_float_to_float16 function converts FP32 to FP16
        quantized_model = onnxconverter_common.convert_float_to_float16(
            model_proto,
            min_positive_val=min_positive_val,
            max_finite_val=max_finite_val,
            keep_io_types=keep_io_types,
            disable_shape_infer=disable_shape_infer
        )
        
        print("[SUCCESS] Quantization completed successfully")
        
    except Exception as e:
        raise RuntimeError(f"FP16 quantization failed: {str(e)}")
    
    # Validate the quantized model
    print("\n[INFO] Validating quantized FP16 model...")
    try:
        onnx.checker.check_model(quantized_model)
        print("[SUCCESS] FP16 model validation passed")
    except onnx.checker.ValidationError as e:
        print(f"[WARNING] FP16 model validation issue: {str(e)}")
        print("[INFO] Attempting to save model anyway...")
    
    # Save the quantized model
    print(f"\n[INFO] Saving FP16 model to: {output_path}")
    try:
        onnx.save(quantized_model, str(output_path))
        print(f"[SUCCESS] FP16 model saved successfully")
        print(f"[INFO] Output model size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Calculate size reduction
        fp32_size = model_path.stat().st_size / (1024 * 1024)
        fp16_size = output_path.stat().st_size / (1024 * 1024)
        size_reduction = ((fp32_size - fp16_size) / fp32_size) * 100
        print(f"[INFO] Size reduction: {size_reduction:.2f}%")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save FP16 model: {str(e)}")
    
    print("\n" + "=" * 70)
    print("Quantization completed successfully!")
    print("=" * 70)
    
    return quantized_model


def quantize_with_custom_params(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    **kwargs
) -> onnx.ModelProto:
    """
    Quantize model with custom parameters.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save quantized model
        **kwargs: Custom quantization parameters
        
    Returns:
        onnx.ModelProto: The quantized model
    """
    print(f"\n[INFO] Quantizing with custom parameters...")
    return quantize_fp32_to_fp16(
        model_path=input_path,
        output_path=output_path,
        **kwargs
    )


def get_quantization_info(model_proto: onnx.ModelProto) -> dict:
    """
    Extract quantization information from the model.
    
    Args:
        model_proto: ONNX ModelProto object
        
    Returns:
        dict: Model information including data types
    """
    info = {
        'ir_version': model_proto.ir_version,
        'opset_version': model_proto.opset_import[0].version if model_proto.opset_import else None,
        'num_nodes': len(model_proto.graph.node),
        'num_inputs': len(model_proto.graph.input),
        'num_outputs': len(model_proto.graph.output),
        'input_dtypes': [],
        'output_dtypes': [],
    }
    
    # Get input data types
    for inp in model_proto.graph.input:
        dtype = inp.type.tensor_type.elem_type
        dtype_name = onnx.TensorProto.DataType.Name(dtype)
        info['input_dtypes'].append({
            'name': inp.name,
            'dtype': dtype,
            'dtype_name': dtype_name
        })
    
    # Get output data types
    for out in model_proto.graph.output:
        dtype = out.type.tensor_type.elem_type
        dtype_name = onnx.TensorProto.DataType.Name(dtype)
        info['output_dtypes'].append({
            'name': out.name,
            'dtype': dtype,
            'dtype_name': dtype_name
        })
    
    # Count FP16 vs FP32 nodes
    fp16_count = 0
    fp32_count = 0
    for node in model_proto.graph.node:
        for attr in node.attribute:
            if attr.name == 'value':
                if hasattr(attr.t, 'data_type'):
                    dtype = attr.t.data_type
                    if dtype == onnx.TensorProto.FLOAT16:
                        fp16_count += 1
                    elif dtype == onnx.TensorProto.FLOAT:
                        fp32_count += 1
    
    info['fp16_tensors'] = fp16_count
    info['fp32_tensors'] = fp32_count
    
    return info


def print_quantization_info(model_proto: onnx.ModelProto):
    """
    Print detailed quantization information.
    
    Args:
        model_proto: ONNX ModelProto object
    """
    info = get_quantization_info(model_proto)
    
    print("\n" + "-" * 70)
    print("Quantization Information:")
    print("-" * 70)
    print(f"IR Version: {info['ir_version']}")
    print(f"Opset Version: {info['opset_version']}")
    print(f"Number of nodes: {info['num_nodes']}")
    
    print(f"\nInputs ({info['num_inputs']}):")
    for inp in info['input_dtypes']:
        print(f"  - {inp['name']}: {inp['dtype_name']}")
    
    print(f"\nOutputs ({info['num_outputs']}):")
    for out in info['output_dtypes']:
        print(f"  - {out['name']}: {out['dtype_name']}")
    
    print(f"\nTensor Statistics:")
    print(f"  - FP16 tensors: {info['fp16_tensors']}")
    print(f"  - FP32 tensors: {info['fp32_tensors']}")
    print("-" * 70)


def main():
    """Main execution function."""
    print("=" * 70)
    print("YOLOv5s FP16 Quantization Module")
    print("=" * 70)
    
    # Perform quantization
    try:
        quantized_model = quantize_fp32_to_fp16()
        
        # Print quantization info
        print_quantization_info(quantized_model)
        
        print("\n[SUCCESS] FP16 quantization completed!")
        print(f"[INFO] FP16 model saved to: {ONNX_FP16_PATH}")
        
    except Exception as e:
        print(f"\n[ERROR] Quantization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()