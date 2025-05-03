from pathlib import Path
from typing import Dict, List, Optional, Union
from diffusers.loaders.lora_pipeline import _fetch_state_dict
from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING

def load_lora(transformer, lora_path: Path, weight_name: Optional[str] = "pytorch_lora_weights.safetensors"):
    """
    Load LoRA weights into the transformer model.

    Args:
        transformer: The transformer model to which LoRA weights will be applied.
        lora_path (Path): Path to the LoRA weights file.
        weight_name (Optional[str]): Name of the weight to load.

    """
    
    state_dict = _fetch_state_dict(
        lora_path,
        weight_name,
        True,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None)

    state_dict = _convert_hunyuan_video_lora_to_diffusers(state_dict)
    
    adapter_name = weight_name.split(".")[0]
    
    # Check if adapter already exists and delete it if it does
    if hasattr(transformer, 'peft_config') and adapter_name in transformer.peft_config:
        print(f"Adapter '{adapter_name}' already exists. Removing it before loading again.")
        # Use delete_adapters (plural) instead of delete_adapter
        transformer.delete_adapters([adapter_name])
    
    # Load the adapter with the original name
    transformer.load_lora_adapter(state_dict, network_alphas=None, adapter_name=adapter_name)
    print(f"LoRA weights '{adapter_name}' loaded successfully.")
    
    return transformer

def unload_all_loras(transformer):
    """
    Unload all LoRA adapters from the transformer model.
    
    Args:
        transformer: The transformer model from which to remove all LoRA adapters.
    
    Returns:
        The transformer model with all LoRA adapters removed.
    """
    if hasattr(transformer, 'peft_config') and transformer.peft_config:
        # Get all adapter names
        adapter_names = list(transformer.peft_config.keys())
        
        if adapter_names:
            print(f"Removing all LoRA adapters: {', '.join(adapter_names)}")
            # Delete all adapters
            transformer.delete_adapters(adapter_names)
            print("All LoRA adapters have been removed.")
        else:
            print("No LoRA adapters found to remove.")
    else:
        print("Model doesn't have any LoRA adapters or peft_config.")
    
    return transformer

    
# TODO(neph1): remove when HunyuanVideoTransformer3DModelPacked is in _SET_ADAPTER_SCALE_FN_MAPPING
def set_adapters(
        transformer,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):

    adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

    # Expand weights into a list, one entry per adapter
    # examples for e.g. 2 adapters:  [{...}, 7] -> [7,7] ; None -> [None, None]
    if not isinstance(weights, list):
        weights = [weights] * len(adapter_names)

    if len(adapter_names) != len(weights):
        raise ValueError(
            f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
        )

    # Set None values to default of 1.0
    # e.g. [{...}, 7] -> [{...}, 7] ; [None, None] -> [1.0, 1.0]
    weights = [w if w is not None else 1.0 for w in weights]

    # e.g. [{...}, 7] -> [{expanded dict...}, 7]
    scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING["HunyuanVideoTransformer3DModel"]
    weights = scale_expansion_fn(transformer, weights)

    set_weights_and_activate_adapters(transformer, adapter_names, weights)