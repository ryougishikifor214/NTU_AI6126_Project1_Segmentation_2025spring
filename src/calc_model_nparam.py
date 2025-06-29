import os
import sys

from global_config import *

sys.path.append(
    os.path.join(MMSEG_DIR_PATH, "tools", "analysis_tools")
)
print(sys.path[-1])

from get_flops import inference
from mmengine.logging import MMLogger

logger = MMLogger.get_instance(name='MMLogger', log_level="ERROR")

import argparse
def parse_args(arg_list):
    parser = argparse.ArgumentParser(description='Get the FLOPs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        help='override some settings in the used config'
    )

    return parser.parse_args(arg_list)

model_types_nflops_nparams = {
}

units = {"K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12,}

def convert_size_to_int(size_str):
    """Convert a size string like '1.4M', '50K', '0.936G' to an integer in bytes."""
    size_str = size_str.strip().upper()
    
    if size_str[-1] in units:
        num = float(size_str[:-1])
        return int(num * units[size_str[-1]])
    else:
        return int(size_str)

def filter_dict_by_flag(data, key="flag", target_value=True):
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            filtered_value = filter_dict_by_flag(v, key, target_value)
            if isinstance(filtered_value, dict) and filtered_value:
                new_dict[k] = filtered_value
            elif isinstance(filtered_value, list) and filtered_value:
                new_dict[k] = filtered_value
            elif isinstance(v, dict) and v.get(key) == target_value:
                new_dict[k] = v
        return new_dict if new_dict else None
    elif isinstance(data, list): 
        filtered_list = [filter_dict_by_flag(item, key, target_value) for item in data]
        return [item for item in filtered_list if item]  
    else:
        return None

def inference_wrapper(config_path):
    arg_list = [
        config_path,
        "--shape",
        "512",
    ]
    args = parse_args(arg_list)
    result = inference(args,logger)
    return result

THRESHOLD = 1821085
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm
import torch

total_files = sum(len(files) for _, _, files in os.walk(MMSEG_MODEL_CONFIG_DIR_PATH))

def calc_model_types_nparams_nflops():
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for root, dirs, files in os.walk(MMSEG_MODEL_CONFIG_DIR_PATH):
            # print(root)
            # print(dirs)
            # print(files)
            model_type = os.path.relpath(root, MMSEG_MODEL_CONFIG_DIR_PATH)
            print(model_type)
            print(model_types_nflops_nparams.keys())
            for file in files:
                if model_type not in ["maskformer", "mask2former", "point_rend", "convnext", "beit", "poolformer", "ocrnet", "vpd"] and not model_type.startswith("_base_"):
                    if file.endswith(".py"):
                        if file != "swin-tiny-patch4-window7_upernet_1xb8-20k_levir-256x256.py":
                            config_path = os.path.join(root, file)
                            result = inference_wrapper(config_path)
                            params = convert_size_to_int(result["params"])
                            
                            if model_type not in model_types_nflops_nparams:
                                model_types_nflops_nparams[model_type] = {}
                            models_nflops_nparams = model_types_nflops_nparams[model_type]
                            
                            if file not in models_nflops_nparams:
                                models_nflops_nparams[file] = {}
                            model_nflops_nparams = models_nflops_nparams[file]
                            model_nflops_nparams["params"] = result["params"]
                            model_nflops_nparams["flops"] = result["flops"]
                            model_nflops_nparams["flag"] = True if params <= THRESHOLD else False
                pbar.update(1)
                torch.cuda.empty_cache()
            
import pprint

if __name__ == "__main__":
    calc_model_types_nparams_nflops()
    filtered_model_types_nflops_nparams = filter_dict_by_flag(model_types_nflops_nparams)
    
    with open(os.path.join(SRC_DIR_PATH, "models_nparams_dict.py"), "w", encoding="utf-8") as f:
        f.write('model_types_nflops_nparams = ')
        f.write(pprint.pformat(model_types_nflops_nparams))
        f.write('\n\n')
        f.write('filtered_model_types_nflops_nparams = ')
        f.write(pprint.pformat(filtered_model_types_nflops_nparams))