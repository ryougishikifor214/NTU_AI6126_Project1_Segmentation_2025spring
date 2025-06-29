# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Haozhe Xie
# @Date:   2025-02-18 19:09:59
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-02-19 08:13:52
# @Email:  root@haozhexie.com

import sys
import os
SRC_DIR_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "src"
    )
)
sys.path.append(SRC_DIR_PATH)

from global_config import *
from utils.inference import inference_single_img
import logging

import argparse
import cv2
import torch

from PIL import Image

from mmseg.apis import init_model

def main(input, output, weights):
    # # Load the input image
    # img = cv2.imread(input)

    # # TODO: Initialize the neural network model
    # # Example:
    # # from models import YourSegModel
    # # model = YourSegModel()
    # model = None

    # # Load the checkpoint
    # ckpt = torch.load(weights)
    # # NOTE: Make sure that the weights are saved in the "state_dict" key
    # # DO NOT CHANGE THIS VALUE, i.e., ckpt["state_dict"]
    # model.load_state_dict(ckpt["state_dict"])
    # model.eval()

    # # Inference with the model (Update as needed)
    # # Normalize the image.
    # # NOTE: Make sure it is aligned with the training data
    # # Example: img = (img / 255.0 - 0.5) * 2.0
    # prediction = model(img)

    # # Convert PyTorch Tensor to numpy array
    # mask = prediction.cpu().numpy()
    # # Save the prediction
    # Image.fromarray(mask.convert("P")).save(output)
    if weights not in CKPT_CONFIG_LOOKUPS:
        logging.error(f"Invalid weights: {weights}")
        exit(-1)
    
    config = CKPT_CONFIG_LOOKUPS[weights]["config"]
    checkpoint = CKPT_CONFIG_LOOKUPS[weights]["checkpoint"]
    
    model = init_model(config, checkpoint, device="cuda:0")
    img_path = input
    save_path = output
    inference_single_img(model, weights, img_path, save_flag=True, save_path=save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--weights", type=str, default="ckpt.pth")
    args = parser.parse_args()
    main(args.input, args.output, args.weights)
