
from global_config import *
import cv2
import numpy as np
from mmseg.apis import inference_model, show_result_pyplot, init_model
from mmseg.datasets import CelebAMaskHQDataset
from PIL import Image

classes = CelebAMaskHQDataset.METAINFO["classes"]
palette = CelebAMaskHQDataset.METAINFO["palette"]

def inference_single_img(model, model_name, img_path, save_flag=True, save_path=None):
    img_bgr = cv2.imread(img_path)
    
    result = inference_model(model, img_bgr)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
    seg_map = pred_mask.astype(np.uint8)
    seg_img = Image.fromarray(seg_map).convert("P")
    seg_img.putpalette(np.array(palette, dtype=np.uint8).flatten())
    
    if save_flag:
        img_name = os.path.basename(img_path)
        if not save_path:
            save_path = os.path.join(MASK_DIR_PATH, img_name.replace(".jpg", ".png"))
            
        seg_img.save(save_path)
        
    return pred_mask