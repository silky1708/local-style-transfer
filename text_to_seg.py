import sys
sys.path.append('./LLaVA')
from llava.eval.run_llava import eval_model
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

sys.path.append('./segment-anything')
from segment_anything import sam_model_registry, SamPredictor

import argparse
import numpy as np
import cv2


def pre_load_models(model_path = "liuhaotian/llava-v1.6-34b", sam_checkpoint = "./segment-anything/checkpoints/sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda", model_base = None):
    model_name = get_model_name_from_path(model_path)
    
#     llava
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_base, model_name
    )
    
#     SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    return tokenizer, model, image_processor, predictor
    
    

def get_segmentation_from_user_prompt(image_path, prompt, tokenizer, model, image_processor, predictor, threshold=0):
    llava_prompt = f"For a given user prompt: '{prompt}', give the bounding box coordinates of the object that should be stylized. Also return the corresponding style in quotes."
    model_path = "liuhaotian/llava-v1.6-34b"
    model_name = get_model_name_from_path(model_path)
    
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": model_name,
        "query": llava_prompt,
        "conv_mode": None,
        "image_file": image_path,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    
    outputs = eval_model(args, model_name, tokenizer, model, image_processor)
    print("Llava output =>", outputs)
    
    style_text = outputs.split('"')[1]
    coords = [float(item) for item in outputs.split("]")[0].split("[")[1].split(",")]
    
#     calling SAM to get segmentation masks from bbox inputs.
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w, _ = image.shape
    x1, y1, x2, y2 = int(coords[0]*w), int(coords[1]*h), int(coords[2]*w), int(coords[3]*h)
    
    predictor.set_image(image)
    
    input_box = np.array([x1-threshold, y1-threshold, x2+threshold, y2+threshold])
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    
    seg_mask = masks[0]  # HxW binary mask with values in [0,1]
    return seg_mask, style_text.strip().lower().replace("style", "")
    
    
    