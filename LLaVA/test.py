from llava.eval.run_llava import eval_model
from llava.mm_utils import get_model_name_from_path

import argparse
from PIL import Image
import numpy as np

model_path = "liuhaotian/llava-v1.6-34b"

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Arguments to test Llava.')
    parser.add_argument("--image_file", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--save_path", type=str)
    args1 = parser.parse_args()
#     prompt = "What are the things I should be cautious about when I visit here?"
#     image_file = "https://llava-vl.github.io/static/images/view.jpg"

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": args1.prompt,
        "conv_mode": None,
        "image_file": args1.image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    
    outputs = eval_model(args)
    
    print('*'*30)
    print("Llava output =>", outputs)
    
    style_text = outputs.split('"')[1]
    print("style text =>", style_text)
    coords = [float(item) for item in outputs.split("]")[0].split("[")[1].split(",")]
    
    img = Image.open(args1.image_file).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = int(coords[0]*w), int(coords[1]*h), int(coords[2]*w), int(coords[3]*h)
    print(f"coords => x1: {x1}/{w}, y1: {y1}/{h}, x2: {x2}/{w}, y2: {y2}/{h}")
    
    img_np = np.array(img)
    img_np[y1-1:y1+1, x1:x2,:] = [255,0,0]
    img_np[y2-1:y2+1, x1:x2,:] = [255,0,0]
    img_np[y1:y2, x1-1:x1+1, :] = [255,0,0]
    img_np[y1:y2, x2-1:x2+1, :] = [255,0,0]
    
    Image.fromarray(img_np.astype(np.uint8)).save(args1.save_path)
    print(f"image saved at {args1.save_path}.")
    print('*'*30)
    
    
