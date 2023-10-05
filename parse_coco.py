import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(args):
    clip_model_type = args.clip_model_type
    data_dir = args.data_dir
    split = args.split
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')

    #
    out_path = os.path.join(data_dir,f"oscar_split_{clip_model_name}_{split}.pkl")
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open(f'/ssd_scratch/cvit/manu/clip_cap/annotations/{split}_caption.json', 'r') as f:
        data = json.load(f)
    print(f"%0d captions loaded from {split} json " % len(data))
    
    
    all_embeddings = [] # all images clip embeddings stored 
    all_captions = []  # stores data dict for all images. also stores idx for clip_embedding

    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        filename = os.path.join(data_dir,f"train2014/COCO_train2014_{int(img_id):012d}.jpg")
        if not os.path.isfile(filename):
            filename = os.path.join(data_dir,f"val2014/COCO_val2014_{int(img_id):012d}.jpg")
        # assert os.path.isfile(filename)==True
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="RN50", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--data_dir', default="/ssd_scratch/cvit/manu/clip_cap/")
    parser.add_argument('--split', type = str, default = "val")
    
    args = parser.parse_args()
    exit(main(args))
