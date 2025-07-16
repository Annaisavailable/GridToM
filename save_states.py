import av
import os
import torch
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm

from utils import read_video_pyav, load_json, get_key_frames

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='llava-hf/llava-v1.6-mistral-7b-hf')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--annotation', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--image_process_mode', type=str, default='Default')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--temperature', default=0, type=float)
    parser.add_argument('--top_p', default=0, type=float)
    parser.add_argument('--max_new_tokens', default=0, type=int)
    parser.add_argument('--indice_num', default=-1, type=int)
    parser.add_argument('--belief', type=str)

    args = parser.parse_args()
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    return args

def save_states(args):
    if "Qwen2-VL" in args.model:
        from vlm_qwen import VLM
    elif "LLaVA-NeXT-Video" in args.model:
        from vlm import VLM
    else:
        raise NotImplementedError
    annotations = load_json(args.annotation)
    annotations = annotations[-148:]
    visualLLM = VLM(args)

    for index, annotation in tqdm(enumerate(annotations), total=len(annotations), desc="Processing Annotations"):
        desp = annotation['env_desc']
        first_order_beliefs = annotation["first_order_belief"]
        colors = annotation["colors"]
        key_frames = get_key_frames(annotation['nodes'])
        if args.belief == 'TrueBelief':
            belief = first_order_beliefs[0]
            if belief['type'] != args.belief:
                raise ValueError(f"{index}, belief type mismatch!")
            
            if "Qwen2-VL" in args.model:
                clip = [os.path.join(args.dataset, str(annotation['index']), f"{args.belief}/full/{key_id}.png") for key_id in key_frames]
            elif "LLaVA-NeXT-Video" in args.model:
                contrainer = av.open(os.path.join(args.dataset, str(annotation['index']), f"{args.belief}/full/fps3.mp4"))
                clip = read_video_pyav(args, contrainer, key_frames)
            else:
                raise NotImplementedError

            caption = f'Description: {desp} {belief["caption"]}'

            _belief = belief["belief_true"]
            text = f'{caption}\nBelief: {_belief}'
            visualLLM.create_conversation(text, clip)
            attn_list = visualLLM.forward()
            np.save(os.path.join(args.output_dir, f'attn_{args.belief}', f'{index}_true_attn.npy'), attn_list)

            if "Qwen2-VL" in args.model:
                clip = [os.path.join(args.dataset, str(annotation['index']), f"{args.belief}/full/{key_id}.png") for key_id in key_frames]
            elif "LLaVA-NeXT-Video" in args.model:
                contrainer = av.open(os.path.join(args.dataset, str(annotation['index']), f"{args.belief}/full/fps3.mp4"))
                clip = read_video_pyav(args, contrainer, key_frames)
            else:
                raise NotImplementedError

            _belief = belief["belief_false"]
            text = f'{caption}\nBelief: {_belief}'
            visualLLM.create_conversation(text, clip)
            attn_list = visualLLM.forward()
            np.save(os.path.join(args.output_dir, f'attn_{args.belief}', f'{index}_false_attn.npy'), attn_list)
        elif args.belief == 'FalseBelief':
            belief = first_order_beliefs[1]
            if belief['type'] != args.belief:
                raise ValueError(f"{index}, belief type mismatch!")
            
            if "Qwen2-VL" in args.model:
                clip = [os.path.join(args.dataset, str(annotation['index']), f"{args.belief}/protagonist_pov/{key_id}.png") for key_id in key_frames]
            elif "LLaVA-NeXT-Video" in args.model:
                contrainer = av.open(os.path.join(args.dataset, str(annotation['index']), f"{args.belief}/protagonist_pov/fps3.mp4"))
                clip = read_video_pyav(args, contrainer, key_frames)
            else:
                raise NotImplementedError

            _caption = belief["caption"].split(" 3. ")
            caption = f'Description: {desp} {_caption[0]} So the {colors["protagonist"]} agent is unable to know the following information: {_caption[1].replace(" That is where the video ends.", "")}'
            _belief = belief["belief_true"]
            text = f'{caption}\nBelief: {_belief}'
            visualLLM.create_conversation(text, clip)
            attn_list = visualLLM.forward()
            np.save(os.path.join(args.output_dir, f'attn_{args.belief}', f'{index}_true_attn.npy'), attn_list)

            if "Qwen2-VL" in args.model:
                clip = [os.path.join(args.dataset, str(annotation['index']), f"{args.belief}/full/{key_id}.png") for key_id in key_frames]
            elif "LLaVA-NeXT-Video" in args.model:
                contrainer = av.open(os.path.join(args.dataset, str(annotation['index']), f"{args.belief}/full/fps3.mp4"))
                clip = read_video_pyav(args, contrainer, key_frames)
            else:
                raise NotImplementedError
            
            caption = f'Description: {desp} {belief["caption"]}'

            _belief = belief["belief_false"]
            text = f'{caption}\n\nBelief: {_belief}'
            visualLLM.create_conversation(text, clip)
            attn_list = visualLLM.forward()
            np.save(os.path.join(args.output_dir, f'attn_{args.belief}', f'{index}_false_attn.npy'), attn_list)

if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    if os.path.exists(os.path.join(args.output_dir, f'attn_{args.belief}')):
        shutil.rmtree(os.path.join(args.output_dir, f'attn_{args.belief}'))
    os.makedirs(os.path.join(args.output_dir, f'attn_{args.belief}'))
    save_states(args)