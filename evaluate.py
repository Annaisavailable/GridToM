import av
import os
import ast
import time
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm

from utils import load_json, read_video_pyav, colorize, ensure_dir, get_key_frames

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
    parser.add_argument('--indice_num', default=0, type=int)
    parser.add_argument('--belief', type=str)
    parser.add_argument('--alpha', default=0, type=int)
    parser.add_argument('--K', default=0, type=int)
    parser.add_argument('--direction', type=str)
    parser.add_argument('--intervene', dest='intervene', type=ast.literal_eval)
    parser.add_argument('--perspective', type=str)

    args = parser.parse_args()
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    return args

def write_error(i, itemid, res, ans, args):
    error_file = os.path.join(
        args.output_dir, 
        f"interv_evaluate/alpha{args.alpha}_k{args.K}/{args.perspective}_{args.direction}" if args.intervene else "evaluate",
        f"errors_{args.belief}.txt"
    )
    with open(error_file, 'a') as error_f:
        error_f.write(f'id:{i}, itemid:{itemid}, res:{res}, ans:{ans}\n')

def evaluate(args):
    if "Qwen2-VL" in args.model:
        from vlm_qwen import VLM
    elif "LLaVA-NeXT-Video" in args.model:
        from vlm import VLM
    else:
        raise NotImplementedError
    sum = 0
    right = 0
    wrong = 0
    error = 0
    annotations = load_json(args.annotation)
    annotations = annotations[:-148]
    visualLLM = VLM(args)
    if args.intervene:
        visualLLM.load_interv()
    with open(os.path.join(args.output_dir, f"interv_evaluate/alpha{args.alpha}_k{args.K}/{args.perspective}_{args.direction}" if args.intervene else "evaluate", f"{args.belief}.txt"), 'w') as f:
        f.write(time.strftime('Output Time: %Y-%m-%d %H:%M:%S\n------------------------\n', time.localtime(time.time())))
        with tqdm(total=len(annotations), dynamic_ncols=True) as pbar:
            for index, annotation in enumerate(annotations):
                desp = annotation['env_desc']
                first_order_beliefs = annotation["first_order_belief"]
                belief = first_order_beliefs[1 if args.belief == 'FalseBelief' else 0]
                if belief['type'] != args.belief:
                    raise ValueError(f"{index}, belief type mismatch!")
                
                colors = annotation["colors"]
                question = belief['question']
                options = belief['options']
                answer = belief['answer']

                key_frames = get_key_frames(annotation['nodes'])

                if "Qwen2-VL" in args.model:
                    clip = [os.path.join(args.dataset, str(annotation['index']), f"{args.belief}/full/{key_id}.png") for key_id in key_frames]
                elif "LLaVA-NeXT-Video" in args.model:
                    contrainer = av.open(os.path.join(args.dataset, str(annotation['index']), f"{args.belief}/full/fps3.mp4"))
                    clip = read_video_pyav(args, contrainer, key_frames)
                else:
                    raise NotImplementedError

                caption = f"Description: {desp} {belief['caption']}"
                req = 'Please answer only in the following JSON format without adding any additional text or instructions:```json{"answer": "your option"}'
                ## MultiModal
                # text = f'{caption}\nQuestion: {question} Choose from {options[0]} or {options[1]}.\nRequirement: {req}'
                ## Video
                text = f'Question: {question} Choose from {options[0]} or {options[1]}.\nRequirement: {req}'
                
                visualLLM.create_conversation(text, clip)

                if args.intervene:
                    res = visualLLM.generate_interv()
                else:
                    res = visualLLM.generate()

                try:
                    norm_res = json.loads(res)['answer']
                    norm_res = norm_res.lower()
                    if norm_res == answer:
                        right += 1
                        f.write(f"Right. ID:{index},ItemId:{sum},Res:{res},Ans:{answer}\n")
                    elif norm_res in options:
                        wrong += 1
                        f.write(f"Wrong. ID:{index},ItemId:{sum},Prompt:{text},Res:{res},Ans:{answer}\n")
                    else:
                        write_error(index, sum, res, answer, args)
                        error += 1
                except:
                    write_error(index, sum, res, answer, args)
                    error += 1
                sum += 1

                desc = colorize(f"Right: Wrong: Error = ({right}: {wrong}: {error}) / {sum}. Acc: {right / sum * 100:.1f}%", "red")
                pbar.set_description(desc)
                pbar.update(1)

        f.write("------------------------\n")
        f.write("         RESULTS        \n")
        f.write("------------------------\n")
        f.write(f"Right: Wrong: Error = ({right}: {wrong}: {error}) / {sum}\n")
        f.write(f"ACCURACY: {right / sum * 100:.1f}%\n")
        f.write("------------------------")
    if args.intervene:
        with open(os.path.join(args.output_dir, f"interv_evaluate", f"{args.perspective}_{args.direction}_{args.belief}.txt"), 'a') as f:
            f.write(f"{args.alpha},{args.K},{right},{wrong},{error},{right/sum*100:.1f}\n")

if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    ensure_dir(os.path.join(args.output_dir, f"interv_evaluate/alpha{args.alpha}_k{args.K}/{args.perspective}_{args.direction}" if args.intervene else "evaluate"))
    evaluate(args)