import av
import os
import time
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

    args = parser.parse_args()
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    return args

def write_error(i, itemid, res, ans, args):
    error_file = os.path.join(
        args.output_dir, 
        "init_belief", 
        f"errors.txt"
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
    visualLLM = VLM(args)
    with open(os.path.join(args.output_dir, "init_belief", "results.txt"), 'w') as f:
        f.write(time.strftime('Output Time: %Y-%m-%d %H:%M:%S\n------------------------\n', time.localtime(time.time())))
        with tqdm(total=len(annotations)*(len(annotations[0]["init_belief"])), dynamic_ncols=True) as pbar:
            for index, annotation in enumerate(annotations):
                init_beliefs = annotation["init_belief"]
                description = annotation['init_env_desc']
                key_frames = get_key_frames(annotation['nodes'])
                if "LLaVA-NeXT-Video" in args.model:
                    contrainer = av.open(os.path.join(args.dataset, str(annotation['index']), "FalseBelief/full/fps3.mp4"))
                for init_belief in init_beliefs:
                    question = init_belief['question']
                    answer = init_belief['answer']
                    colors = annotation['colors']
                    colors = list(colors.values())
                    text = f"Description: {description}\nQuestion: {question}"
                    if "Qwen2-VL" in args.model:
                        clip = [os.path.join(args.dataset, str(annotation['index']), f"FalseBelief/full/{i}.png") for i in key_frames if init_belief['section'][0]<= i <=init_belief['section'][1]]
                    elif "LLaVA-NeXT-Video" in args.model:
                        indices = [x for x in key_frames if init_belief['section'][0]<= x <=init_belief['section'][1]]
                        clip = read_video_pyav(args, contrainer, indices)
                    visualLLM.create_conversation(text, clip)
                    res = visualLLM.generate()
                    norm_res = res.lower()

                    if isinstance(answer, list):
                        if "LLaVA-NeXT-Video" in args.model:
                            # Formatting issues
                            norm_res = norm_res.replace("'","").replace(".", "").replace("]","").replace("[","").strip().split(" ")
                        if answer[0] in norm_res and answer[1] in norm_res and colors[2] not in norm_res and colors[3] not in norm_res and colors[4] not in norm_res:
                            right += 1
                            f.write(f"Right. ID:{index},ItemId:{sum},Res:{norm_res},Ans:{answer}\n")
                        elif answer[0] in norm_res or answer[1] in norm_res or colors[2] in norm_res or colors[3] in norm_res or colors[4] in norm_res:
                            wrong += 1
                            f.write(f"Wrong. ID:{index},ItemId:{sum},Res:{norm_res},Ans:{answer}\n")
                        else:
                            write_error(index, sum, norm_res, answer, args)
                            error += 1
                    else:
                        if norm_res == answer:
                            right += 1
                            f.write(f"Right. ID:{index},ItemId:{sum},Res:{norm_res},Ans:{answer}\n")
                        elif norm_res in colors:
                            wrong += 1
                            f.write(f"Wrong. ID:{index},ItemId:{sum},Res:{norm_res},Ans:{answer}\n")
                        else:
                            write_error(index, sum, norm_res, answer, args)
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

if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    ensure_dir(os.path.join(args.output_dir, "init_belief"))
    evaluate(args)