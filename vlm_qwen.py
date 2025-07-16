import torch
import numpy as np
from functools import partial
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def parse_chat_response(response):
    response = response.strip()
    return response.replace("```json", "").replace("```", "")

def attn_hook(module, input, output, head_dim, layer_id, alpha, interventions_dict):
    if layer_id in interventions_dict:
        device = output[0].device
        for (head, dir, std, _) in interventions_dict[layer_id]:
            if not isinstance(dir, torch.Tensor):
                dir = torch.tensor(dir, device=device)
            else:
                dir = dir.to(device)
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std, device=device)
            else:
                std = std.to(device)
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.to(device)

            output[0][0, -1, head * head_dim: (head + 1) * head_dim] += alpha * std * dir
                
    return output

class VLM():
    def __init__(self, args):
        model_dir = args.model
        self.args = args

        self.processor = AutoProcessor.from_pretrained(model_dir, device_map="balanced_low_0")
        self.processor.patch_size = 14
        self.processor.vision_feature_select_strategy = "default"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto", low_cpu_mem_usage=True)

        self.model.config.return_dict = True
        self.model.config.output_attentions = False
        self.model.config.output_hidden_states = False
        self.model.config.eos_token_id = self.processor.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.eos_token_id

        if args.temperature == 0:
            self.model.config.do_sample = False
            self.model.config.temperature = None
            self.model.config.top_p = None
            self.model.config.top_k = None
            self.model.config.eos_token_id = None
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
    def create_conversation(self, text, video):

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video,
                        "max_pixels": 294 * 420,
                        "fps": 3.0,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        self.inputs = self.processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=False, truncation=False).to("cuda:0")

    def forward(self):
        with torch.no_grad():
            output = self.model(
                **self.inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True
            )
            attentions = output.hidden_states[1:]
            all_attention_states = []
            for layer in attentions:
                atts = layer[0].cpu().to(torch.float32).numpy()
                all_attention_states.append(atts.reshape(atts.shape[0], 28, -1))
            all_attention_states = np.array(all_attention_states)
            all_attention_states = all_attention_states[:,-1]
            if np.isnan(all_attention_states).any():
                raise ValueError("Has Value NAN.")
            return all_attention_states
        
    def generate(self):
        with torch.no_grad():
            output_ids = self.model.generate(**self.inputs, max_new_tokens=self.args.max_new_tokens)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(self.inputs.input_ids, output_ids)
            ]
            decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            return parse_chat_response(decoded)
        
    def load_data(self):
        import os
        from tqdm import tqdm
        from utils import load_json

        attentions = []
        labels = []

        for belief in ['TrueBelief', 'FalseBelief']:
            annotations = load_json(self.args.annotation)
            annotations = annotations[-148:]
            for index, annotation in tqdm(enumerate(annotations), total=len(annotations), desc=f"{self.args.perspective}_{belief} DataLoading"):
                for belief_statement in ['true', 'false']:
                    attn = np.load(os.path.join(self.args.output_dir, f'attn_{belief}', f'{index}_{belief_statement}_attn.npy'))
                    attentions.append(attn)
                    if self.args.perspective == 'protagonist':
                        label = (belief_statement=='true')
                    elif self.args.perspective == 'oracle':
                        label = (belief[:-6].lower()==belief_statement)
                    else:
                        raise NotImplementedError
                    labels.append(label)

        attentions = np.array(attentions)
        labels = np.array(labels)
        return attentions, labels
        
    def load_interv(self):
        import os
        from utils import normalize_vectors, get_interventions_dict, find_largest_k_items

        all_activations, _ = self.load_data()

        if 'multi' in self.args.direction or 'random' in self.args.direction:
            val_acc_multinomial = np.load(os.path.join(self.args.output_dir, "probe", f'multinomial', "val_acc.npy"))
            coefs_multinomial = np.load(os.path.join(self.args.output_dir, "probe", f'multinomial', "coef.npy"))
            coefs_multinomial = normalize_vectors(coefs_multinomial)
            top_heads = find_largest_k_items(val_acc_multinomial, self.args.K)
        else:
            val_acc_all = np.load(os.path.join(self.args.output_dir, "probe", f'{self.args.perspective}', "val_acc.npy"))
            coefs_all = np.load(os.path.join(self.args.output_dir, "probe", f'{self.args.perspective}', "coef.npy"))
            coefs_all = normalize_vectors(coefs_all)
            top_heads = find_largest_k_items(val_acc_all, self.args.K)
        # Single Logistic Regression Directions
        if self.args.direction == "Coef":
            directions = coefs_all
        # Multinomial Logistic Regression Directions
        elif self.args.direction == "multi_o0p0":
            directions = coefs_multinomial[:,:,0,:]
        elif self.args.direction == "multi_o0p1":
            directions = coefs_multinomial[:,:,1,:]
        elif self.args.direction == "multi_o1p0":
            directions = coefs_multinomial[:,:,2,:]
        elif self.args.direction == "multi_o1p1":
            directions = coefs_multinomial[:,:,3,:]
        elif self.args.direction == "random":
            shape = coefs_multinomial.shape
            np.random.seed(self.args.seed)
            random_array = np.random.randn(shape[0], shape[1], shape[3])
            directions = normalize_vectors(random_array)
        else:
            raise NotImplementedError
        self.interventions_dict = get_interventions_dict(all_activations, top_heads, directions=directions)

    def generate_interv(self):
        hooks = []
        for id, layer in enumerate(self.model.model.layers):
            hook = partial(attn_hook, head_dim=int(self.model.config.hidden_size / self.model.config.num_hidden_layers), layer_id=id, alpha=self.args.alpha, interventions_dict=self.interventions_dict)
            hooks.append(layer.self_attn.register_forward_hook(hook))

        with torch.no_grad():
            output_ids = self.model.generate(**self.inputs, max_new_tokens=self.args.max_new_tokens)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(self.inputs.input_ids, output_ids)
            ]
            decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            for hook in hooks:
                hook.remove()
            return parse_chat_response(decoded)