import torch
import sys
from PIL import Image
from transformers import TextStreamer
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.mm_utils import process_images


class LlavaAgent(object):

    def __init__(self, model_path, model_base=None, load_8bit=False, load_4bit=False,
                 temperature=0.2, max_new_tokens=1024, conv_mode=None, debug=False):
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        print(model_name)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name, load_8bit, load_4bit)
        print(self.model)
        print(self.model.generation_config)

        if 'llama-2' in model_name.lower():
            _conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            _conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            _conv_mode = "mpt"
        else:
            _conv_mode = "llava_v0"

        if conv_mode is not None and _conv_mode != conv_mode:
            print(f'[WARNING] the auto inferred conversation mode is {_conv_mode}, '
                  f'while parameter `conv_mode` is {conv_mode}, using {conv_mode}')
        else:
            conv_mode = _conv_mode

        self.conv_mode = conv_mode
        self.conv = None
        self.init_conv()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.debug = debug

    def init_conv(self):
        self.conv = conv_templates[self.conv_mode].copy()

    def act(self, image, goal_image, inp, next_image=None, num_propose_actions=1, return_score=False, temperature=0):
        self.init_conv()
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            if goal_image is not None:
                goal_image = Image.fromarray(goal_image)
            if next_image is not None:
                next_image = Image.fromarray(next_image)
        image_tensor = process_images([goal_image, image] if next_image is None else [goal_image, image, next_image], 
                                      self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            if image_tensor.ndim == 4:
                image_tensor = image_tensor.unsqueeze(0)

        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_dict = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=(temperature > 0),
                temperature=temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                return_dict_in_generate=True,
                output_scores=True,
                num_return_sequences=num_propose_actions,
            )

        transition_scores = self.model.compute_transition_scores(
            output_dict.sequences, output_dict.scores, normalize_logits=True
        )

        generated_tokens = output_dict.sequences[:, input_ids.shape[1]:]
        seq_scores = torch.zeros(len(output_dict.sequences)).to(self.model.device)
        for i in range(len(transition_scores)):
            for tok, score in zip(generated_tokens[i], transition_scores[i]):
                seq_scores[i] += score
                if tok == self.tokenizer.eos_token_id:
                    break

        output_ids = output_dict.sequences

        actions_with_scores = [
            (self.tokenizer.decode(output_ids[i, input_ids.shape[1]:]).strip().split('</s>')[0], s.cpu().item())
            for i, s in enumerate(seq_scores)
        ]
        actions_with_scores.sort(key=lambda x: x[1], reverse=True)        

        self.conv.messages[-1][-1] = actions_with_scores[0][0]
        
        if num_propose_actions > 1:
            if return_score:
                return actions_with_scores
            else:
                return [a for a, s in actions_with_scores]
        else:
            if return_score:
                return actions_with_scores[0]
            else:
                return actions_with_scores[0][0]
