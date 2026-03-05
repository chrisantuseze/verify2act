import numpy as np
from openai import OpenAI
import os
from PIL import Image
from io import BytesIO
import base64


def pil_to_base64(image_pil, size=(336, 336)):
    image_pil = image_pil.resize(size)
    buffered = BytesIO()
    image_pil.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

class GPTAgent(object):

    def __init__(self, api_key=None, model='gpt-4o'):
        api_key = os.environ.get("OPENAI_API_KEY", api_key)
        if api_key is None:
            raise ValueError("Please set OpenAI api_key!")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = "You are an intelligent robot equipped with cameras and robotic arms, your primary task is to observe and interact with the objects on the desktop."

    def parse_prompt(self, prompt, image_list):
        prompt_list = prompt.split('<image>')
        user_prompts = []
        for i, prompt in enumerate(prompt_list):
            user_prompts.append({"type": "text", "text": prompt})
            if i < len(image_list) and image_list[i] is not None:
                user_prompts.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_list[i]}"}
                })
        
        user_prompts.append({"type": "text", "text": "You can only output the action, e.g., pick up red. Do not output anything else."})

        final_prompt = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": user_prompts}
        ]
        return final_prompt
    
    def act(self, image, goal_image, inp, next_image=None):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image = pil_to_base64(image)
            if goal_image is not None:
                goal_image = Image.fromarray(goal_image)
                goal_image = pil_to_base64(goal_image)
            if next_image is not None:
                next_image = Image.fromarray(next_image)
                next_image = pil_to_base64(next_image)
        image_list = [goal_image, image, next_image]
        final_prompt = self.parse_prompt(inp, image_list)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=final_prompt,
        )
        result = response.choices[0].message.content
        return result.strip()