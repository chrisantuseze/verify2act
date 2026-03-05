import numpy as np
import os
from PIL import Image
from google import genai

def image_resize(img, size=(336, 336)):
    return img.resize(size)


class GeminiAgent(object):

    def __init__(self, api_key=None, model='gemini-2.0-flash-thinking-exp-1219'):
        api_key = os.environ.get("GOOGLE_API_KEY", api_key)
        if api_key is None:
            raise ValueError("Please set Google api_key!")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        # self.system_prompt = "You are an intelligent robot equipped with cameras and robotic arms, your primary task is to observe and interact with the objects on the desktop."

    def parse_prompt(self, prompt, image_list):
        prompt_list = prompt.split('<image>')
        user_prompts = []
        for i, prompt in enumerate(prompt_list):
            user_prompts.append(prompt)
            if i < len(image_list) and image_list[i] is not None:
                user_prompts.append(image_list[i])
        
        user_prompts.append("You can only output the action, e.g., pick up red. Do not output anything else.")

        return user_prompts
    
    def act(self, image, goal_image, inp, next_image=None):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image = image_resize(image)
            if goal_image is not None:
                goal_image = Image.fromarray(goal_image)
                goal_image = image_resize(goal_image)
            if next_image is not None:
                next_image = Image.fromarray(next_image)
                next_image = image_resize(next_image)
        image_list = [goal_image, image, next_image]
        final_prompt = self.parse_prompt(inp, image_list)
        response = self.client.models.generate_content(
            model=self.model,
            contents=final_prompt
        )
        if 'thinking' in self.model:
            answer = response.candidates[0].content.parts[1].text
        else:
            answer = response.text
        return answer.strip()