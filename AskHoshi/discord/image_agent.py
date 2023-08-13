import re
import regex
import openai
import tiktoken
import traceback
from decouple import config
import googletrans
from googletrans import Translator
from datetime import datetime

class Image_Agent:
    def __init__(self, api_key=None, agent=None, handle=None, handle_name=None, name=None, tweet_url=None, error_messages=None, tweet_category=None):
        self.agent = agent
        self.handle = handle
        self.handle_name =  handle_name
        self.name = name
        self.tweet_url = tweet_url
        self.error_messages = error_messages
        self.tweet_category = tweet_category

        self.size = "1024x1024"

    '''Function to run Image Agent'''
    def run_agent(self, prompt):

        # Trim the prompt
        trimmed_prompt = prompt.split("/imagine")[-1]
        trimmed_prompt = trimmed_prompt.replace("/imagine ", "")
        trimmed_prompt = trimmed_prompt.replace(f"{self.handle}", "")
        trimmed_prompt = re.sub(r'\s+', ' ', trimmed_prompt).strip()

        print(f"\nPre-trimmed prompt: {trimmed_prompt}\n")

        image = self.agent.generate_image(prompt=trimmed_prompt, size=self.size)
        return trimmed_prompt, image

