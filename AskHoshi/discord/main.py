import os
import discord
import requests
import traceback
from io import BytesIO
from decouple import config
from agents import Agents

class DiscordBot(discord.Client):
    def __init__(self, intents):
        super().__init__(intents=intents)
        self.agent = Agents()
        self.NAME = config('NAME')
        self.HANDLE = config('HANDLE')
        self.OPENAI_API_KEY = config('OPENAI_API_KEY')
        self.DISCORD_TOKEN = config('DISCORD_TOKEN')

        self.user_mention_limit = 500
        self.user_image_limit = 50
        self.system_image_limit = 2000
        self.conversation_data = {}
        self.responses_data = {}
        self.images_data = {}
        self.system_images_count = 0

    async def on_ready(self):
        print(f'Logged in as {self.user.name}')

    async def on_message(self, message):
        if message.author == self.user:
            return

        user_id = message.author.id
        if hasattr(message.author, 'name'):
            handle = '@' + message.author.name
        else:
            handle = '@' + message.author.display_name

        text = message.content

        mentioned_users = message.mentions
        for user in mentioned_users:
            text = text.replace(f'<@{user.id}>', f'@{user.name}')

        # Detect text language
        target_langauge = None
        _, target_langauge = self.agent.detect_language(text)

        timestamp = int(message.created_at.timestamp())

        # Check if the message is a reply
        if message.reference and message.reference.cached_message:
            replied_message = message.reference.cached_message
            thread = replied_message.content
            if replied_message.author == self.user:
                thread_handle = '@' + self.user.name
                thread_text = thread
            else:
                thread_handle = '@' + replied_message.author.name
                thread_text = thread
        else:
            thread = None
            thread_handle = ''
            thread_text = ''

        prompt = f"{handle}: {text}"

        if thread is not None:
            prompt_thread = f"{thread_handle}: {thread_text}"
        else:
            prompt_thread = None

        if 'conversation' not in self.conversation_data:
            self.conversation_data['conversation'] = {}
            self.conversation_data['responses'] = {}
            self.conversation_data['images'] = {}
            self.conversation_data['system_images'] = 0

        if user_id not in self.conversation_data['conversation']:
            self.conversation_data['conversation'][user_id] = []
            self.conversation_data['responses'][user_id] = 0
            self.conversation_data['images'][user_id] = 0

        if thread is not None:
            self.conversation_data['conversation'][user_id].append(prompt_thread)

        self.conversation_data['conversation'][user_id].append(prompt)

        if self.conversation_data['responses'][user_id] < self.user_mention_limit and "/imagine" not in text and ("@askhoshi" in text.lower() or "@AskHoshi" in thread_handle):
            try:
                if thread is not None:
                    print(f"\n(Message thread) {prompt_thread}")
                print(f"(Message) {prompt}\n")

                # Generate a response based on the conversation history
                conversation = self.conversation_data['conversation'][user_id][-10:]
                generated_response, _ = Agents().generate_agent_output(agent=Agents(), tweet_thread=conversation, url=None, error_messages=[], tweet_category='discord.response')

                # Translate generated response to target language
                if target_langauge:
                    generated_response, _ = self.agent.translate_language(generated_response, target_langauge)

                if generated_response:
                    print(f"\nResponse generated (discord.response):\n{generated_response}\n")

                    output = f"{self.HANDLE}: {generated_response}"
                    self.conversation_data['conversation'][user_id].append(output)
                    self.conversation_data['responses'][user_id] += 1

                    await message.reply(generated_response)

            except Exception as e:
                traceback.print_exc()
                print(f"\nWARNING: Error generating response: {str(e)}\n")

        if "/imagine" in text and self.conversation_data['images'][user_id] < self.user_image_limit and self.system_images_count < self.system_image_limit and "@askhoshi" in text.lower():
            try:
                # Call your existing image generation method within the Agents class
                trimmed_prompt, generated_image = Agents().generate_agent_output(agent=Agents(), tweet_thread=prompt, url=None, error_messages=[], tweet_category='discord.image') 

                if generated_image:
                    print(f"Image generated (discord.image):\n{generated_image}\n")

                    self.conversation_data['images'][user_id] += 1
                    self.system_images_count += 1

                    response = requests.get(generated_image)
                    if response.status_code == 200:
                        image_bytes = BytesIO(response.content)
                        await message.reply(trimmed_prompt, file=discord.File(image_bytes, filename=f'{trimmed_prompt}.png'))

            except Exception as e:
                traceback.print_exc() 
                print(f"\nWARNING: Error generating image: {str(e)}\n")

if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.typing = False
    intents.presences = False
    intents.message_content = True

    DISCORD_TOKEN = config('DISCORD_TOKEN')

    bot = DiscordBot(intents)
    bot.run(DISCORD_TOKEN)
