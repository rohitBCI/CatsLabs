import re
import regex
import openai
import tiktoken
import traceback
from decouple import config
import googletrans
from googletrans import Translator
from datetime import datetime
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

from reply_agent import Reply_Agent
from image_agent import Image_Agent

class Agents:
    def __init__(self):
        self.openai_api_key = config('OPENAI_API_KEY')
        openai.api_key = self.openai_api_key
        
        self.name = config('NAME')
        self.handle = config('HANDLE')
        self.handle_name = config('HANDLENAME')
        self.token_overflow = 1000
        
        self.nonsensical_keywords = [
            "@catcoin",
            "catcoin.com",
            "I'm not sure what you're asking.", 
            "That doesn't make any sense", 
            "I can't provide a coherent response",
            "I'm confused", 
            "That's beyond my comprehension",
            "I have no idea what you're talking about",
            "I'm unable to process that request", 
            "I can't generate a meaningful reply",
            "That's too complicated for me to answer",
            "I don't have the information to respond",
            "I'm drawing a blank",
            "I'm not equipped to handle that question", 
            "I'm experiencing a glitch"
            # Add more nonsensical keywords as needed
            ]
    
    '''Function to return the number of tokens in a text string'''
    def token_count(self, string, model):
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    '''Function to check for token overflow in tweet thread'''
    def check_token_overlow(self, tweet_thread, model):
        num_tokens = 0
        for tweet in tweet_thread:
            num_tokens += self.token_count(tweet, model)
        
        if num_tokens >= self.token_overflow:
            return True
        else:
            return False

    '''Function to detect the language and store information in text'''
    def detect_language(self, text):
        try:
            translator = Translator()
            dt = translator.detect(text)
            detected_language = dt.lang
            if dt.confidence < 0.90:
                detected_language = 'en'
            text = f"[{detected_language}] {text}"
            target_language= detected_language
        except Exception as e:
            traceback.print_exc()
            target_language = None
            print(f"\nWARNING: Error detecting langauge from tweet\n{str(e)}\n")
        
        return text, target_language
    
    '''Function to translate response to target language'''
    def translate_language(self, response, target_language, tweet_url=None,  error_messages=None):
        try:
            translated_response = response
            if target_language is not None:
                translator = Translator()
                translation = translator.translate(response, dest=target_language)
                translated_response = translation.text
        except Exception as e:
            traceback.print_exc()
            translated_response = response
            print(f"\nWARNING: Error translating response to target language\nurl: {tweet_url} error:, {str(e)}\n")
            error_messages.append("Error translating response to target language")
        
        return translated_response, error_messages

    '''Function to trim agent handle from tweet threads containing multiple tweets'''
    def trim_tweet_thread(self, tweet_thread):
        tweet_thread_trimmed = []
        if len(tweet_thread) > 1 :
            for tweet in tweet_thread:
                tweet_username, tweet_text = re.split(r':\s+', tweet, maxsplit=1)
                pattern = fr'(?i)@{re.escape(self.handle_name.lower())}\b'
                extracted_text = re.sub(pattern, '', tweet_text.strip())
                
                extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
                if len(extracted_text) > 0:
                    tweet_thread_trimmed.append(tweet)
        else:
            tweet_thread_trimmed = tweet_thread
        
        return tweet_thread_trimmed

    '''Function to trim incomplete sentence from respomse'''
    def trim_response(self, response):
        # Find the last punctuation that marks the end of a sentence
        punctuation = ['.', '!', '?', '"']
        last_punctuation_index = 0
        last_punctuation_index_excess = 0
        response_untrimmed = response

        # BODGE: Handle bullet points
        if "1." in response and response[-1] not in punctuation:
            response += "."

        # BODGE: Handle hashtags
        for p in punctuation:
            matches = list(re.finditer(rf'(?<!\d{re.escape(p)})(?<!\d{re.escape(p)}\d){re.escape(p)}(?!\d|\w)', response))
            for match in matches:
                current_punctuation_index_excess = max(last_punctuation_index_excess, match.start() + 1)
                sentence_excess = response[last_punctuation_index:current_punctuation_index_excess]
                last_punctuation_index_excess = current_punctuation_index_excess

        excess_response = response[last_punctuation_index_excess:]
        if "#" in excess_response and response[-1] not in punctuation:
            response += "."

        if response.startswith("AskHoshi"):
            response = response.replace("AskHoshi", "")
            response = re.sub(r'\s+', ' ', response).strip()

        # Handle invalid text in response
        if response.startswith("<"):
            response = re.sub(r"<[^>]+>", "", response)
            response = re.sub(r'\s+', ' ', response).strip()

        # Handle brackets
        if response.startswith("["):
            response = re.sub(r"\[\w{2}\]\s*", "", response)
            response = re.sub(r"\[([^]]+)\]", r"\1", response)
            response = re.sub(r'\s+', ' ', response).strip()

        # BODGE: Handle edge case
        if "[en]" in response:
            response = response.replace("[en]", "")
            response = re.sub(r'\s+', ' ', response).strip()

        if "[ja]" not in response_untrimmed:
            # Find the last punctuation mark that is not followed by a number
            for p in punctuation:
                matches = list(re.finditer(rf'(?<!\d{re.escape(p)})(?<!\d{re.escape(p)}\d){re.escape(p)}(?!\d|\w)', response))
                for match in matches:
                    current_punctuation_index = max(last_punctuation_index, match.start() + 1)
                    sentence = response[last_punctuation_index:current_punctuation_index]
                    last_punctuation_index = current_punctuation_index

            # Trim the response by removing the unfinished sentence
            trimmed_response = response[:last_punctuation_index]

        else:
            trimmed_response = response

        # Return None if trimmed_response is empty
        if trimmed_response == "":
            trimmed_response = None

        # BODGE: Incorrect handle in response
        if trimmed_response!=None:
            trimmed_response = trimmed_response.replace("OpenAI", "CatsLabs")

        return trimmed_response
    
    '''Function to check if response generated conveys similar meaning to existing prior knowledge'''
    def response_similarity(self, prior_knowledge, response):    
        similairty_flag = False

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        response_embedding = model.encode([response])[0]
        
        for sentence in prior_knowledge:
            sentence_embedding = model.encode([sentence])[0]
            similarity = 1 - cosine(response_embedding, sentence_embedding)
            
            if similarity >= 0.9:
                similairty_flag = True
                break
        
        return similairty_flag

    '''Function to check whether the prompt is safe to pass into the model'''
    def moderation_agent(self, sentence):
        errors = {
            "hate": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
            "hate/threatening": "Hateful content that also includes violence or serious harm towards the targeted group.",
            "self-harm": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
            "sexual": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
            "sexual/minors": "Sexual content that includes an individual who is under 18 years old.",
            "violence": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
            "violence/graphic": "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
        }
        response = openai.Moderation.create(input=sentence)
        if response.results[0].flagged:
            result = [
                error
                for category, error in errors.items()
                if response.results[0].categories[category]
            ]
            return result
        return None

    '''Function to generate prompt to be passed to the Agent'''
    def generate_prompt(self, prior_knowledge, tweet_thread, tweet_category):
        messages = []
        messages_content = []
        target_language = 'en'
        if 'response' in tweet_category:
            # Construct messages from prior knowledge
            for sentence in prior_knowledge:
                # Extract the username from the sentence
                username = re.search(r'^(\w+):', sentence)
                username = username.group(1)

                # Remove the username prefix from the sentence
                text = re.sub(r'^\w+:', '', sentence).strip()
                # Remove additional spaces
                text = re.sub(r'\s+', ' ', text).strip()

                if username == "System":
                    messages.append({"role": "system", "content": text})
                elif username == "Assistant":
                    messages.append({"role": "assistant", "content": text})
                else:
                    messages.append({"role": "system", "content": text})

            # Construct messages from tweet_thread
            for sentence in tweet_thread:
                # Extract the username and sentence from the tweet
                match = re.search(r'^@(\w+):\s*(.*)$', sentence)
                if match:
                    username = match.group(1)
                    sentence = match.group(2)

                    # Remove additional spaces
                    text = re.sub(r'\s+', ' ', sentence).strip()

                    # Store detected language in text
                    text, target_language = self.detect_language(text)

                role = "assistant" if username.lower() == self.handle_name.lower() else "user"

                messages.append({"role": role, "content": text, "name": username})
                messages_content.append(sentence)
        
            return messages, messages_content, target_language
        
        else:
            
            # Construct messages from prior knowledge
            for sentence in prior_knowledge:
                messages.append({"role": "system", "content": sentence, "name": self.handle_name})
                messages_content.append(sentence)
            
            # Construct messages from previous tweets
            if len(tweet_thread) != 0:
                if tweet_category == 'catfacts.tweet':
                    messages.append({"role": "system", "content": tweet_thread[-1], "name": self.handle_name})
                for sentence in tweet_thread:
                    messages_content.append(sentence)

            return messages, messages_content

    '''Function to generate response from an Agent'''
    def generate_response(self, model, prompt, temperature, max_tokens, frequency_penalty, presence_penalty, n=1, stop=None):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0.7,
            max_tokens=50,
            frequency_penalty=0,
            presence_penalty=0.6,
            n=1,
            stop=None
        )

        # Extract the generated response
        generated_response = response['choices'][0]['message']['content'].strip()
        return generated_response
    
    '''Function to generate image from an Agent'''
    def generate_image(self, prompt, size, n=1):
        response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=size
        )

        # Extract the generated image
        image_url = response['data'][0]['url']
        return image_url

    '''Functon to generate output from an agent'''
    def generate_agent_output(self, agent, tweet_thread, url=None, error_messages=None, tweet_category=None):
        if 'response' in tweet_category:
            agent_instance = Reply_Agent(api_key=self.openai_api_key, agent=agent, handle=self.handle, handle_name=self.handle_name, name=self.name, tweet_url=url, error_messages=error_messages, tweet_category=tweet_category)
            generated_response, error_messages = agent_instance.run_agent(tweet_thread)
            return generated_response, error_messages

        elif 'image' in tweet_category:
            agent_instance = Image_Agent(api_key=self.openai_api_key, agent=agent, handle=self.handle, handle_name=self.handle_name, name=self.name, tweet_url=url, error_messages=error_messages, tweet_category=tweet_category)
            trimmed_prompt, generated_image = agent_instance.run_agent(tweet_thread)
            return trimmed_prompt, generated_image

        elif 'tweet' in tweet_category:
            agent_instance = Tweet_Agent(api_key=self.openai_api_key, agent=agent, handle=self.handle, handle_name=self.handle_name, name=self.name, tweet_category=tweet_category)
            generated_response = agent_instance.run_agent(tweet_thread)
            return generated_response

if __name__ == "__main__":
    
    #tweet_thread = ["@GhahramaniTaj: ow hoshi. a beautifuL souL and sun shine. this poem is for you: if hoshi wasn't here, i wouLd crave sun, i wouLd crave moon, i wouLd crave for hoshi, for aLL hoshies to be, here and forever B and C."]
    tweet_thread = ["@Crypto_Wheein: Do you know why cats ddon't like water?"]
    agent = Agents()
    generated_response, error_messages = agent.generate_agent_output(agent=agent, tweet_thread=tweet_thread, url=None, error_messages=[], tweet_category='mention.response')
    print(generated_response)