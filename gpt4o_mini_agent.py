import base64
import json
import logging
import os
import re
import tempfile
import time
from traceback import StackSummary
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, List

import backoff
import openai
import requests
import tiktoken
from PIL import Image
from google.api_core.exceptions import InvalidArgument, ResourceExhausted, InternalServerError, BadRequest
from requests.exceptions import SSLError

from mm_agents.accessibility_tree_wrap.heuristic_retrieve import filter_nodes, draw_bounding_boxes
from mm_agents.prompts import GPT4O_MINI_BOTH_OUT_CODE_PROMPT2,SYS_PROMPT_IN_BOTH_OUT_CODE,GPT_40_Prompt,summary_prompt

OPENAI_API_KEY = ''
logger = logging.getLogger("desktopenv.agent")

pure_text_settings = ['a11y_tree']

attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"
# More namespaces defined in OSWorld, please check desktop_env/server/main.py


# Function to encode the image
def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')



def linearize_accessibility_tree(accessibility_tree, platform="ubuntu"):

    if platform == "ubuntu":
        _attributes_ns = attributes_ns_ubuntu
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
        _value_ns = value_ns_ubuntu
    elif platform == "windows":
        _attributes_ns = attributes_ns_windows
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    linearized_accessibility_tree = ["tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"]

    # Linearize the accessibility tree nodes into a table format
    for node in filtered_nodes:
        if node.text:
            text = (
                node.text if '"' not in node.text \
                    else '"{:}"'.format(node.text.replace('"', '""'))
            )

        elif node.get("{{{:}}}class".format(class_ns_windows), "").endswith("EditWrapper") \
                and node.get("{{{:}}}value".format(_value_ns)):
            node_text = node.get("{{{:}}}value".format(_value_ns), "")
            text = (node_text if '"' not in node_text \
                        else '"{:}"'.format(node_text.replace('"', '""'))
                    )
        else:
            text = '""'

        linearized_accessibility_tree.append(
            "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format(
                node.tag, node.get("name", ""),
                text,
                node.get("{{{:}}}class".format(_attributes_ns), "") if platform == "ubuntu" else node.get("{{{:}}}class".format(class_ns_windows), ""),
                node.get("{{{:}}}description".format(_attributes_ns), ""),
                node.get('{{{:}}}screencoord'.format(_component_ns), ""),
                node.get('{{{:}}}size'.format(_component_ns), "")
            )
        )

    return "\n".join(linearized_accessibility_tree)


def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(linearized_accessibility_tree)
    if len(tokens) > max_tokens:
        linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
        linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree


class MyAgent:
    def __init__(
            self,
            platform="ubuntu",
            model="gpt-4o-mini",
            max_tokens=1500,
            top_p=0.9,
            temperature=0.5,
            action_space="computer_13",
            observation_type="screenshot_a11y_tree",
            # observation_type can be in ["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]
            max_trajectory_length=3,
            a11y_tree_max_tokens=10000,
            client_password="password"
    ):
        self.platform = platform
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.client_password = client_password
        self.step_count = 0
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.summary = []
        if observation_type == "screenshot_a11y_tree":
            # if action_space == "computer_13":
            #     self.system_message = GPT4O_MINI_BOTH_OUT_PROMPT 
            if action_space == "pyautogui":
                self.system_message = summary_prompt
            else:
                raise ValueError("Invalid action space: " + action_space)
      
        else:
            raise ValueError("Invalid experiment type: yes " + observation_type)
        
        self.system_message = self.system_message.format(CLIENT_PASSWORD=self.client_password)

    #function to parse actions
    
    def parse_actions(self, response: str, masks=None):
        if self.observation_type in ["screenshot", "a11y_tree", "screenshot_a11y_tree"]:
            if self.action_space == "pyautogui":
                actions = self.parse_code_from_response(response)  # Call as method
            else:
                raise ValueError("Invalid action space: " + self.action_space)

            self.actions.append(actions)
            return actions

    #function to parse summary from response
    def parse_summary(self, response: str):
        summ = self.parse_summary_from_response(response)  # Call as method
        self.summary.append(summ)    
        return summ
    
    def parse_code_from_response(self, response):
        """
        Extract Python code blocks from response
        """
        if not response:
            return []
        
        # Find all code blocks
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        
        # If no python-specific blocks, try generic code blocks
        if not code_blocks:
            code_blocks = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)
        
        codes = []
        for block in code_blocks:
            block = block.strip()
            if block in ['WAIT', 'DONE', 'FAIL']:
                codes.append(block)
            else:
                codes.append(block)
        
        return codes
    
    def parse_summary_from_response(self, response):
        """Extract summary and auto-number it"""
        if not response:
            return None
        
        # Find summary block
        summary_match = re.search(r'```SUMMARY\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        
        if summary_match:
            raw_summary = summary_match.group(1).strip()
            self.step_count += 1
            numbered_summary = f"Step {self.step_count}: {raw_summary}"
            return numbered_summary
        return None

    def predict(self, instruction: str, obs: Dict) -> List:
        """
        Predict the next action(s) based on the current observation.
        """
        system_message = self.system_message + "\nYou are asked to complete the following task: {}".format(instruction)

        # Prepare the payload for the API call
        messages = []
        masks = None

        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                },
            ]
        })

        if self.observation_type in ["screenshot_a11y_tree"]:
            base64_image = encode_image(obs["screenshot"])
            linearized_accessibility_tree = linearize_accessibility_tree(accessibility_tree=obs["accessibility_tree"],
                                                                         platform=self.platform) if self.observation_type == "screenshot_a11y_tree" else None
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(linearized_accessibility_tree,
                                                                        self.a11y_tree_max_tokens)

            if self.observation_type == "screenshot_a11y_tree":
                self.observations.append({
                    "screenshot": base64_image,
                    "accessibility_tree": linearized_accessibility_tree
                })

            # Build the user message content
            content_blocks = []

            # Add previous summary if available
            if self.summary:
                content_blocks.append({
                    "type": "text",
                    "text": f"PREVIOUS ACTION SUMMARY:\n{chr(10).join(self.summary) if self.summary else 'No previous actions'}\n\nBased on this progress, what's the next step to complete the task? If you think the task is completed, return ```DONE```."})
            else:
                content_blocks.append({
                    "type": "text", 
                    "text": "This is the first action. What's the first step you will do to complete the task?\n\n"
                })

            # Add accessibility tree if available
            if linearized_accessibility_tree:
                content_blocks.append({
                    "type": "text",
                    "text": f"Current Accessibility Tree:\n{linearized_accessibility_tree}"
                })

            # Add current screenshot
            content_blocks.append({
                "type": "text",
                "text": "Current Screenshot:"
            })
            content_blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high"
                }
            })

            messages.append({
                "role": "user",
                "content": content_blocks
            })
                    
        else:
            raise ValueError("Invalid observation_type type: " + self.observation_type)  # 1}}}

        with open("messages.json", "w") as f:
             f.write(json.dumps(messages, indent=4))
        try:
            response = self.call_llm({
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature
            })
        except Exception as e:
            logger.error("Failed to call" + self.model + ", Error: " + str(e))
            response = ""

        logger.info("RESPONSE: %s", response)
        try:
            actions = self.parse_actions(response, masks)
            summary = self.parse_summary(response)
            self.thoughts.append(response)
        except ValueError as e:
            print("Failed to parse action from response", e)
            actions = None
            self.thoughts.append("")
        logger.info("Summary: %s", self.summary)

        return response, actions

 

    @backoff.on_exception(
        backoff.constant,
        # here you should add more model exceptions as you want,
        # but you are forbidden to add "Exception", that is, a common type of exception
        # because we want to catch this kind of Exception in the outside to ensure each example won't exceed the time limit
        (
                # General exceptions
                SSLError,

                # OpenAI exceptions
                openai.RateLimitError,
                openai.BadRequestError,
                openai.InternalServerError,
        ),
        interval=30,
        max_tries=10
    )

    def call_llm(self, payload):
        if self.model.startswith("gpt-4o-mini"):
            # Support custom OpenAI base URL via environment variable
            base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com')
            # Smart handling: avoid duplicate /v1 if base_url already ends with /v1
            api_url = f"{base_url}/chat/completions" if base_url.endswith('/v1') else f"{base_url}/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            max_retries = 5
            retry_delay = 10  # Start with 10 seconds
            
            for attempt in range(max_retries):
                logger.info("Generating content with GPT model: %s (attempt %d/%d)", self.model, attempt + 1, max_retries)
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=30  # Add timeout to prevent hanging
                )

                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                    
                elif response.status_code == 429:  # Rate limit
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', '')
                    
                    # Extract wait time from error message if possible
                    if 'try again in' in error_msg:
                        import re
                        match = re.search(r'try again in ([0-9.]+)s', error_msg)
                        if match:
                            retry_delay = float(match.group(1)) + 10  # Add buffer
                        logger.warning("Rate limit hit, waiting %.1f seconds...", retry_delay)
                    else:
                        logger.warning("Rate limit hit (no specific wait time), waiting %.1f seconds...", retry_delay)
                    
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                    
                elif response.status_code != 200:
                    if response.json().get('error', {}).get('code') == "context_length_exceeded":
                        logger.error("Context length exceeded. Retrying with a smaller context.")
                        payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]
                        # Don't continue to retry loop, just try once with shorter context
                        retry_response = requests.post(
                            api_url,
                            headers=headers,
                            json=payload,
                            timeout=30
                        )
                        if retry_response.status_code == 200:
                            return retry_response.json()['choices'][0]['message']['content']

                    logger.error("Failed to call LLM: " + response.text)
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        return ""

            logger.error("All %d retry attempts failed", max_retries)
            return ""
        
        else:
            raise ValueError("Invalid model: " + self.model)
        
    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.agent")

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.summary = []
        self.step_count = 0
