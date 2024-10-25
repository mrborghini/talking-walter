import json
import os

import httpx

from core.config_reader import Configuration
from core.logger import Logger, Severity

class OllamaMessage:
    content: str
    author: str

class Conversation:
    messages: list[OllamaMessage]
    
    def __init__(self, max_messages: int) -> None:
        self.messages = []
        self.max_messages = max_messages
    
    def add_message(self, content: str, author: str):
        new_message = OllamaMessage()
        new_message.author = author
        new_message.content = content
        self.messages.append(new_message)
        self.trim_messages()
    
    def trim_messages(self):
        if self.max_messages == 0:
            return
        
        # Remove the max
        if len(self.messages) == self.max_messages + 1:
            self.messages.pop(0)
            return
        
        # If more than allowed messages are detected remove multiple
        if len(self.messages) > self.max_messages + 1:
            for _ in self.messages:
                if len(self.messages) == self.max_messages:
                    return
                
                self.messages.pop(0)
        
    def to_dict(self):
        return [{"role": message.author, "content": message.content} for message in self.messages]
    
    def to_string(self):
        return "\n".join(f"{message.author}: {message.content}" for message in self.messages)

class TextAI:
    def __init__(self, cfg: Configuration) -> None:
        self._cfg = cfg
        self.__base_url = cfg.ollama_url
        self.__ollama_model = cfg.ollama_model
        self.__conversation_file = "conversation.json"
        self.__system_message_file = "system_message.txt"
        self.__conversation: Conversation
        self.__logger = Logger("TextAI")
        
    def get_system_message(self):
        try:
            with open(self.__system_message_file, "r") as f:
                return f.read()
        except FileNotFoundError as e:
            self.__logger.error(f"Could not find {self.__system_message_file}: {str(e)}", severity=Severity.MEDIUM)
            return "You are now Walter White from Breaking Bad"
    
    def load_conversation(self) -> Conversation:
        try:
            with open(self.__conversation_file, "r") as f:
                out_json = json.load(f)
                
                self.__conversation = Conversation(self._cfg.max_stored_messages)
                
                for ollama_message in out_json:
                    self.__conversation.add_message(ollama_message["content"], ollama_message["role"])
        except FileNotFoundError as e:
            self.__logger.warning(f"Conversation file doesn't exist {str(e)}", severity=Severity.LOW)
            self.__conversation = Conversation(self._cfg.max_stored_messages)
        except Exception as e:
            self.__logger.error(f"Uncaught exception: {str(e)}", severity=Severity.HIGH)
            self.__conversation = Conversation(self._cfg.max_stored_messages)
    
    def save_conversation(self):
        with open(self.__conversation_file, "w") as f:
            json.dump(self.__conversation.to_dict(), f, indent=4)
    
    def clear_conversation(self):
        self.load_conversation()

        if os.path.exists(self.__conversation_file):
            os.remove(self.__conversation_file)
            
        return len(self.__conversation.messages)

    async def get_ollama_message(self, message: str, author: str = "user") -> str:
        self.load_conversation()
        self.__conversation.add_message(message, author)

        messages = [
            {
                "role": "system",
                "content": self.get_system_message()
            }
        ]

        conversation_dict = self.__conversation.to_dict()

        messages.extend(conversation_dict)

        async with httpx.AsyncClient() as client:
            content = {
                "model": self.__ollama_model,
                "messages": messages,
                "stream": False,
            }

            try:
                response = await client.post(f"{self.__base_url}/api/chat", json=content, timeout=None)
                
                text_message = response.json()
                
                self.__conversation.add_message(text_message["message"]["content"], text_message["message"]["role"])
                
                self.save_conversation()
                
                return text_message["message"]["content"]
            except Exception as e:
                self.__logger.error(str(e), severity=Severity.MEDIUM)
                return "Something went wrong 😭"