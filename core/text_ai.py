from abc import ABC, abstractmethod
import json
import os
from typing import Any, Coroutine

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

class TextAI(ABC):
    def __init__(self, cfg: Configuration) -> None:
        self._cfg = cfg
        self.__conversation_file = "conversation.json"
        self.__system_message_file = "system_message.txt"
        self._conversation: Conversation
        self._logger = Logger("TextAI")
        
    def get_system_message(self):
        try:
            with open(self.__system_message_file, "r") as f:
                return f.read()
        except FileNotFoundError as e:
            self._logger.error(f"Could not find {self.__system_message_file}: {str(e)}", severity=Severity.MEDIUM)
            return "You are now Walter White from Breaking Bad"
    
    def load_conversation(self) -> Conversation:
        try:
            with open(self.__conversation_file, "r") as f:
                out_json = json.load(f)
                
                self._conversation = Conversation(self._cfg.max_stored_messages)
                
                for ollama_message in out_json:
                    self._conversation.add_message(ollama_message["content"], ollama_message["role"])
        except FileNotFoundError as e:
            self._logger.warning(f"Conversation file doesn't exist {str(e)}", severity=Severity.LOW)
            self._conversation = Conversation(self._cfg.max_stored_messages)
        except Exception as e:
            self._logger.error(f"Uncaught exception: {str(e)}", severity=Severity.HIGH)
            self._conversation = Conversation(self._cfg.max_stored_messages)
    
    def save_conversation(self):
        with open(self.__conversation_file, "w") as f:
            json.dump(self._conversation.to_dict(), f, indent=4)
    
    def clear_conversation(self):
        self.load_conversation()

        if os.path.exists(self.__conversation_file):
            os.remove(self.__conversation_file)
            
        return len(self._conversation.messages)
    
    def _prepare_conversation_data(self) -> list[dict[str, str]]:
        """
        Formats the messages as a list of dictionaries, each structured as follows:

        [
            {
                "role": "system",
                "content": "The system message"
            },
            {
                "role": "user",
                "content": "The user's message"
            },
            {
                "role": "assistant",
                "content": "The large language model's response"
            }
        ]

        Returns:
            list[dict[str, str]]: The formatted conversation, including the system message.
        """
        messages = [
            {
                "role": "system",
                "content": self.get_system_message()
            }
        ]

        conversation_dict = self._conversation.to_dict()

        messages.extend(conversation_dict)

        return messages
    
    @abstractmethod
    async def get_llm_message(self, message: str) -> Coroutine[Any, Any, str]:
        """
        This function will get the message from the large language model.

        Returns: 
            Coroutine[Any, Any, str]: with a string as a message.
        """

    @property
    @abstractmethod
    def friendly_identifier(self) -> str:
        """
        A friendly name to identify what service this TextAI is using.

        Returns:
            str: The friendly name like "ChatGPT" or "Ollama" Etc
        """