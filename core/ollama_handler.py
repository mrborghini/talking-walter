import httpx

from typing import Any, Coroutine
from core.logger import Severity
from core.text_ai import TextAI

class OllamaHandler(TextAI):
    def friendly_identifier(self):
        return "Ollama (Local)"
    
    async def get_llm_message(self, message: str):
        self.load_conversation()
        self._conversation.add_message(message, "user")

        messages = self._prepare_conversation_data()

        async with httpx.AsyncClient() as client:
            content = {
                "model": self._cfg.large_language_model,
                "messages": messages,
                "stream": False,
            }

            try:
                response = await client.post(f"{self._cfg.ollama_url}/api/chat", json=content, timeout=None)
                
                text_message = response.json()
                
                self._conversation.add_message(text_message["message"]["content"], text_message["message"]["role"])
                
                self.save_conversation()
                
                return text_message["message"]["content"]
            except Exception as e:
                self._logger.error(str(e), severity=Severity.MEDIUM)
                return "Something went wrong 😭"