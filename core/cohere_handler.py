from typing import Any, Coroutine

import cohere
from core.text_ai import TextAI


class CohereHandler(TextAI):
    def friendly_identifier(self):
        return "CoHere (Cloud)"
    
    async def get_llm_message(self, message: str):
        self.load_conversation()
        self._conversation.add_message(message, "user")

        messages = self._prepare_conversation_data()
        
        co = cohere.AsyncClientV2(
            api_key=self._cfg.cohere_api_key,
        )

        chat = await co.chat(
            model=self._cfg.large_language_model,
            messages=messages
        )
                
        self._conversation.add_message(chat.message.content[0].text, chat.message.role)
        
        self.save_conversation()
        
        return chat.message.content[0].text