
from openai import AsyncOpenAI
from core.logger import Severity
from core.text_ai import TextAI

class ChatGPTHandler(TextAI):
    def friendly_identifier(self):
        return "OpenAI ChatGPT (Cloud)"

    async def get_llm_message(self, message: str):
        self.load_conversation()
        self._conversation.add_message(message, "user")

        messages = self._prepare_conversation_data()
        try:
            client = AsyncOpenAI(
                api_key=self._cfg.openai_api_key,
            )

            chat_completion = await client.chat.completions.create(
                messages=messages,
                model=self._cfg.large_language_model
            )

            response_message = chat_completion.choices[0].message
            role: str = response_message["role"]
            content: str = response_message["content"]

            self._conversation.add_message(content, role)

            return content
        except Exception as e:
            self._logger.error(str(e), severity=Severity.MEDIUM)
            return "Something went wrong 😭"