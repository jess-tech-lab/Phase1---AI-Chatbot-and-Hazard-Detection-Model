from simple_llm.agent import Agent
from simple_llm.agents.delegator import Delegator, DelegatorClient
from simple_llm.clients.openai import OpenAIAgent
from typing import TypeVar, Type

T = TypeVar('T', bound=Agent)

DELEGATOR_PROMPT = """
You will be given information on a few users.
Your job is to perform context analysis and then guess which user sent the message given to you. NEVER respond to the query, only output the number corresponding to the user you think sent the message.
If you are unsure which user sent the message, output 0.
"""

class PersonaDelegator(Delegator):
    def __init__(self, agent_cls: Type[T], model: str, personas: list[str], delegator_client: DelegatorClient):
        sys_msg = self.build_system_message(personas)
        super().__init__(agent_cls, model, sys_msg, delegator_client)

    def build_system_message(self, personas: list[str]) -> None:
        prompt = DELEGATOR_PROMPT
        for i, persona in enumerate(personas):
            prompt += f"\nInformation on User {i + 1}: {persona}\n"