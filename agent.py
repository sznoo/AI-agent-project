# from models.pali3 import Pali3Wrapper
# from models.gemma3 import Gemma3Wrapper
# import prompts
# import numpy as np
# from memory import Memory
# from api import APICodeExecutor
from models.phi3 import Phi3Wrapper
from models.owlvit import OwlViTWrapper
from models.pali3 import Pali3Wrapper

from transformers.utils import logging

logging.set_verbosity_error()
import warnings

warnings.filterwarnings("ignore", message="skipping cudagraphs due to multiple devices")

from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HF_TOKEN")

from huggingface_hub import login


class Agent:
    def __init__(self, token=token):
        """
        구성 요소들을 모두 외부에서 주입하여 유연한 모듈화 가능하게 설계
        """
        self.token = token

        # self.prompt_parser = prompt_parser  # 문장 → video_path, question
        # self.event_parser = event_parser  # question → event queue
        # self.grounder = grounder  # event queue → object, action grounding
        # self.reasoner = reasoner  # grounded info → reasoning plan
        # self.predictor = predictor  # plan → final answer

        self.phi3 = Phi3Wrapper()
        self.owlvit = OwlViTWrapper()
        self.pali3 = Pali3Wrapper()

    def run_single_step(self, prompt: str) -> str:
        """
        단일 단계 실행: prompt → answer
        """
        return prompt
        # parsed = self._parse_prompt(prompt)
        # events = self._parse_events(parsed)
        # grounded = self._ground(events)
        # reasoning_plan = self._reason(grounded)
        # answer = self._predict(reasoning_plan)
        # return answer

    def run(self) -> str:
        """
        전체 파이프라인 실행: prompt → answer
        """
        login(self.token)
        while True:
            prompt = input("Enter your prompt (or 'exit' to quit): ")
            if prompt.lower() == "exit":
                print("Exiting the agent.")
                break
            answer = self.run_single_step(prompt)
            print(f"Answer: {answer}")

    # def _parse_prompt(self, prompt: str) -> dict:
    #     """
    #     prompt (str) → dict with 'video_path', 'question'
    #     """
    #     return self.prompt_parser(prompt)

    # def _parse_events(self, parsed_prompt: dict) -> dict:
    #     """
    #     parsed_prompt → dict with 'question', 'event_queue'
    #     """
    #     return self.event_parser(parsed_prompt)

    # def _ground(self, event_info: dict) -> dict:
    #     """
    #     event_info → dict with visual grounding (bbox, track ids, etc.)
    #     """
    #     return self.grounder(event_info)

    # def _reason(self, grounded_info: dict) -> dict:
    #     """
    #     grounded_info → dict with reasoning plan or structured steps
    #     """
    #     return self.reasoner(grounded_info)

    # def _predict(self, reasoning_info: dict) -> str:
    #     """
    #     reasoning_info → final answer (short string)
    #     """
    #     return self.predictor(reasoning_info)
