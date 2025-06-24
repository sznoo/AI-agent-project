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

from prompt_parser import promptParser
from event_parser import eventParser


class StrategyManager:
    def __init__(self):
        return

    def get_strategy(self, user_input: str) -> dict:
        baseline = {
            "prompt_parsing": "baseline",
            "event_parsing": "baseline",
            "grounding": "baseline",
            "reasoning": "baseline",
            "prediction": "baseline",
        }
        return baseline


class Agent:
    def __init__(self, token=token):
        """
        구성 요소들을 모두 외부에서 주입하여 유연한 모듈화 가능하게 설계
        """
        self.token = token

        self.phi3 = Phi3Wrapper()
        self.owlvit = OwlViTWrapper()
        self.pali3 = Pali3Wrapper()

        self.strategy_manager = StrategyManager()

        models_promptparser = {
            "phi3": self.phi3,
        }
        models_eventparser = {
            "phi3": self.phi3,
        }
        self.prompt_parser = promptParser(models_promptparser)
        self.event_parser = eventParser(models_eventparser)

    def run_single_step(self, user_input: str) -> str:
        """
        단일 단계 실행: prompt → answer
        """
        stratagy = self.strategy_manager.get_strategy(user_input)
        prompt_parsing_method = stratagy.get("prompt_parsing", "baseline")
        prompt_parsing_time, parsed_prompt, method = self.prompt_parser.parse(
            user_input, method=prompt_parsing_method
        )
        print(f">>>Parsed prompt ({method}): {parsed_prompt}")
        print(f">>>Prompt parsing time ({method}): {prompt_parsing_time:.2f} seconds")

        video_path = parsed_prompt.get("video_path", "")
        question = parsed_prompt.get("question", "")

        event_parsing_method = stratagy.get("event_parsing", "baseline")
        event_parsing_time, event_parsed_codes, method = self.event_parser.parse(
            question=question, method=event_parsing_method
        )
        print(f">>>Parsed codes ({method})")
        for code in event_parsed_codes:
            print(f"   {code}")
        print(f">>>Event parsing time ({method}): {event_parsing_time:.2f} seconds")
        return user_input
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
            if prompt == "":
                sample_prompt = (
                    "In the video 'videos/cooking.mp4', what is the woman doing?"
                )
                print(f"Prompt cannot be empty. trying sample prompt :{sample_prompt}")
                prompt = sample_prompt

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
