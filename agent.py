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
from grounder import grounder
from reasoner import reasoner
from predictor import predictor
from api import apiExecutor, strategyManager


class Agent:
    def __init__(
        self,
        token=token,
        sample_prompt=None,
    ):
        """
        구성 요소들을 모두 외부에서 주입하여 유연한 모듈화 가능하게 설계
        """
        self.token = token

        self.phi3 = Phi3Wrapper()
        self.owlvit = OwlViTWrapper()
        self.pali3 = Pali3Wrapper()

        self.strategy_manager = strategyManager

        models_promptparser = {
            "phi3": self.phi3,
        }
        models_eventparser = {
            "phi3": self.phi3,
        }
        models_grounder = {
            "phi3": self.phi3,
        }
        models_reasoner = {
            "phi3": self.phi3,
        }
        models_predictor = {
            "phi3": self.phi3,
        }
        self.prompt_parser = promptParser(models_promptparser)
        self.event_parser = eventParser(models_eventparser)
        self.grounder = grounder(models_grounder)
        self.reasoner = reasoner(models_reasoner)
        self.predictor = predictor(models_predictor)

        self.api_executor = apiExecutor
        self.api_executor.init_memory()
        self.api_executor.set_models(
            models={
                "phi3": self.phi3,
                "owlvit": self.owlvit,
                "pali3": self.pali3,
            }
        )
        self.sample_prompt = sample_prompt

    def api_after_prompt_parsing(self, parsed_result) -> None:
        """
        API 호출 후 비디오 경로와 질문을 메모리에 저장
        """
        video_path = parsed_result.get("video_path", "")
        question = parsed_result.get("question", "")

        self.api_executor.read_video(video_path)
        self.api_executor.memory.video_path = video_path
        self.api_executor.memory.question = question
        return

    def api_after_event_parsing(self, event_parsed_codes) -> None:
        """
        이벤트 파싱 후 API 호출
        """
        for code in event_parsed_codes:
            self.api_executor.run_call(code)
        return

    def api_after_grounding(self, grounded_codes) -> None:
        """
        Grounding 후 API 호출
        """
        full_code = ""
        for code in grounded_codes:
            full_code += code + "\n"
        self.api_executor.run_call(full_code)
        return

    def api_after_reasoning(self, reason_codes) -> None:
        """
        Reasoning 후 API 호출
        """
        for code in reason_codes:
            self.api_executor.run_call(code)
        return

    def api_after_prediction(self, prediction_codes) -> None:
        """
        Prediction 후 API 호출
        """
        full_code = ""
        for code in prediction_codes:
            full_code += code + "\n"
        self.api_executor.run_call(full_code)
        return

    def run_single_step(self, user_input: str) -> str:
        """
        단일 단계 실행: prompt → answer
        """
        stratagy = self.strategy_manager.get_strategy(user_input)
        prompt_parsing_method = stratagy.get("prompt_parsing", "baseline")
        prompt_parsing_time, parsed_result, method = self.prompt_parser.parse(
            user_input, method=prompt_parsing_method
        )
        print(f">>>Parsed prompt ({method}): {parsed_result}")
        self.api_after_prompt_parsing(parsed_result)

        question = parsed_result.get("question", "")
        event_parsing_method = stratagy.get("event_parsing", "baseline")
        event_parsing_time, event_parsed_codes, method = self.event_parser.parse(
            question=question, method=event_parsing_method
        )
        # print(f">>>Parsed codes ({method})")
        # for code in event_parsed_codes:
        #     print(f"   {code}")

        self.api_after_event_parsing(event_parsed_codes)

        event_queue = self.api_executor.memory.event_queue
        conjunction = self.api_executor.memory.conjunction
        grounding_method = stratagy.get("grounding", "baseline")
        grounding_time, grounded_codes, method = self.grounder.parse(
            event_queue=event_queue, conjunction=conjunction, method=grounding_method
        )
        # print(f">>>Grounded info ({method}): {grounded_codes}")

        self.api_after_grounding(grounded_codes)

        qa_type = self.api_executor.memory.question_type
        reasoning_method = stratagy.get("reasoning", "baseline")
        reasoning_time, reasoning_code, method = self.reasoner.parse(
            qa_type=qa_type, question=question, method=reasoning_method
        )
        # print(f">>>Reasoning plan ({method}): {reasoning_code}")

        self.api_after_reasoning(reasoning_code)

        vqas = self.api_executor.memory.vqas
        prediction_method = stratagy.get("prediction", "baseline")
        prediction_time, result, method = self.predictor.parse(
            vqas=vqas, question=question, method=prediction_method
        )
        answer, example = result
        print(f">>>Predicted answer ({method}): {answer}")

        self.strategy_manager.update_result(example)
        time_per_step = {
            "total": prompt_parsing_time
            + event_parsing_time
            + grounding_time
            + reasoning_time
            + prediction_time,
            "prompt_parsing": prompt_parsing_time,
            "event_parsing": event_parsing_time,
            "grounding": grounding_time,
            "reasoning": reasoning_time,
            "prediction": prediction_time,
        }

        return answer, time_per_step

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

                print(
                    f"Prompt cannot be empty. trying sample prompt :{self.sample_prompt}"
                )
                prompt = self.sample_prompt

            answer, time_per_step = self.run_single_step(prompt)
            print(f"Answer: {answer}")

            response_accuracy = input("Was the answer correct (y/n): ")
            response_confidence = input(
                "How confident are you in the answer? (1 to 10): "
            )
            response_time = input("Was the answer fast enough? (y/n): ")
            print("--------------------------------------------------")
