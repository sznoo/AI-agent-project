import time
from models.phi3 import Phi3Wrapper
import prompts
from api import ExampleResult, apiExecutor, strategyManager


def detect_and_remove_repetition(text: str):
    words = text.strip().split()
    for size in range(1, len(words) // 2 + 1):
        chunk = words[:size]
        if chunk * (len(words) // size) == words[: size * (len(words) // size)]:
            return " ".join(chunk)
    return text


class predictor:
    def __init__(self, models):
        self.models = models
        self.mapping = {
            "phi3": self._parse_phi3,
            "baseline": self._parse_phi3,  # baseline uses the same method
        }

        self.phi3: Phi3Wrapper = models.get("phi3")

    def parse(self, vqas, question, method):
        starttime = time.time()
        if method not in self.mapping:
            raise ValueError(
                f"Unknown method: {method}. Available methods: {list(self.mapping.keys())}"
            )
        summary = ""
        for key in vqas:
            frame_id, question = key
            content = vqas[key]
            summary_line = f"[frame {frame_id}] {question}: {content}"
            summary += summary_line + "\n"
        result = self.mapping.get(method)(summary, question)
        elapsed_time = time.time() - starttime
        return elapsed_time, result, method

    def _parse_phi3(self, summary, question):
        """
        Parse the prompt using the Phi3 model.
        This method should be implemented to extract video path and question.
        """
        print(f">>>summary: {summary}")
        print(f">>>question: {question}")

        # prompt = prompts.generate_prompt5(summary, question)
        prompt = prompts.generate_prompt5_from_examples(
            strategyManager.current_examples, summary, question
        )
        print(f">>>prompt: {prompt}")
        result = self.phi3.infer(prompt, max_new_tokens=5)
        answer = detect_and_remove_repetition(result)
        example = ExampleResult(
            video_summary=summary,
            question=question,
            answer=answer,
            question_type=apiExecutor.memory.question_type,
        )
        return answer, example
