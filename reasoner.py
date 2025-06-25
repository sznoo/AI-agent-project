import time
from models.phi3 import Phi3Wrapper
import prompts


class reasoner:
    def __init__(self, models):
        self.models = models
        self.mapping = {
            "phi3": self._parse_phi3,
            "baseline": self._parse_phi3,  # baseline uses the same method
        }

        self.phi3: Phi3Wrapper = models.get("phi3")

    def parse(self, qa_type, question, method):
        starttime = time.time()
        if method not in self.mapping:
            raise ValueError(
                f"Unknown method: {method}. Available methods: {list(self.mapping.keys())}"
            )
        result = self.mapping.get(method)(qa_type, question)
        elapsed_time = time.time() - starttime
        return elapsed_time, result, method

    def _parse_phi3(self, qa_type, question):
        """
        Parse the prompt using the Phi3 model.
        This method should be implemented to extract video path and question.
        """

        prompt = prompts.generate_prompt3(qa_type, question)
        result = self.phi3.infer(prompt)
        lines = result.split("\n")
        for idx in range(len(lines)):
            if lines[idx].strip() == "":
                break
        lines = lines[:idx]
        lines = [line.split(":")[0].strip() for line in lines if line.strip()]

        return lines
