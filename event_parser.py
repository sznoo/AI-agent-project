import time
from models.phi3 import Phi3Wrapper
import prompts


class eventParser:
    def __init__(self, models):
        self.models = models
        self.mapping = {
            "phi3": self._parse_phi3,
            "baseline": self._parse_phi3,  # baseline uses the same method
        }

        self.phi3: Phi3Wrapper = models.get("phi3")

    def parse(self, question, method):
        starttime = time.time()
        if method not in self.mapping:
            raise ValueError(
                f"Unknown method: {method}. Available methods: {list(self.mapping.keys())}"
            )
        result = self.mapping.get(method)(question)
        elapsed_time = time.time() - starttime
        return elapsed_time, result, method

    def _parse_phi3(self, question):
        """
        Parse the prompt using the Phi3 model.
        This method should be implemented to extract video path and question.
        """
        prompt = prompts.generate_prompt1(question)
        result = self.phi3.infer(prompt)

        lines = result.split("\n")[:4]
        codes = [line.split(":")[0].strip() for line in lines if line.strip()]
        return codes
        # Example implementation, replace with actual model inference logic
