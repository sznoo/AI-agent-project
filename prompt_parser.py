import time
from models.phi3 import Phi3Wrapper
import prompts


class promptParser:
    def __init__(self, models):
        self.models = models
        self.mapping = {
            "phi3": self._parse_phi3,
            "baseline": self._parse_phi3,  # baseline uses the same method
        }

        self.phi3: Phi3Wrapper = models.get("phi3")

    def parse(self, prompt, method):
        starttime = time.time()
        if method not in self.mapping:
            raise ValueError(
                f"Unknown method: {method}. Available methods: {list(self.mapping.keys())}"
            )
        result = self.mapping.get(method)(prompt)
        elapsed_time = time.time() - starttime
        return elapsed_time, result, method

    def _parse_phi3(self, user_input):
        """
        Parse the prompt using the Phi3 model.
        This method should be implemented to extract video path and question.
        """
        prompt = prompts.generate_user_prompt(user_input)
        result = self.phi3.infer(prompt)
        answer = result.split("\n")[:2]
        answer = [line.split(":")[1].strip() for line in answer if line.strip()]
        video_path = answer[0] if answer else ""
        question = answer[1] if len(answer) > 1 else ""

        return {
            "video_path": video_path,
            "question": question,
        }
        # Example implementation, replace with actual model inference logic
