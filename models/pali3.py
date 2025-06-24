import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image
import requests
import time


class Pali3Wrapper:
    def __init__(self, model_id="google/paligemma-3b-mix-224", device=None):
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )

        # 장치 설정 (cuda:0 or cpu)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def infer(self, image: Image.Image, prompt: str, max_new_tokens: int = 20) -> str:
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
            output = outputs[0][input_len:]
            answer = self.processor.decode(output, skip_special_tokens=True)

        return answer


# if __name__ == "__main__":
#     wrapper = Pali3Wrapper()

#     image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"
#     image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
#     prompt = "What is on the flower?"
#     starttime = time.time()
#     answer = wrapper.infer(image, prompt)
#     print(f"Answer: {answer}, Time taken: {time.time() - starttime:.2f} seconds")
