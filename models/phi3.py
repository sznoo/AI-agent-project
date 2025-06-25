import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
import os

# 현재 파일 기준으로 상위 폴더 경로 구하기
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import prompts
import time


class Phi3Wrapper:
    def __init__(self, model_id="microsoft/phi-3-mini-128k-instruct", device="cuda:3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def infer(self, prompt: str, max_new_tokens: int = 100) -> str:
        # Phi-3는 시스템 지시 프롬프트가 포함된 구조에 반응을 잘함
        system_prompt = (
            "<|system|>\nYou are a helpful assistant.\n<|user|>\n"
            + prompt
            + "\n<|assistant|>\n"
        )

        inputs = self.tokenizer(system_prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][input_len:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    def generate_similar_noun(self, noun: str) -> str:
        prompt = prompts.generate_prompt_similar_noun(noun)
        phi3_response = self.infer(prompt, max_new_tokens=50)
        nouns = phi3_response.split("\n")[0].split(",")
        nouns = [n.strip() for n in nouns if n.strip()]
        return nouns

    def generate_related_question(self, question: str) -> str:
        prompt = prompts.generate_prompt_related_questions(question)
        phi3_response = self.infer(prompt, max_new_tokens=150)
        related_questions = phi3_response.split("\n")[:3]
        related_questions = [
            q.split("-")[1].strip() for q in related_questions if q.strip()
        ]
        return related_questions


if __name__ == "__main__":
    # parser = Phi3PromptParser()
    phi3 = Phi3Wrapper()

    # 예시 입력

    starttime = time.time()
    # video, question = parser.parse(user_input)
    question = "What is the man doing with the food in the video?"
    print(phi3.generate_related_question(question))
    print(f"Parsed in {time.time() - starttime:.2f} seconds")

    # print(f"📼 Video Path: {video}")
    # print(f"❓ Question: {question}")
