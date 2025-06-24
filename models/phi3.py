import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# import sys
# import os

# # 현재 파일 기준으로 상위 폴더 경로 구하기
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(parent_dir)


class Phi3Wrapper:
    def __init__(self, model_id="microsoft/phi-3-mini-128k-instruct", device="cuda:5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def infer(self, prompt: str, max_new_tokens: int = 50) -> str:
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


# class Phi3PromptParser:
#     def __init__(self, model_id="microsoft/phi-3-mini-128k-instruct", device=None):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             torch_dtype=torch.bfloat16,
#         )
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#     def build_prompt(self, user_input: str) -> str:
#         return f"""Extract the video path and the question from the user's input.

# Input: 영상 'videos/cooking.mp4'에서 여자가 무엇을 하고 있나요?
# Output:
# Video: videos/cooking.mp4
# Question: 여자가 무엇을 하고 있나요?

# Input: {user_input}
# Output:"""

#     def parse(self, user_input: str) -> tuple[str, str]:
#         prompt = self.build_prompt(user_input)
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#         input_len = inputs.input_ids.shape[-1]

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=50,
#                 do_sample=False,
#                 pad_token_id=self.tokenizer.eos_token_id,
#             )
#         gen_text = self.tokenizer.decode(
#             outputs[0][input_len:], skip_special_tokens=True
#         )

#         # 간단한 파싱
#         lines = gen_text.strip().split("\n")
#         video_path = next(
#             (
#                 l.split(":", 1)[1].strip()
#                 for l in lines
#                 if l.lower().startswith("video")
#             ),
#             None,
#         )
#         question = next(
#             (
#                 l.split(":", 1)[1].strip()
#                 for l in lines
#                 if l.lower().startswith("question")
#             ),
#             None,
#         )

#         if not video_path or not question:
#             raise ValueError("Parsing failed. Output was:\n" + gen_text)
#         return video_path, question


# if __name__ == "__main__":
#     parser = Phi3PromptParser()
#     # phi3 = Phi3Wrapper()

#     # 예시 입력
#     user_input = (
#         "다음 영상에서 남자가 뭘 하는지 알려줘. 영상 경로는 'videos/sports.mp4'입니다."
#     )
#     starttime = time.time()
#     video, question = parser.parse(user_input)
#     print(f"Parsed in {time.time() - starttime:.2f} seconds")

#     # prompt = "In the video, what is the man doing after he opens the fridge?"
#     # print(f"Parsed in {time.time() - starttime:.2f} seconds")

#     # print(f"📼 Video Path: {video}")
#     # print(f"❓ Question: {question}")
