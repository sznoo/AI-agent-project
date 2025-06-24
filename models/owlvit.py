import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# import re


class OwlViTWrapper:
    def __init__(self, model_name="google/owlvit-base-patch32"):
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)

    def detect_objects(self, image, text_labels):
        inputs = self.processor(text=text_labels, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([(image.height, image.width)])
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.1,
            text_labels=text_labels,
        )
        return results[0]  # Return the first result

    # def answer_question(self, image, question):
    #     """
    #     image: PIL.Image
    #     question: str
    #     return: str (short answer)
    #     """
    #     question = question.lower().strip()

    #     # 명사 추출 (단수/복수 처리 포함)
    #     match = re.search(
    #         r"(cat|dog|person|man|woman|car|bird|bottle|book|laptop|phone|chair|.*)",
    #         question,
    #     )
    #     object_name = match.group(1) if match else None
    #     if object_name is None:
    #         return "unsupported"

    #     label = f"a photo of a {object_name}"
    #     result = self.detect_objects(image, [[label]])

    #     num_boxes = len(result["boxes"])
    #     labels = result["text_labels"]

    #     # 질문 유형 분기
    #     if question.startswith("is there") or question.startswith("are there"):
    #         return "yes" if num_boxes > 0 else "no"
    #     elif question.startswith("how many"):
    #         return str(num_boxes)
    #     elif question.startswith("where"):
    #         if num_boxes == 0:
    #             return "not found"
    #         else:
    #             boxes = result["boxes"]
    #             return str([box.tolist() for box in boxes])
    #     else:
    #         return "unsupported"


# if __name__ == "__main__":
#     wrapper = OwlViTWrapper()
#     # # 모델 및 프로세서 로드
#     # processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
#     # model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

#     # 테스트할 이미지 불러오기
#     url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     image = Image.open(requests.get(url, stream=True).raw)

#     # 탐지할 객체 텍스트 쿼리 정의
#     text_labels = [["a photo of a cat", "a photo of a dog"]]
#     starttime = time.time()
#     # result = wrapper.detect_objects(image, text_labels)
#     # print(result)
#     print(wrapper.answer_question(image, "what is the cat doing?"))  # → "yes"

#     print(f"Time taken: {time.time() - starttime:.2f} seconds")
#     # 입력 데이터 전처리
