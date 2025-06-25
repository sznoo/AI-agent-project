import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torchvision.transforms.functional as TF

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

    def detect_images_from_frame(self, frame, text_labels):
        pil_image = TF.to_pil_image(frame)
        result = self.detect_objects(pil_image, text_labels)
        return result


if __name__ == "__main__":
    import time
    import requests
    from PIL import Image
    import cv2

    def read_video_frames(video_path, max_frames=None):
        frames_dict = {}
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # BGR -> RGB 변환, float32 정규화 후 텐서로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = (
                torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            )  # (3, H, W)
            frames_dict[frame_count] = frame_tensor

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break

        cap.release()

        return frames_dict

    wrapper = OwlViTWrapper()
    frames_dict = read_video_frames(
        "/home/intern/jinwoo/AI-agent-project/videos/ivqa_example2.webm",
    )  #
    # 탐지할 객체 텍스트 쿼리 정의
    image = frames_dict[0]  # 첫 번째 프레임을 사용
    text_labels = [["hand"]]
    starttime = time.time()
    result = wrapper.detect_images_from_frame(image, text_labels)
    print(result, result["scores"].shape)
    # print(wrapper.answer_question(image, "what is the cat doing?"))  # → "yes"

    print(f"Time taken: {time.time() - starttime:.2f} seconds")
# # 입력 데이터 전처리
