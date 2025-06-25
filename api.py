import dataloader
import torch
import random
from collections import Counter


class Memory:
    def __init__(self):
        return


class ExampleResult:

    def __init__(
        self,
        video_summary: str,
        question,
        answer,
        question_type,
        main_query="Provide an answer to the Question. Keep your answer short and concise; your answer",
    ):
        self.video_summary = video_summary
        self.question = question
        self.main_query = main_query
        self.answer = answer
        self.question_type = question_type

    def calculate_features(self, gt_answers):
        hit = [1 if ans[0].lower() in self.answer.lower() else 0 for ans in gt_answers]

        self.accuracy_level = hit / len(gt_answers)

        flat_answers = [a[0].lower() for a in gt_answers]
        count = Counter(flat_answers)
        majority = max(count.values())
        total = len(flat_answers)
        confidence = (majority - 1) / (
            total - 1
        )  # normalize: all same → 1, all different → 0
        self.confidence_level = confidence

    def features(self):
        return (
            f"ExampleResult(video_summary={self.video_summary}, "
            f"question={self.question}, main_query={self.main_query}, "
            f"answer={self.answer}, question_type={self.question_type})"
        )

    def __str__(self):
        return f"""Video summary:
        {self.video_summary}
        Question: {self.question}
        Main query: {self.main_query}
        Answer: {self.answer}
        """


class StrategyManager:
    def __init__(self):
        self.similar_nouns_candidates = dict()
        self.similar_questions_candidates = dict()
        self.stratagy = {
            "prompt_parsing": "baseline",
            "event_parsing": "baseline",
            "grounding": "baseline",
            "reasoning": "baseline",
            "prediction": "baseline",
        }

        self.current_examples = []
        self.new_examples = []
        return

    def get_strategy(self, user_input: str) -> dict:
        baseline = {
            "prompt_parsing": "baseline",
            "event_parsing": "baseline",
            "grounding": "baseline",
            "reasoning": "baseline",
            "prediction": "baseline",
        }
        return baseline

    def stratagy(self):
        return self.stratagy

    def grounding_method(self):
        return self.stratagy["grounding"]

    def update_result(self, example: ExampleResult):
        if len(self.current_examples) < 10:
            self.current_examples.append(example)
            return
        if len(self.new_examples) < 5:
            self.new_examples.append(example)

        if len(self.new_examples) >= 5:
            self.exchange_examples()

    def exchange_examples(self):
        self.new_examples = []


strategyManager = StrategyManager()


class APICodeExecutor:
    def __init__(self, memory: Memory):
        self.memory = memory

    def init_memory(self):
        self.memory = Memory()

    def set_models(self, models):
        self.models = models

    def read_video(self, video_path):
        self.frames = dataloader.read_video_frames(video_path)
        frame_ids = list(self.frames.keys())
        frame_num = len(frame_ids)
        frame_distance = frame_num // 16
        self.memory.filtered_frame_ids = frame_ids[
            ::frame_distance
        ]  # Take every 10th frame
        self.memory.filtered_frames = {
            fid: self.frames[fid] for fid in self.memory.filtered_frame_ids
        }
        self.memory.beginning_id = 0
        self.memory.middle_id = frame_distance * 5
        self.memory.end_id = frame_distance * 10

        self.memory.vqas = dict()

    def run_call(self, full_code: str):
        try:
            exec(full_code)
        except Exception as e:
            print(f"[Error] Failed to run: {full_code}")
            print(f"        Reason: {e}")


apiExecutor = APICodeExecutor(Memory())


def trim(section: str, truncated_question=None):
    assert section in ["beginning", "middle", "end", "none"]
    if section == "beginning":
        apiExecutor.memory.filtered_frames = {
            fid: frame
            for fid, frame in apiExecutor.memory.filtered_frames.items()
            if int(fid) < apiExecutor.memory.middle_id
        }
    elif section == "middle":
        apiExecutor.memory.filtered_frames = {
            fid: frame
            for fid, frame in apiExecutor.memory.filtered_frames.items()
            if int(fid) < apiExecutor.memory.end_id
            and int(fid) >= apiExecutor.memory.middle_id
        }
    elif section == "end":
        apiExecutor.memory.filtered_frames = {
            fid: frame
            for fid, frame in apiExecutor.memory.filtered_frames.items()
            if int(fid) >= apiExecutor.memory.end_id
        }
    else:
        pass

    return


def parse_event(conjunction: str, event=None, truncated_question=None):
    assert conjunction in ["before", "when", "after", "none"]
    apiExecutor.memory.conjunction = conjunction

    if conjunction == "none":
        apiExecutor.memory.event_queue = [apiExecutor.memory.question]
        return
    apiExecutor.memory.conjunction = conjunction
    apiExecutor.memory.event_queue = [event, truncated_question]
    return


def classify(question_type: str):
    assert question_type in [
        "what",
        "where",
        "counting",
        "why",
        "how",
        "location",
        "when",
        "who",
    ]
    apiExecutor.memory.question_type = question_type
    return


def require_ocr(require: bool):
    return


def localize(noun, noun_with_modifier=None):
    assert noun != "" or noun_with_modifier == ""
    if noun == "":
        return noun

    owlvit = apiExecutor.models.get("owlvit", None)
    labels = [noun]
    if owlvit is None:
        return labels[0]

    # if strategyManager.stratagy["grounding"] == "baseline":
    #     similar_nouns = strategyManager.similar_nouns_candidates.get(noun, None)
    #     if similar_nouns is None:
    #         phi3 = apiExecutor.models.get("phi3", None)
    #         if phi3 is not None:
    #             labels += phi3.generate_similar_noun(noun)
    #             strategyManager.update_similar_nouns_candidates(noun, labels)

    filtered_frame_ids = []
    for fid, frame in apiExecutor.memory.filtered_frames.items():
        result = owlvit.detect_images_from_frame(frame, [labels])
        if result["scores"].shape == torch.Size([0]):
            continue

        filtered_frame_ids.append(fid)

    apiExecutor.memory.filtered_frame_ids = filtered_frame_ids
    apiExecutor.memory.filtered_frames = {
        fid: apiExecutor.memory.filtered_frames[fid] for fid in filtered_frame_ids
    }

    print(f">>>{result}")

    return labels[0]


def verify_action(question, labels):
    return


def truncate(conjunction, action=""):
    return


def vqa(questions, require_ocr=False):
    if type(questions) is str:
        questions = [questions]
    assert type(questions) is list, "Questions must be a list of strings."
    assert len(questions) > 0, "Questions list cannot be empty."

    results = []
    frame_items = list(apiExecutor.memory.filtered_frames.items())
    sampled_frames = random.sample(frame_items, k=min(5, len(frame_items)))
    questions = ["Describe this image."] + questions  # Add a default question
    for question in questions:
        for fid, frame in sampled_frames:
            result = apiExecutor.models["pali3"].infer_from_frame(frame, question)

            apiExecutor.memory.vqas[(fid, question)] = (result,)

    return results
