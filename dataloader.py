import cv2
import torch
import os

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass


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


@dataclass
class IVQAConfig:
    data_root: str = "/hub_data2/intern/jinwoo/iVQA"
    video_dir: str = "videos"
    annotation_file: str = "ivqa.csv"
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "test.csv"

    def get_video_path(self, video_id, start, end):
        filename = f"{video_id}_{int(start)}_{int(end)}.webm"
        return os.path.join(self.data_root, self.video_dir, filename)


ivqa_config = IVQAConfig()


class iVQADataset(Dataset):
    def __init__(self, config: IVQAConfig = ivqa_config):
        """
        Args:
            csv_path (str): path to train.csv, val.csv, or test.csv
            video_dir (str): path to directory containing video files (optional)
            return_video (bool): whether to return video filepath (or only metadata)
        """
        csv_path = os.path.join(config.data_root, config.annotation_file)
        video_dir = os.path.join(config.data_root, config.video_dir)

        self.df = pd.read_csv(csv_path)
        self.video_dir = video_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Core components
        question = row["question"]
        answers = [row[f"answer{i}"] for i in range(1, 6)]
        confs = [row[f"conf{i}"] for i in range(1, 6)]

        # Video metadata
        video_id = row["video_id"]
        start = row["start"]
        end = row["end"]
        video_path = os.path.join(self.video_dir, f"{video_id}_{start}_{end}.webm")

        frames = read_video_frames(video_path)

        sample = {
            "question": question,
            "answers": answers,
            "confs": confs,
            "video_id": video_id,
            "start": start,
            "end": end,
            "video_path": video_path,
            "frames": frames,  # 이 부분 추가됨
        }

        return sample


class iVQADataloader:
    def __init__(self, config: IVQAConfig = ivqa_config, batch_size=1, shuffle=False):
        self.dataset = iVQADataset(config)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_loader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )


if __name__ == "__main__":
    # 설정
    config = IVQAConfig()
    dataloader_wrapper = iVQADataloader(config=config, batch_size=1, shuffle=True)
    loader = dataloader_wrapper.get_loader()

    # 하나의 배치만 확인
    for batch in loader:
        question = batch["question"]
        answers = batch["answers"]
        video_id = batch["video_id"]
        frames = batch["frames"]  # dict 형태

        print(f"🟡 Video ID: {video_id}")
        print(f"🔹 Question: {question}")
        print(f"🔸 Answers: {answers}")
        break

    # 프레임 시각화 (최대 3개)
