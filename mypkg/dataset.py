import torch
import pandas as pd
from pydantic import BaseModel

MEMORY_SIZE = 16
HIDDEN_SIZE = 768
TRAJECTORY_SIZE = 3

torch.manual_seed(42)
INITIAL_MEMORY = torch.randn(MEMORY_SIZE, HIDDEN_SIZE) * 0.02


class Trajectory(BaseModel):
    questions: list[int]
    correct_answers: list[int]


def build_dataset(df: pd.DataFrame, size: int, trajectory_size: int = TRAJECTORY_SIZE) -> list[Trajectory]:
    dataset = []
    for _ in range(size):
        rows = df.sample(n=trajectory_size)
        dataset.append(Trajectory(
            questions=rows["question_id"].tolist(),
            correct_answers=rows["correct_option_id"].tolist(),
        ))
    return dataset
