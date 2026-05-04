from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from mypkg.dataset import INITIAL_MEMORY, Trajectory
from mypkg.model import MemoryMPNetModel


@dataclass
class StepMetrics:
    step: int           # trajectory step at which this check was performed
    source_step: int    # step at which the Q+A pair was originally encoded
    similarity: float


@dataclass
class TrajectoryResult:
    per_step: list[StepMetrics] = field(default_factory=list)

    def mean_similarity(self) -> float:
        if not self.per_step:
            return 0.0
        return sum(m.similarity for m in self.per_step) / len(self.per_step)

    def similarity_by_lag(self) -> dict[int, float]:
        """Average similarity grouped by how many steps ago the pair was encoded (lag = step - source_step)."""
        from collections import defaultdict
        buckets: dict[int, list[float]] = defaultdict(list)
        for m in self.per_step:
            buckets[m.step - m.source_step].append(m.similarity)
        return {lag: sum(v) / len(v) for lag, v in sorted(buckets.items())}


def _cls(sequence_output: torch.Tensor) -> torch.Tensor:
    return sequence_output[:, 0, :]  # [B, H]


def _tokenize(text: str, tokenizer: PreTrainedTokenizer, device: torch.device) -> dict:
    return tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)


def evaluate_trajectory(
    trajectory: Trajectory,
    model: MemoryMPNetModel,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],   # full list indexed by question_id
    options: list[str],     # full list indexed by option_id
    device: torch.device,
) -> TrajectoryResult:
    """
    For each step t in the trajectory:
      1. Remembering run: encode Q_t + A_t, update memory, save target CLS.
      2. Query runs: re-evaluate all Q_0..Q_t with the current memory, compare to saved targets.
    Returns per-step similarity metrics.
    """
    result = TrajectoryResult()
    memory = INITIAL_MEMORY.to(device)

    target_cls_list: list[torch.Tensor] = []  # saved CLS [1, H] per encoded step

    model.eval()
    with torch.no_grad():
        for t, (q_id, a_id) in enumerate(zip(trajectory.questions, trajectory.correct_answers)):
            q_text = questions[q_id]
            a_text = options[a_id]

            # --- Remembering run: Q+A encodes info into memory ---
            qa_enc = _tokenize(f"{q_text} {a_text}", tokenizer, device)
            seq_out, memory = model(qa_enc["input_ids"], memory, qa_enc["attention_mask"])
            target_cls = _cls(seq_out).detach()  # [1, H]
            target_cls_list.append(target_cls)

            # --- Query runs: check all Q_0..Q_t with current memory ---
            for src_step, prev_q_id in enumerate(trajectory.questions[: t + 1]):
                q_enc = _tokenize(questions[prev_q_id], tokenizer, device)
                q_seq_out, _ = model(q_enc["input_ids"], memory, q_enc["attention_mask"])
                q_cls = _cls(q_seq_out)  # [1, H]

                sim = F.cosine_similarity(q_cls, target_cls_list[src_step], dim=-1).item()
                result.per_step.append(StepMetrics(step=t, source_step=src_step, similarity=sim))

    return result


def evaluate_dataset(
    dataset: list[Trajectory],
    model: MemoryMPNetModel,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    options: list[str],
    device: torch.device,
) -> dict:
    """
    Evaluate all trajectories and return aggregate metrics:
      - mean_similarity: average cosine similarity across all steps/trajectories
      - similarity_by_lag: average similarity broken down by how many steps have passed since encoding
    """
    all_metrics: list[StepMetrics] = []
    for traj in dataset:
        result = evaluate_trajectory(traj, model, tokenizer, questions, options, device)
        all_metrics.extend(result.per_step)

    if not all_metrics:
        return {}

    from collections import defaultdict
    lag_buckets: dict[int, list[float]] = defaultdict(list)
    for m in all_metrics:
        lag_buckets[m.step - m.source_step].append(m.similarity)

    return {
        "mean_similarity": sum(m.similarity for m in all_metrics) / len(all_metrics),
        "similarity_by_lag": {lag: sum(v) / len(v) for lag, v in sorted(lag_buckets.items())},
    }