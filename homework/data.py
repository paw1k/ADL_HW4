"""
data.py – utilities for loading QA‑pair datasets and benchmarking a VLM

Key changes
-----------
✓   Always coerce `data_dir` to `pathlib.Path`, so `.glob()` never fails.
✓   Accept both directory and flat‑file layouts:
        data/train/*_qa_pairs.json
        data/train_qa_pairs.json
✓   Clear error message if nothing is found.
✓   No other public API signatures changed – finetune.py, base_vlm.py, etc.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import tqdm


# ----------------------------------------------------------------------
#  Dataset
# ----------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"


class VQADataset:
    """
    Lightweight in‑memory loader for SuperTuxKart VQA JSONs.

    Each sample returned by ``__getitem__`` is a dict with keys
    ``image_path, question, answer`` – exactly what `finetune.py`
    expects.
    """

    def __init__(
        self,
        split: str,
        data_dir: Path | str | None = None,
        max_samples: int | None = None,
    ):
        # Path‑ify early so downstream code can safely call .glob()
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self.split = split

        # ------------------------------------------------------------------
        # Gather *_qa_pairs.json files (nested layout)
        # ------------------------------------------------------------------
        qa_files: List[Path] = list(self.data_dir.glob(f"{split}/*_qa_pairs.json"))

        # Fallback: flat layout  data/train_qa_pairs.json
        flat_file = self.data_dir / f"{split}_qa_pairs.json"
        if not qa_files and flat_file.exists():
            qa_files = [flat_file]

        if not qa_files:
            raise FileNotFoundError(
                f"No *_qa_pairs.json found for split '{split}' in {self.data_dir} "
                "(tried both nested and flat layouts)."
            )

        # ------------------------------------------------------------------
        # Load JSONs
        # ------------------------------------------------------------------
        self.qa_pairs: List[dict[str, Any]] = []
        for file in qa_files:
            with file.open() as f:
                self.qa_pairs.extend(json.load(f))

        if max_samples is not None:
            self.qa_pairs = self.qa_pairs[: max_samples]

        print(f"Loaded {len(self.qa_pairs):,} QA pairs for split '{split}'")

    # ---------------- Dataset protocol ----------------
    def __len__(self) -> int:
        return len(self.qa_pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.qa_pairs[idx]
        image_path = os.path.join(self.data_dir, item["image_file"])
        return {
            "image_path": image_path,
            "question": item["question"],
            "answer": item["answer"],
        }


# ----------------------------------------------------------------------
#  Benchmarking helpers
# ----------------------------------------------------------------------
@dataclass
class VQABenchmarkResult:
    """
    Structured result returned by `benchmark`.
    """

    @dataclass
    class Sample:
        image_path: str
        question: str
        model_answer: str
        correct_answer: str
        is_correct: bool

    accuracy: float
    samples: List[Sample]

    # ---------------- Construction helper ----------------
    @classmethod
    def from_answers(
        cls,
        answers: List[str],
        gt_dataset: List[dict[str, Any]],
        max_samples: int | None = None,
    ) -> "VQABenchmarkResult":
        n = max_samples or min(len(answers), len(gt_dataset))
        correct = 0
        samples: List[VQABenchmarkResult.Sample] = []

        for ans, gt in zip(answers[:n], gt_dataset[:n]):
            answer_len = len(gt["answer"].strip())
            is_correct = ans.strip().lower()[:answer_len] == gt["answer"].strip().lower()[:answer_len]
            samples.append(
                cls.Sample(
                    image_path=gt["image_path"],
                    question=gt["question"],
                    model_answer=ans,
                    correct_answer=gt["answer"],
                    is_correct=is_correct,
                )
            )
            correct += int(is_correct)

        acc = correct / n if n else 0.0
        return cls(accuracy=acc, samples=samples)


# ----------------------------------------------------------------------
#  Public benchmark function
# ----------------------------------------------------------------------
def benchmark(model, dataset: VQADataset, max_samples: int | None = None) -> VQABenchmarkResult:
    """
    Run `model.answer` on a random subset of *dataset* and compute accuracy.

    Parameters
    ----------
    model        : any object implementing `.answer(image_paths, questions)`
    dataset      : VQADataset
    max_samples  : int | None, optional
        If given, sample up to this many items (default: full dataset).

    Returns
    -------
    VQABenchmarkResult
    """
    if len(dataset) == 0:
        raise ValueError("Empty dataset – nothing to benchmark.")

    k = min(max_samples or len(dataset), len(dataset))
    indices = random.sample(range(len(dataset)), k)

    questions = [dataset[i]["question"] for i in indices]
    image_paths = [dataset[i]["image_path"] for i in indices]

    print(f"Benchmarking on {k} samples …")
    answers = model.answer(image_paths, questions)

    gt_subset = [dataset[i] for i in indices]
    return VQABenchmarkResult.from_answers(answers, gt_subset, k)


# ----------------------------------------------------------------------
#  Quick manual test (optional)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ds = VQADataset("train", max_samples=3)
    print(ds[0])
