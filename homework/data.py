import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent / "data"


class VQADataset:
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
        qa_files: list[Path] = list(self.data_dir.glob(f"{split}/*_qa_pairs.json"))

        # Fallback: flat layout  data/train_qa_pairs.json
        flat_file = self.data_dir / f"{split}_qa_pairs.json"
        if not qa_files and flat_file.exists():
            qa_files = [flat_file]

        if not qa_files:
            raise FileNotFoundError(
                f"No *_qa_pairs.json found for split '{split}' in {self.data_dir} "
                "(tried both nested and flat layouts)."
            )

        self.qa_pairs: list[dict[str, Any]] = []
        for file in qa_files:
            with file.open() as f:
                self.qa_pairs.extend(json.load(f))

        if max_samples is not None:
            self.qa_pairs = self.qa_pairs[: max_samples]

        print(f"Loaded {len(self.qa_pairs):,} QA pairs for split '{split}'")

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a QA pair by index.

        Args:
            idx: Index of the QA pair

        Returns:
            Dictionary containing the QA pair and image path
        """
        qa_pair = self.qa_pairs[idx]

        # Construct the full path to the image
        image_path = os.path.join(self.data_dir, qa_pair["image_file"])

        return {
            "image_path": image_path,
            "question": qa_pair["question"],
            "answer": qa_pair["answer"],
        }


@dataclass
class VQABenchmarkResult:
    @dataclass
    class Sample:
        image_path: str
        question: str
        model_answer: str
        correct_answer: str
        is_correct: bool

    accuracy: float
    samples: list[Sample]

    @classmethod
    def from_answers(
        cls, answers: list[str], gt_dataset: list[dict[str, Any]], max_samples: int = None
    ) -> "VQABenchmarkResult":
        """
        Create a benchmark result from model answers.

        Args:
            answers: List of model answers
            dataset: Dataset used for evaluation

        Returns:
            Benchmark result
        """
        samples = []
        correct_count = 0

        if max_samples is None:
            max_samples = min(len(answers), len(gt_dataset))
        else:
            max_samples = len(gt_dataset)

        for i in range(max_samples):
            item = gt_dataset[i]
            answer = answers[i]

            # For string answers, we use exact matching
            answer_len = len(item["answer"].strip())
            is_correct = answer.strip().lower()[:answer_len] == item["answer"].strip().lower()[:answer_len]
            samples.append(
                cls.Sample(
                    image_path=item["image_path"],
                    question=item["question"],
                    model_answer=answer,
                    correct_answer=item["answer"],
                    is_correct=is_correct,
                )
            )

            if is_correct:
                correct_count += 1

        print(correct_count)
        print(len(samples))

        return cls(accuracy=correct_count / len(samples) if samples else 0, samples=samples)


def benchmark(model, dataset: VQADataset, max_samples: int = None) -> VQABenchmarkResult:
    """
    Benchmark a VLM model on a dataset.

    Args:
        model: VLM model to evaluate
        dataset: Dataset to evaluate on
        max_samples: Maximum number of samples to evaluate

    Returns:
        Benchmark result
    """

    if len(dataset) == 0 or max_samples == 0:
        raise ValueError("Dataset or model is empty")

    # Limit the number of samples if specified
    if max_samples is not None:
        dataset_size = min(len(dataset), max_samples)
    else:
        dataset_size = len(dataset)

    import random

    sample_indices = random.sample(range(len(dataset)), dataset_size)

    # Extract questions and image paths
    questions = [dataset[i]["question"] for i in sample_indices]
    image_paths = [dataset[i]["image_path"] for i in sample_indices]
    answers = [dataset[i]["answer"] for i in sample_indices]
    # Get model answers
    responses = []
    gt_dataset = []
    mini_batch_size = 32
    import tqdm

    for i in tqdm.tqdm(range(0, dataset_size, mini_batch_size)):  # Process in batches
        batch_size = min(mini_batch_size, dataset_size - i)
        batch_questions = questions[i : i + batch_size]
        batch_image_paths = image_paths[i : i + batch_size]
        batch_indices = sample_indices[i : i + batch_size]

        batch_responses = model.answer(batch_image_paths, batch_questions)
        responses.extend(batch_responses)
        gt_dataset.extend([dataset[i] for i in batch_indices])
        print(f"\tProcessed {i + batch_size} samples")
        print(f"\tQuestions: {batch_questions}")
        print(f"\tResponses: {batch_responses}")
        print(f"\tAnswers: {answers[i : i + batch_size]}")

    return VQABenchmarkResult.from_answers(responses, gt_dataset, max_samples)


if __name__ == "__main__":
    # Test the dataset
    dataset = VQADataset("train")
    print(f"Dataset size: {len(dataset)}")

    # Print a sample
    sample = dataset[0]
    print("\nSample:")
    print(f"Image: {sample['image_path']}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
