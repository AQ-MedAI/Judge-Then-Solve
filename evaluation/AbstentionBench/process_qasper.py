#!/usr/bin/env python3
"""
处理 QASPER 数据集，转换为代码需要的格式
"""

import datasets
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"

print("=" * 60)
print("处理 QASPER 数据集")
print("=" * 60)

def extract_full_text(sample):
    lines = []

    for section, section_name in zip(
        sample["full_text"]["paragraphs"], sample["full_text"]["section_name"]
    ):
        if section_name:
            lines.append(section_name + "\n")

        for paragraph in section:
            if paragraph:
                lines.append(paragraph.strip() + "\n")

    full_text = "\n".join(lines)
    return full_text

def extract_reference_answers(answer_set):
    reference_answers = []

    for annotation in answer_set["answer"]:
        if annotation["free_form_answer"]:
            reference_answers.append(annotation["free_form_answer"])

        if annotation["yes_no"] is not None:
            reference_answers.append("Yes" if annotation["yes_no"] else "No")

        for extractive_span in annotation["extractive_spans"]:
            reference_answers.append(extractive_span)

    reference_answers = list(sorted(set([a.strip() for a in reference_answers])))
    return reference_answers

def extract_is_unanswerable(answer_set):
    is_unanswerable_annotations = []

    for annotation in answer_set["answer"]:
        is_unanswerable = annotation["unanswerable"]
        is_unanswerable_annotations.append(is_unanswerable)

    has_consensus = len(set(is_unanswerable_annotations)) == 1
    is_unanswerable_consensus = (
        is_unanswerable_annotations[0] if has_consensus else None
    )

    return is_unanswerable_consensus

def prepare_dataset(dataset):
    data = []

    for sample in dataset:
        id = sample["id"]
        title = sample["title"]
        full_text = extract_full_text(sample)

        # Each paper has multiple QA pairs associated with it
        questions = sample["qas"]["question"]
        answers = sample["qas"]["answers"]

        for question, answer_set in zip(questions, answers):
            # An answer_set is a set of annotations with possible answers
            reference_answers = extract_reference_answers(answer_set)
            is_unanswerable = extract_is_unanswerable(answer_set)

            data.append(
                (id, title, full_text, question, reference_answers, is_unanswerable)
            )

    data_df = pd.DataFrame(
        data,
        columns=[
            "id",
            "title",
            "full_text",
            "question",
            "reference_answers",
            "is_unanswerable",
        ],
    )

    new_dataset = datasets.Dataset.from_pandas(data_df)
    return new_dataset

try:
    # Load the raw QASPER dataset
    print("\n加载原始 QASPER 数据集...")
    raw_dataset = datasets.load_from_disk(str(DATA_DIR / "qasper"))

    print(f"原始数据集包含的 splits: {list(raw_dataset.keys())}")

    # Process the test split
    print("\n处理 test split...")
    test_dataset = raw_dataset["test"]
    print(f"原始 test split: {len(test_dataset)} 条")

    processed_dataset = prepare_dataset(test_dataset)
    print(f"处理后: {len(processed_dataset)} 条")

    # Save the processed dataset
    save_path = DATA_DIR / "qasper_processed"
    processed_dataset.save_to_disk(str(save_path))

    print(f"\n✓ QASPER 数据集处理成功")
    print(f"  保存路径: {save_path}")
    print(f"  数据量: {len(processed_dataset)} 条")

except Exception as e:
    print(f"\n✗ 处理失败: {e}")
    import traceback
    traceback.print_exc()
