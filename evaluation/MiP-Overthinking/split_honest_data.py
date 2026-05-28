import json
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
NEW_HONEST_DIR = BASE_DIR.parent / "datasets" / "new_honest_training"
TEST_DIR = DATA_DIR / "test"

DATASETS = ["gsm8k", "math", "svamp", "formula"]


def expand_record(dataset: str, rec: dict):
    """
    将原始 MiP 记录展开成若干条训练样本（question-level）。

    返回一组 dict：
    {
        "dataset": <str>,
        "question": <str>,
        "answer": <any>,
        "label": "well_defined" | "insufficient",
    }
    """
    examples = []

    # gsm8k / math: 一条记录里有 question + insufficient_question
    if dataset in {"gsm8k", "math"}:
        q_full = rec.get("question")
        q_insuff = rec.get("insufficient_question")
        ans = rec.get("answer", None)

        if q_full is None or q_insuff is None:
            raise ValueError(
                f"{dataset} record 缺少 question 或 insufficient_question: {rec}"
            )

        examples.append(
            {
                "dataset": dataset,
                "question": q_full,
                "answer": ans,
                "label": "well_defined",
            }
        )
        examples.append(
            {
                "dataset": dataset,
                "question": q_insuff,
                "answer": ans,
                "label": "insufficient",
            }
        )

    # svamp / formula: 只有 insufficient_question
    elif dataset in {"svamp", "formula"}:
        q_insuff = rec.get("insufficient_question")
        ans = rec.get("answer", None)

        if q_insuff is None:
            raise ValueError(
                f"{dataset} record 缺少 insufficient_question: {rec}"
            )

        examples.append(
            {
                "dataset": dataset,
                "question": q_insuff,
                "answer": ans,
                "label": "insufficient",
            }
        )
    else:
        raise ValueError(f"未知数据集: {dataset}")

    return examples


def split_records(records, train_ratio=0.7, test_ratio=0.2, seed=42):
    """对一个数据集的原始 record 做 7:2:1 切分，返回 (train, val, test) 三个列表。"""
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)

    n = len(records)
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)
    n_val = n - n_train - n_test  # 把舍入误差全给 val

    train_idxs = idxs[:n_train]
    test_idxs = idxs[n_train : n_train + n_test]
    val_idxs = idxs[n_train + n_test :]

    train = [records[i] for i in train_idxs]
    val = [records[i] for i in val_idxs]
    test = [records[i] for i in test_idxs]

    return train, val, test


def main():
    NEW_HONEST_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    all_train_examples = []
    all_val_examples = []

    stats = {}

    for dataset in DATASETS:
        src_path = DATA_DIR / f"{dataset}.json"
        if not src_path.exists():
            raise FileNotFoundError(f"未找到源数据文件: {src_path}")

        with src_path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        train_recs, val_recs, test_recs = split_records(
            records, train_ratio=0.7, test_ratio=0.2, seed=42
        )

        # 展开 train / val
        for rec in train_recs:
            all_train_examples.extend(expand_record(dataset, rec))
        for rec in val_recs:
            all_val_examples.extend(expand_record(dataset, rec))

        # 写出 test 原始格式（保持 question / insufficient_question 成对存在）
        test_out_path = TEST_DIR / f"{dataset}.json"
        with test_out_path.open("w", encoding="utf-8") as f:
            json.dump(test_recs, f, ensure_ascii=False, indent=2)

        stats[dataset] = {
            "total_records": len(records),
            "train_records": len(train_recs),
            "val_records": len(val_recs),
            "test_records": len(test_recs),
        }

    # 写出新的 train_all.jsonl / eval_all.jsonl
    train_path = NEW_HONEST_DIR / "train_all.jsonl"
    eval_path = NEW_HONEST_DIR / "eval_all.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for ex in all_train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with eval_path.open("w", encoding="utf-8") as f:
        for ex in all_val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # 简单打印统计信息，方便检查
    summary = {
        "per_dataset_record_stats": stats,
        "total_train_examples": len(all_train_examples),
        "total_eval_examples": len(all_val_examples),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

