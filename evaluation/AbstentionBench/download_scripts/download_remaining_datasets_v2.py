#!/usr/bin/env python3
"""
下载剩余的数据集：KUQ, QASPER, SituatedQA
使用替代方法避免 HuggingFace 的限制

使用方法:
    python download_remaining_datasets_v2.py
"""

import os
import json
import requests
from pathlib import Path
from datasets import Dataset, DatasetDict

# 仓库根目录
REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"

print("=" * 60)
print("下载剩余数据集 (v2)")
print("=" * 60)

def download_kuq():
    """下载 KUQ 数据集 - 使用直接下载方法"""
    print("\n[1/3] 下载 KUQ 数据集...")
    print("数据源: GitHub Raw")

    try:
        # 直接从 GitHub 下载 JSONL 文件
        url = "https://huggingface.co/datasets/amayuelas/KUQ/resolve/main/knowns_unknowns.jsonl"

        print(f"正在下载: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # 保存 JSONL 文件
        save_dir = DATA_DIR / "kuq_raw"
        save_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = save_dir / "knowns_unknowns.jsonl"

        with open(jsonl_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✓ JSONL 文件下载成功: {jsonl_path}")

        # 读取 JSONL 并转换为 Dataset
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        # 创建 Dataset
        dataset = Dataset.from_list(data)
        dataset_dict = DatasetDict({"train": dataset})

        # 保存为 Arrow 格式
        save_path = DATA_DIR / "kuq_dataset"
        dataset_dict.save_to_disk(str(save_path))

        print(f"✓ KUQ 数据集处理成功")
        print(f"  保存路径: {save_path}")
        print(f"  数据量: {len(dataset)} 条")
        return True

    except Exception as e:
        print(f"✗ KUQ 数据集下载失败: {e}")
        return False

def download_qasper():
    """下载 QASPER 数据集 - 使用 Parquet 文件"""
    print("\n[2/3] 下载 QASPER 数据集...")
    print("数据源: HuggingFace Parquet files")

    try:
        # QASPER 数据集的 Parquet 文件 URL
        base_url = "https://huggingface.co/datasets/allenai/qasper/resolve/main/data"
        splits = {
            "train": f"{base_url}/train-00000-of-00001.parquet",
            "validation": f"{base_url}/validation-00000-of-00001.parquet",
            "test": f"{base_url}/test-00000-of-00001.parquet"
        }

        save_dir = DATA_DIR / "qasper_raw"
        save_dir.mkdir(parents=True, exist_ok=True)

        datasets = {}
        for split_name, url in splits.items():
            print(f"正在下载 {split_name} split...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            parquet_path = save_dir / f"{split_name}.parquet"
            with open(parquet_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # 读取 Parquet 文件
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            datasets[split_name] = Dataset.from_pandas(df)
            print(f"  ✓ {split_name}: {len(datasets[split_name])} 条")

        # 创建 DatasetDict
        dataset_dict = DatasetDict(datasets)

        # 保存为 Arrow 格式
        save_path = DATA_DIR / "qasper"
        dataset_dict.save_to_disk(str(save_path))

        print(f"✓ QASPER 数据集处理成功")
        print(f"  保存路径: {save_path}")
        return True

    except Exception as e:
        print(f"✗ QASPER 数据集下载失败: {e}")
        return False

def download_situated_qa():
    """下载 SituatedQA 数据集 - 使用 Parquet 文件"""
    print("\n[3/3] 下载 SituatedQA 数据集...")
    print("数据源: HuggingFace Parquet files")

    try:
        # SituatedQA geo 数据集的 Parquet 文件 URL
        url = "https://huggingface.co/datasets/siyue/SituatedQA/resolve/main/geo/test-00000-of-00001.parquet"

        print(f"正在下载 test split...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        save_dir = DATA_DIR / "situated_qa_raw"
        save_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = save_dir / "test.parquet"

        with open(parquet_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # 读取 Parquet 文件
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        test_dataset = Dataset.from_pandas(df)
        print(f"  ✓ test: {len(test_dataset)} 条")

        # 创建 DatasetDict
        dataset_dict = DatasetDict({"test": test_dataset})

        # 保存为 Arrow 格式
        save_path = DATA_DIR / "situated_qa_geo"
        dataset_dict.save_to_disk(str(save_path))

        print(f"✓ SituatedQA 数据集处理成功")
        print(f"  保存路径: {save_path}")
        return True

    except Exception as e:
        print(f"✗ SituatedQA 数据集下载失败: {e}")
        return False


if __name__ == "__main__":
    results = {}

    # 下载 KUQ
    results['KUQ'] = download_kuq()

    # 下载 QASPER
    results['QASPER'] = download_qasper()

    # 下载 SituatedQA
    results['SituatedQA'] = download_situated_qa()

    # 输出总结
    print("\n" + "=" * 60)
    print("下载完成！总结：")
    print("=" * 60)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for dataset, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{status}: {dataset}")

    print(f"\n总计: {success_count}/{total_count} 个数据集下载成功")

    if success_count == total_count:
        print("\n🎉 所有数据集下载成功！")
        print("\n下一步：运行修复脚本来修改数据集加载代码")
        print("  python fix_remaining_datasets.py")
    else:
        print("\n⚠️  部分数据集下载失败，请检查错误信息")
