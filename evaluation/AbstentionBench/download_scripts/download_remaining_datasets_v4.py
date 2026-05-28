#!/usr/bin/env python3
"""
下载剩余的数据集：QASPER, SituatedQA
修复版本 v4 - 为所有数据集添加手动下载备用方案

使用方法:
    python download_remaining_datasets_v4.py
"""

import os
import json
import requests
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset

# 仓库根目录
REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"

print("=" * 60)
print("下载剩余数据集 (v4 - 完整修复版)")
print("=" * 60)

def download_qasper():
    """下载 QASPER 数据集 - 带手动下载备用方案"""
    print("\n[1/2] 下载 QASPER 数据集...")
    print("数据源: HuggingFace")

    try:
        # 方法1: 使用 datasets 库直接加载
        print("尝试使用 datasets 库加载...")
        dataset = load_dataset("allenai/qasper")

        # 保存为 Arrow 格式
        save_path = DATA_DIR / "qasper"
        dataset.save_to_disk(str(save_path))

        print(f"✓ QASPER 数据集处理成功")
        print(f"  保存路径: {save_path}")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} 条")
        return True

    except Exception as e1:
        print(f"datasets 库加载失败: {e1}")
        print("尝试手动下载 Parquet 文件...")

        try:
            # 方法2: 手动下载 Parquet 文件
            # 正确的 Parquet 文件路径
            splits_urls = {
                "train": "https://huggingface.co/datasets/allenai/qasper/resolve/main/qasper/train-00000-of-00001.parquet",
                "validation": "https://huggingface.co/datasets/allenai/qasper/resolve/main/qasper/validation-00000-of-00001.parquet",
                "test": "https://huggingface.co/datasets/allenai/qasper/resolve/main/qasper/test-00000-of-00001.parquet"
            }

            save_dir = DATA_DIR / "qasper_raw"
            save_dir.mkdir(parents=True, exist_ok=True)

            headers = {'Accept-Encoding': 'identity'}
            datasets = {}

            for split_name, url in splits_urls.items():
                print(f"正在下载 {split_name} split...")
                response = requests.get(url, headers=headers, stream=True)
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

        except Exception as e2:
            print(f"✗ QASPER 数据集下载失败: {e2}")
            return False

def download_situated_qa():
    """下载 SituatedQA 数据集 - 带手动下载备用方案"""
    print("\n[2/2] 下载 SituatedQA 数据集...")
    print("数据源: HuggingFace")

    try:
        # 方法1: 使用 datasets 库直接加载 geo 配置
        print("尝试使用 datasets 库加载 geo 配置...")
        dataset = load_dataset("siyue/SituatedQA", "geo")

        # 保存为 Arrow 格式
        save_path = DATA_DIR / "situated_qa_geo"
        dataset.save_to_disk(str(save_path))

        print(f"✓ SituatedQA 数据集处理成功")
        print(f"  保存路径: {save_path}")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} 条")
        return True

    except Exception as e1:
        print(f"datasets 库加载失败: {e1}")
        print("尝试手动下载 Parquet 文件...")

        try:
            # 方法2: 手动下载 Parquet 文件
            # 正确的 Parquet 文件路径
            url = "https://huggingface.co/datasets/siyue/SituatedQA/resolve/main/geo/test-00000-of-00001-c0e0e0e0e0e0e0e0.parquet"

            print(f"正在下载 test split...")
            headers = {'Accept-Encoding': 'identity'}
            response = requests.get(url, headers=headers, stream=True)
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

        except Exception as e2:
            print(f"✗ SituatedQA 数据集下载失败: {e2}")
            return False


if __name__ == "__main__":
    results = {}

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
