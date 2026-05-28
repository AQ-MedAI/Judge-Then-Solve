#!/usr/bin/env python3
"""
下载剩余的数据集：KUQ, QASPER, SituatedQA
修复版本 v3 - 解决 Brotli 和 404 问题

使用方法:
    python download_remaining_datasets_v3.py
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
print("下载剩余数据集 (v3 - 修复版)")
print("=" * 60)

def download_kuq():
    """下载 KUQ 数据集 - 禁用 Brotli 压缩"""
    print("\n[1/3] 下载 KUQ 数据集...")
    print("数据源: HuggingFace")

    try:
        # 方法1: 使用 datasets 库直接加载
        print("尝试使用 datasets 库加载...")
        dataset = load_dataset("amayuelas/KUQ", split="train")

        # 创建 DatasetDict
        dataset_dict = DatasetDict({"train": dataset})

        # 保存为 Arrow 格式
        save_path = DATA_DIR / "kuq_dataset"
        dataset_dict.save_to_disk(str(save_path))

        print(f"✓ KUQ 数据集处理成功")
        print(f"  保存路径: {save_path}")
        print(f"  数据量: {len(dataset)} 条")
        return True

    except Exception as e1:
        print(f"datasets 库加载失败: {e1}")
        print("尝试手动下载...")

        try:
            # 方法2: 手动下载，禁用压缩
            url = "https://huggingface.co/datasets/amayuelas/KUQ/resolve/main/knowns_unknowns.jsonl"

            # 禁用 Brotli 压缩
            headers = {
                'Accept-Encoding': 'identity'  # 只接受未压缩的内容
            }

            print(f"正在下载: {url}")
            response = requests.get(url, headers=headers, stream=True)
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
                    if line.strip():
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

        except Exception as e2:
            print(f"✗ KUQ 数据集下载失败: {e2}")
            return False

def download_qasper():
    """下载 QASPER 数据集 - 使用 datasets 库"""
    print("\n[2/3] 下载 QASPER 数据集...")
    print("数据源: HuggingFace")

    try:
        # 使用 datasets 库直接加载
        print("使用 datasets 库加载...")
        dataset = load_dataset("allenai/qasper")

        # 保存为 Arrow 格式
        save_path = DATA_DIR / "qasper"
        dataset.save_to_disk(str(save_path))

        print(f"✓ QASPER 数据集处理成功")
        print(f"  保存路径: {save_path}")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} 条")
        return True

    except Exception as e:
        print(f"✗ QASPER 数据集下载失败: {e}")
        return False

def download_situated_qa():
    """下载 SituatedQA 数据集 - 使用 datasets 库"""
    print("\n[3/3] 下载 SituatedQA 数据集...")
    print("数据源: HuggingFace")

    try:
        # 使用 datasets 库直接加载 geo 配置
        print("使用 datasets 库加载 geo 配置...")
        dataset = load_dataset("siyue/SituatedQA", "geo")

        # 保存为 Arrow 格式
        save_path = DATA_DIR / "situated_qa_geo"
        dataset.save_to_disk(str(save_path))

        print(f"✓ SituatedQA 数据集处理成功")
        print(f"  保存路径: {save_path}")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} 条")
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
