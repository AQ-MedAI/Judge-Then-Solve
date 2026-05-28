#!/usr/bin/env python3
"""
下载剩余的数据集：KUQ, QASPER, SituatedQA

使用方法:
    python download_remaining_datasets.py
"""

import os
from pathlib import Path
from datasets import load_dataset

# 仓库根目录
REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"

print("=" * 60)
print("下载剩余数据集")
print("=" * 60)

def download_kuq():
    """下载 KUQ 数据集"""
    print("\n[1/3] 下载 KUQ 数据集...")
    print("数据源: amayuelas/KUQ")

    try:
        # 下载数据集
        dataset = load_dataset("amayuelas/KUQ", data_files="knowns_unknowns.jsonl")

        # 保存到本地
        save_path = DATA_DIR / "kuq_dataset"
        dataset.save_to_disk(str(save_path))

        print(f"✓ KUQ 数据集下载成功")
        print(f"  保存路径: {save_path}")
        print(f"  数据量: {len(dataset['train'])} 条")
        return True

    except Exception as e:
        print(f"✗ KUQ 数据集下载失败: {e}")
        return False

def download_qasper():
    """下载 QASPER 数据集"""
    print("\n[2/3] 下载 QASPER 数据集...")
    print("数据源: allenai/qasper")

    try:
        # 下载数据集
        dataset = load_dataset("allenai/qasper")

        # 保存到本地
        save_path = DATA_DIR / "qasper"
        dataset.save_to_disk(str(save_path))

        print(f"✓ QASPER 数据集下载成功")
        print(f"  保存路径: {save_path}")
        print(f"  数据量: train={len(dataset['train'])}, validation={len(dataset['validation'])}, test={len(dataset['test'])}")
        return True

    except Exception as e:
        print(f"✗ QASPER 数据集下载失败: {e}")
        return False

def download_situated_qa():
    """下载 SituatedQA 数据集"""
    print("\n[3/3] 下载 SituatedQA 数据集...")
    print("数据源: siyue/SituatedQA")

    try:
        # 下载数据集
        dataset = load_dataset("siyue/SituatedQA", "geo", trust_remote_code=True)

        # 保存到本地
        save_path = DATA_DIR / "situated_qa_geo"
        dataset.save_to_disk(str(save_path))

        print(f"✓ SituatedQA 数据集下载成功")
        print(f"  保存路径: {save_path}")
        print(f"  数据量: test={len(dataset['test'])}")
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
    else:
        print("\n⚠️  部分数据集下载失败，请检查错误信息")
