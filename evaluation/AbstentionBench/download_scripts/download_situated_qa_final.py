#!/usr/bin/env python3
"""
下载 SituatedQA 数据集 - 最终版本
使用 trust_remote_code=True
"""

from pathlib import Path
from datasets import load_dataset

REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"

print("=" * 60)
print("下载 SituatedQA 数据集")
print("=" * 60)

try:
    print("\n使用 datasets 库加载 geo 配置 (trust_remote_code=True)...")
    dataset = load_dataset("siyue/SituatedQA", "geo", trust_remote_code=True)

    # 保存为 Arrow 格式
    save_path = DATA_DIR / "situated_qa_geo"
    dataset.save_to_disk(str(save_path))

    print(f"\n✓ SituatedQA 数据集处理成功")
    print(f"  保存路径: {save_path}")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} 条")

    print("\n🎉 所有数据集下载完成！")

except Exception as e:
    print(f"\n✗ SituatedQA 数据集下载失败: {e}")
    import traceback
    traceback.print_exc()
