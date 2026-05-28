#!/usr/bin/env python3
"""
下载所有 HuggingFace 数据集
使用镜像站: https://hf-mirror.com
"""

import os
import sys
from pathlib import Path

# 设置 HuggingFace 镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据保存目录
DATA_DIR = project_root / "data"
DATA_DIR.mkdir(exist_ok=True)

def download_coconot():
    """下载 CoCoNot 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 CoCoNot 数据集...")
    try:
        # Original split
        logger.info("下载 CoCoNot original split...")
        dataset = load_dataset("allenai/coconot", "original", split="test")
        save_path = DATA_DIR / "coconot_original"
        dataset.save_to_disk(str(save_path))
        logger.info(f"✓ CoCoNot original 保存到: {save_path}")

        # Contrast split
        logger.info("下载 CoCoNot contrast split...")
        dataset = load_dataset("allenai/coconot", "contrast", split="test")
        save_path = DATA_DIR / "coconot_contrast"
        dataset.save_to_disk(str(save_path))
        logger.info(f"✓ CoCoNot contrast 保存到: {save_path}")

        return True
    except Exception as e:
        logger.error(f"✗ CoCoNot 下载失败: {e}")
        return False

def download_gpqa():
    """下载 GPQA 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 GPQA 数据集...")
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        save_path = DATA_DIR / "gpqa_diamond"
        dataset.save_to_disk(str(save_path))
        logger.info(f"✓ GPQA 保存到: {save_path}")
        return True
    except Exception as e:
        logger.error(f"✗ GPQA 下载失败: {e}")
        return False

def download_kuq():
    """下载 KUQ 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 KUQ 数据集...")

    # 尝试多次下载，避免 brotli 解压错误
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"  尝试 {attempt + 1}/{max_retries}...")
            dataset = load_dataset("amayuelas/KUQ", split="train", download_mode="force_redownload" if attempt > 0 else None)
            save_path = DATA_DIR / "kuq_train"
            dataset.save_to_disk(str(save_path))
            logger.info(f"✓ KUQ 保存到: {save_path}")
            return True
        except Exception as e:
            logger.error(f"  ✗ 尝试 {attempt + 1} 失败: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
            else:
                logger.error("  尝试使用备用方法...")
                try:
                    # 备用方法：直接下载 parquet 文件
                    import requests
                    url = "https://hf-mirror.com/datasets/amayuelas/KUQ/resolve/main/data/train-00000-of-00001.parquet"
                    save_path = DATA_DIR / "kuq_train.parquet"
                    logger.info(f"  下载 parquet 文件: {url}")
                    response = requests.get(url, timeout=120)
                    response.raise_for_status()
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"✓ KUQ parquet 保存到: {save_path}")
                    return True
                except Exception as e2:
                    logger.error(f"✗ 备用方法也失败: {e2}")
                    return False
    return False

def download_mmlu():
    """下载 MMLU 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 MMLU 数据集...")

    # Math subsets
    math_subsets = ["college_mathematics", "abstract_algebra", "high_school_mathematics"]
    # History subsets
    history_subsets = ["global_facts", "high_school_world_history", "prehistory"]

    all_subsets = math_subsets + history_subsets
    success_count = 0

    for subset in all_subsets:
        try:
            logger.info(f"下载 MMLU {subset}...")
            dataset = load_dataset("cais/mmlu", subset, split="test")
            save_path = DATA_DIR / f"mmlu_{subset}"
            dataset.save_to_disk(str(save_path))
            logger.info(f"✓ MMLU {subset} 保存到: {save_path}")
            success_count += 1
        except Exception as e:
            logger.error(f"✗ MMLU {subset} 下载失败: {e}")

    return success_count == len(all_subsets)

def download_moralchoice():
    """下载 MoralChoice 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 MoralChoice 数据集...")
    try:
        # Question templates
        logger.info("下载 MoralChoice question_templates...")
        dataset = load_dataset("ninoscherrer/moralchoice", data_dir="question_templates")
        save_path = DATA_DIR / "moralchoice_templates"
        dataset.save_to_disk(str(save_path))
        logger.info(f"✓ MoralChoice templates 保存到: {save_path}")

        # Scenarios
        logger.info("下载 MoralChoice scenarios...")
        dataset = load_dataset("ninoscherrer/moralchoice", data_dir="scenarios")
        save_path = DATA_DIR / "moralchoice_scenarios"
        dataset.save_to_disk(str(save_path))
        logger.info(f"✓ MoralChoice scenarios 保存到: {save_path}")

        return True
    except Exception as e:
        logger.error(f"✗ MoralChoice 下载失败: {e}")
        return False

def download_qasper():
    """下载 QASPER 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 QASPER 数据集...")
    try:
        # 使用 data_files 参数直接加载 parquet 文件，避免使用已弃用的加载脚本
        dataset = load_dataset("allenai/qasper", split="test", download_mode="force_redownload")
        save_path = DATA_DIR / "qasper_test"
        dataset.save_to_disk(str(save_path))
        logger.info(f"✓ QASPER 保存到: {save_path}")
        return True
    except Exception as e:
        logger.error(f"✗ QASPER 下载失败: {e}")
        logger.error("  尝试使用备用方法...")
        try:
            # 备用方法：直接下载 parquet 文件
            import requests
            url = "https://hf-mirror.com/datasets/allenai/qasper/resolve/main/data/test-00000-of-00001.parquet"
            save_path = DATA_DIR / "qasper_test.parquet"
            logger.info(f"  下载 parquet 文件: {url}")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"✓ QASPER parquet 保存到: {save_path}")
            return True
        except Exception as e2:
            logger.error(f"✗ 备用方法也失败: {e2}")
            return False

def download_situated_qa():
    """下载 SituatedQA 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 SituatedQA 数据集...")
    try:
        # 不使用 trust_remote_code，直接加载数据
        dataset = load_dataset("siyue/SituatedQA", "geo", split="test", download_mode="force_redownload")
        save_path = DATA_DIR / "situated_qa_geo"
        dataset.save_to_disk(str(save_path))
        logger.info(f"✓ SituatedQA 保存到: {save_path}")
        return True
    except Exception as e:
        logger.error(f"✗ SituatedQA 下载失败: {e}")
        logger.error("  尝试使用备用方法...")
        try:
            # 备用方法：直接下载 parquet 文件
            import requests
            url = "https://hf-mirror.com/datasets/siyue/SituatedQA/resolve/main/geo/test-00000-of-00001.parquet"
            save_path = DATA_DIR / "situated_qa_geo.parquet"
            logger.info(f"  下载 parquet 文件: {url}")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"✓ SituatedQA parquet 保存到: {save_path}")
            return True
        except Exception as e2:
            logger.error(f"✗ 备用方法也失败: {e2}")
            return False

def main():
    logger.info("开始下载所有 HuggingFace 数据集...")
    logger.info(f"使用镜像站: {os.environ.get('HF_ENDPOINT')}")
    logger.info(f"数据保存目录: {DATA_DIR}")

    results = {
        "CoCoNot": download_coconot(),
        "GPQA": download_gpqa(),
        "KUQ": download_kuq(),
        "MMLU": download_mmlu(),
        "MoralChoice": download_moralchoice(),
        "QASPER": download_qasper(),
        "SituatedQA": download_situated_qa(),
    }

    logger.info("=" * 60)
    logger.info("下载完成！统计结果：")
    success = sum(results.values())
    total = len(results)
    logger.info(f"成功: {success}/{total}")

    for name, status in results.items():
        status_str = "✓" if status else "✗"
        logger.info(f"  {status_str} {name}")

    return success == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
