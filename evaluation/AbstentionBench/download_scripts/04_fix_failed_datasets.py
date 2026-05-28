#!/usr/bin/env python3
"""
修复失败的数据集下载
专门用于下载之前失败的4个数据集: KUQ, QASPER, SituatedQA, UMWP
"""

import os
import sys
from pathlib import Path
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置 HuggingFace 镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 数据保存目录
project_root = Path(__file__).parent.parent
DATA_DIR = project_root / "data"
DATA_DIR.mkdir(exist_ok=True)

def download_file_with_retry(url, save_path, max_retries=3, timeout=120):
    """下载文件，支持重试"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"  重试 {attempt + 1}/{max_retries}: {url}")
                import time
                time.sleep(2)
            else:
                logger.info(f"  下载: {url}")

            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"  ✓ 保存到: {save_path}")
            return True
        except Exception as e:
            logger.error(f"  ✗ 尝试 {attempt + 1} 失败: {e}")
            if attempt == max_retries - 1:
                return False
    return False

def fix_kuq():
    """修复 KUQ 数据集下载"""
    logger.info("=" * 60)
    logger.info("修复 KUQ 数据集...")

    # 尝试直接下载 JSONL 文件
    urls = [
        "https://hf-mirror.com/datasets/amayuelas/KUQ/resolve/main/knowns_unknowns.jsonl",
        "https://huggingface.co/datasets/amayuelas/KUQ/resolve/main/knowns_unknowns.jsonl",
    ]

    save_dir = DATA_DIR / "kuq_train"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "knowns_unknowns.jsonl"

    for url in urls:
        logger.info(f"  尝试 URL: {url}")
        if download_file_with_retry(url, save_path):
            logger.info("✓ KUQ 下载成功")
            return True

    logger.error("✗ KUQ 下载失败")
    logger.error("  请手动访问: https://huggingface.co/datasets/amayuelas/KUQ")
    return False

def fix_qasper():
    """修复 QASPER 数据集下载"""
    logger.info("=" * 60)
    logger.info("修复 QASPER 数据集...")

    # 尝试多个可能的 parquet 文件路径
    urls = [
        "https://hf-mirror.com/datasets/allenai/qasper/resolve/main/test/data-00000-of-00001.parquet",
        "https://hf-mirror.com/datasets/allenai/qasper/resolve/main/qasper-test.parquet",
        "https://huggingface.co/datasets/allenai/qasper/resolve/main/test/data-00000-of-00001.parquet",
    ]

    save_dir = DATA_DIR / "qasper_test"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "data.parquet"

    for url in urls:
        logger.info(f"  尝试 URL: {url}")
        if download_file_with_retry(url, save_path):
            logger.info("✓ QASPER 下载成功")
            return True

    logger.error("✗ QASPER 下载失败")
    logger.error("  请手动访问: https://huggingface.co/datasets/allenai/qasper")
    return False

def fix_situated_qa():
    """修复 SituatedQA 数据集下载"""
    logger.info("=" * 60)
    logger.info("修复 SituatedQA 数据集...")

    # 尝试多个可能的 parquet 文件路径
    urls = [
        "https://hf-mirror.com/datasets/siyue/SituatedQA/resolve/main/geo/test/data-00000-of-00001.parquet",
        "https://hf-mirror.com/datasets/siyue/SituatedQA/resolve/main/data/geo-test.parquet",
        "https://huggingface.co/datasets/siyue/SituatedQA/resolve/main/geo/test/data-00000-of-00001.parquet",
    ]

    save_dir = DATA_DIR / "situated_qa_geo"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "data.parquet"

    for url in urls:
        logger.info(f"  尝试 URL: {url}")
        if download_file_with_retry(url, save_path):
            logger.info("✓ SituatedQA 下载成功")
            return True

    logger.error("✗ SituatedQA 下载失败")
    logger.error("  请手动访问: https://huggingface.co/datasets/siyue/SituatedQA")
    return False

def fix_umwp():
    """修复 UMWP 数据集下载"""
    logger.info("=" * 60)
    logger.info("修复 UMWP 数据集...")

    # 尝试多个可能的路径
    urls = [
        "https://raw.githubusercontent.com/eth-lre/unbiased-math-word-problems/main/data/umwp_test.jsonl",
        "https://raw.githubusercontent.com/eth-lre/unbiased-math-word-problems/master/data/umwp_test.jsonl",
        "https://raw.githubusercontent.com/eth-lre/solving-biases/main/datasets/umwp_test.jsonl",
    ]

    save_path = DATA_DIR / "umwp_test.jsonl"

    for url in urls:
        logger.info(f"  尝试 URL: {url}")
        if download_file_with_retry(url, save_path):
            logger.info("✓ UMWP 下载成功")
            return True

    logger.error("✗ UMWP 下载失败")
    logger.error("  请手动访问: https://github.com/eth-lre/solving-biases")
    return False

def main():
    logger.info("=" * 80)
    logger.info("修复失败的数据集下载")
    logger.info("=" * 80)
    logger.info(f"数据保存目录: {DATA_DIR}")
    logger.info("")

    results = {
        "KUQ": fix_kuq(),
        "QASPER": fix_qasper(),
        "SituatedQA": fix_situated_qa(),
        "UMWP": fix_umwp(),
    }

    logger.info("")
    logger.info("=" * 80)
    logger.info("修复完成！统计结果：")
    logger.info("=" * 80)
    success = sum(results.values())
    total = len(results)
    logger.info(f"成功: {success}/{total}")

    for name, status in results.items():
        status_str = "✓" if status else "✗"
        logger.info(f"  {status_str} {name}")

    if success < total:
        logger.warning("")
        logger.warning("部分数据集下载失败，请尝试手动下载：")
        logger.warning("1. 访问对应的 HuggingFace 或 GitHub 页面")
        logger.warning("2. 查看 Files 标签，找到数据文件")
        logger.warning("3. 手动下载并保存到对应的 data/ 目录")

    return success == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
