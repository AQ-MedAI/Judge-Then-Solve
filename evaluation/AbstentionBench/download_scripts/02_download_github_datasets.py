#!/usr/bin/env python3
"""
下载所有 GitHub 数据集
使用 requests 库从 GitHub raw content 下载
"""

import os
import sys
from pathlib import Path
import requests
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据保存目录
project_root = Path(__file__).parent.parent
DATA_DIR = project_root / "data"
DATA_DIR.mkdir(exist_ok=True)

def download_file(url, save_path, timeout=60, max_retries=3):
    """下载单个文件，支持重试"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"  重试 {attempt + 1}/{max_retries}: {url}")
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
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False
    return False

def download_bbq():
    """下载 BBQ 数据集 (11个文件)"""
    logger.info("=" * 60)
    logger.info("开始下载 BBQ 数据集...")

    base_url = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/"
    files = [
        "Age.jsonl",
        "Disability_status.jsonl",
        "Gender_identity.jsonl",
        "Nationality.jsonl",
        "Physical_appearance.jsonl",
        "Race_ethnicity.jsonl",
        "Race_x_SES.jsonl",
        "Race_x_gender.jsonl",
        "Religion.jsonl",
        "SES.jsonl",
        "Sexual_orientation.jsonl",
    ]

    save_dir = DATA_DIR / "bbq_raw"
    success_count = 0

    for filename in files:
        url = base_url + filename
        save_path = save_dir / filename
        if download_file(url, save_path):
            success_count += 1
        time.sleep(0.5)  # 避免请求过快

    logger.info(f"BBQ: {success_count}/{len(files)} 文件下载成功")
    return success_count == len(files)

def download_falseqa():
    """下载 FalseQA 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 FalseQA 数据集...")

    url = "https://raw.githubusercontent.com/thunlp/FalseQA/main/dataset/test.csv"
    save_path = DATA_DIR / "falseqa_test.csv"

    return download_file(url, save_path)

def download_mediq():
    """下载 MediQ 数据集 (2个文件)"""
    logger.info("=" * 60)
    logger.info("开始下载 MediQ 数据集...")

    base_url = "https://raw.githubusercontent.com/stellalisy/mediQ/main/data/"
    files = {
        "all_craft_md.jsonl": "iCRAFT-MD",
        "all_dev_good.jsonl": "iMEDQA",
    }

    save_dir = DATA_DIR / "mediq_raw"
    success_count = 0

    for filename, desc in files.items():
        url = base_url + filename
        save_path = save_dir / filename
        logger.info(f"  下载 {desc}...")
        if download_file(url, save_path):
            success_count += 1
        time.sleep(0.5)

    logger.info(f"MediQ: {success_count}/{len(files)} 文件下载成功")
    return success_count == len(files)

def download_selfaware():
    """下载 SelfAware 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 SelfAware 数据集...")

    url = "https://raw.githubusercontent.com/yinzhangyue/SelfAware/main/data/SelfAware.json"
    save_path = DATA_DIR / "selfaware_raw.json"

    return download_file(url, save_path)

def download_umwp():
    """下载 UMWP 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 UMWP 数据集...")

    # 修正仓库名称：solving-biases 而不是 solving-biased-math
    urls = [
        "https://raw.githubusercontent.com/eth-lre/solving-biases/main/data/umwp_test.jsonl",
        "https://raw.githubusercontent.com/eth-lre/solving-biases/main/umwp_test.jsonl",
        "https://raw.githubusercontent.com/eth-lre/solving-biases/master/data/umwp_test.jsonl",
    ]

    save_path = DATA_DIR / "umwp_test.jsonl"

    for url in urls:
        logger.info(f"  尝试 URL: {url}")
        if download_file(url, save_path):
            return True

    logger.error("  所有 URL 都失败了")
    return False

def download_bigbench_disambiguate():
    """下载 BigBench Disambiguate 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 BigBench Disambiguate 数据集...")

    url = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/disambiguation_qa.json"
    save_path = DATA_DIR / "big_bench_disambiguate.json"

    return download_file(url, save_path)

def download_worldsense():
    """下载 WorldSense 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 WorldSense 数据集...")

    url = "https://github.com/facebookresearch/worldsense/raw/main/data/worldsense/training_set/trials_10k.jsonl.bz2"
    save_path = DATA_DIR / "world_sense_raw.jsonl.bz2"

    if download_file(url, save_path, timeout=120):
        # 解压 bz2 文件
        import bz2
        try:
            logger.info("  解压 bz2 文件...")
            decompressed_path = DATA_DIR / "world_sense_raw.jsonl"
            with bz2.open(save_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            logger.info(f"  ✓ 解压完成: {decompressed_path}")
            return True
        except Exception as e:
            logger.error(f"  ✗ 解压失败: {e}")
            return False
    return False

def download_squad2():
    """下载 Squad2 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 Squad2 数据集...")

    # 使用镜像站
    url = "https://hf-mirror.com/datasets/rajpurkar/squad_v2/resolve/main/squad_v2/validation-00000-of-00001.parquet"
    save_path = DATA_DIR / "squad2_validation.parquet"

    return download_file(url, save_path, timeout=120)

def main():
    logger.info("开始下载所有 GitHub 数据集...")
    logger.info(f"数据保存目录: {DATA_DIR}")

    results = {
        "BBQ": download_bbq(),
        "FalseQA": download_falseqa(),
        "MediQ": download_mediq(),
        "SelfAware": download_selfaware(),
        "UMWP": download_umwp(),
        "BigBench Disambiguate": download_bigbench_disambiguate(),
        "WorldSense": download_worldsense(),
        "Squad2": download_squad2(),
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
