#!/usr/bin/env python3
"""
下载所有 Google Drive 数据集
注意：Google Drive 在中国大陆无法访问，需要使用代理或 VPN
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据保存目录
project_root = Path(__file__).parent.parent
DATA_DIR = project_root / "data"
DATA_DIR.mkdir(exist_ok=True)

def download_with_gdown(file_id, output_path, is_folder=False, max_retries=3):
    """使用 gdown 下载文件，支持重试和 SSL 错误处理"""
    try:
        import gdown
        import ssl
        import certifi
    except ImportError as e:
        logger.error(f"  ✗ 缺少依赖: {e}")
        logger.error(f"  请运行: pip install gdown certifi")
        return False

    # 尝试配置 SSL 上下文
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except:
        pass

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"  重试 {attempt + 1}/{max_retries}...")
                import time
                time.sleep(3)

            if is_folder:
                url = f"https://drive.google.com/drive/folders/{file_id}"
                logger.info(f"  下载文件夹: {url}")
                gdown.download_folder(url, output=str(output_path), quiet=False, verify=False)
            else:
                url = f"https://drive.google.com/uc?id={file_id}"
                logger.info(f"  下载文件: {url}")
                gdown.download(url, str(output_path), quiet=False, verify=False)

            logger.info(f"  ✓ 保存到: {output_path}")
            return True
        except Exception as e:
            logger.error(f"  ✗ 尝试 {attempt + 1} 失败: {e}")
            if attempt == max_retries - 1:
                logger.error(f"  提示: 请确保已安装 gdown (pip install gdown certifi)")
                logger.error(f"  提示: 如果在中国大陆，需要使用代理或 VPN")
                logger.error(f"  提示: 如果 SSL 错误持续，可以手动下载:")
                logger.error(f"       https://drive.google.com/file/d/{file_id}/view")
                return False
    return False

def download_alcuna():
    """下载 ALCUNA 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 ALCUNA 数据集...")
    logger.info("注意: 需要从 Google Drive 下载，请确保网络可访问")

    save_dir = DATA_DIR / "alcuna"
    save_dir.mkdir(exist_ok=True)

    file_id_and_file_names = [
        ("19xjgOuFZe7WdAglX71OgUJXJoqDnPUzp", "id2question.json"),
        ("1kolOjXhS5AWI20RnwpA--xZf2ghojCxB", "meta_data.jsonl"),
    ]

    success_count = 0
    for file_id, file_name in file_id_and_file_names:
        output_path = save_dir / file_name
        logger.info(f"  下载 {file_name}...")
        if download_with_gdown(file_id, output_path):
            success_count += 1

    logger.info(f"ALCUNA: {success_count}/{len(file_id_and_file_names)} 文件下载成功")
    return success_count == len(file_id_and_file_names)

def download_nq_musique():
    """下载 NQ/Musique 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 NQ/Musique 数据集...")
    logger.info("注意: 需要从 Google Drive 下载，请确保网络可访问")

    save_dir = DATA_DIR / "nq_musique"
    save_dir.mkdir(exist_ok=True)

    file_id = "1q-6FIEGufKVBE3s6OdFoLWL2iHQPJh8h"
    zip_path = save_dir / "raw_data.zip"

    logger.info("  下载 raw_data.zip...")
    if download_with_gdown(file_id, zip_path):
        # 解压文件
        import zipfile
        try:
            logger.info("  解压 ZIP 文件...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
            logger.info(f"  ✓ 解压完成: {save_dir}")
            return True
        except Exception as e:
            logger.error(f"  ✗ 解压失败: {e}")
            return False
    return False

def download_qaqa():
    """下载 QAQA 数据集"""
    logger.info("=" * 60)
    logger.info("开始下载 QAQA 数据集...")
    logger.info("注意: 需要从 Google Drive 下载，请确保网络可访问")

    save_dir = DATA_DIR / "qaqa"
    save_dir.mkdir(exist_ok=True)

    file_id = "12aLKsSKe85G0u5bBTq0X0aKICsdxpaFL"
    tar_path = save_dir / "qaqa.tar.gz"

    logger.info("  下载 qaqa.tar.gz...")
    if download_with_gdown(file_id, tar_path):
        # 解压文件
        import tarfile
        try:
            logger.info("  解压 tar.gz 文件...")
            with tarfile.open(tar_path, 'r:gz') as tar_ref:
                tar_ref.extractall(save_dir)
            logger.info(f"  ✓ 解压完成: {save_dir}")
            return True
        except Exception as e:
            logger.error(f"  ✗ 解压失败: {e}")
            return False
    return False

def main():
    logger.info("开始下载所有 Google Drive 数据集...")
    logger.info("=" * 60)
    logger.info("⚠️  重要提示:")
    logger.info("  1. Google Drive 在中国大陆无法访问")
    logger.info("  2. 请确保已安装 gdown: pip install gdown")
    logger.info("  3. 需要使用代理或 VPN")
    logger.info("=" * 60)
    logger.info(f"数据保存目录: {DATA_DIR}")

    results = {
        "ALCUNA": download_alcuna(),
        "NQ/Musique": download_nq_musique(),
        "QAQA": download_qaqa(),
    }

    logger.info("=" * 60)
    logger.info("下载完成！统计结果：")
    success = sum(results.values())
    total = len(results)
    logger.info(f"成功: {success}/{total}")

    for name, status in results.items():
        status_str = "✓" if status else "✗"
        logger.info(f"  {status_str} {name}")

    if success < total:
        logger.warning("\n部分数据集下载失败。")
        logger.warning("如果在中国大陆，请确保:")
        logger.warning("  1. 使用了有效的代理或 VPN")
        logger.warning("  2. 代理设置正确 (HTTP_PROXY, HTTPS_PROXY)")

    return success == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
