#!/usr/bin/env python3
"""
主下载脚本 - 下载所有 AbstentionBench 数据集
使用 HuggingFace 镜像站: https://hf-mirror.com
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
project_root = Path(__file__).parent.parent
scripts_dir = Path(__file__).parent

def run_script(script_name, description):
    """运行下载脚本"""
    logger.info("=" * 80)
    logger.info(f"开始执行: {description}")
    logger.info("=" * 80)

    script_path = scripts_dir / script_name

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            logger.info(f"✓ {description} 完成")
            return True
        else:
            logger.error(f"✗ {description} 失败 (返回码: {result.returncode})")
            return False
    except Exception as e:
        logger.error(f"✗ {description} 执行出错: {e}")
        return False

def main():
    logger.info("=" * 80)
    logger.info("AbstentionBench 数据集下载工具")
    logger.info("=" * 80)
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"数据保存目录: {project_root / 'data'}")
    logger.info("")
    logger.info("将按以下顺序下载数据集:")
    logger.info("  1. HuggingFace 数据集 (7个) - 使用镜像站")
    logger.info("  2. GitHub 数据集 (8个)")
    logger.info("  3. Google Drive 数据集 (3个) - 需要代理/VPN")
    logger.info("")

    # 检查依赖
    logger.info("检查依赖...")
    try:
        import datasets
        import requests
        logger.info("✓ 必需依赖已安装")
    except ImportError as e:
        logger.error(f"✗ 缺少依赖: {e}")
        logger.error("请运行: pip install datasets requests")
        return False

    results = {}

    # 1. 下载 HuggingFace 数据集
    results["HuggingFace"] = run_script(
        "01_download_hf_datasets.py",
        "HuggingFace 数据集下载"
    )

    # 2. 下载 GitHub 数据集
    results["GitHub"] = run_script(
        "02_download_github_datasets.py",
        "GitHub 数据集下载"
    )

    # 3. 下载 Google Drive 数据集
    logger.info("")
    logger.info("⚠️  注意: Google Drive 数据集需要代理或 VPN")
    user_input = input("是否下载 Google Drive 数据集? (y/n): ").strip().lower()

    if user_input == 'y':
        results["Google Drive"] = run_script(
            "03_download_gdrive_datasets.py",
            "Google Drive 数据集下载"
        )
    else:
        logger.info("跳过 Google Drive 数据集下载")
        results["Google Drive"] = None

    # 打印最终统计
    logger.info("")
    logger.info("=" * 80)
    logger.info("下载完成！最终统计：")
    logger.info("=" * 80)

    for category, status in results.items():
        if status is None:
            status_str = "⊘ 跳过"
        elif status:
            status_str = "✓ 成功"
        else:
            status_str = "✗ 失败"
        logger.info(f"  {status_str} {category}")

    logger.info("")
    logger.info("下一步:")
    logger.info("  1. 检查 data/ 目录确认数据已下载")
    logger.info("  2. 使用 git/zeta 推送到服务器")
    logger.info("  3. 在服务器上运行实验")

    return all(v for v in results.values() if v is not None)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
