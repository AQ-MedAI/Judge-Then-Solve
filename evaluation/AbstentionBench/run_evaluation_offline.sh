#!/bin/bash
# 设置环境变量禁用 HuggingFace 在线检查
# 运行评测命令

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "已设置离线模式环境变量："
echo "  HF_DATASETS_OFFLINE=1"
echo "  TRANSFORMERS_OFFLINE=1"
echo "  HF_HUB_OFFLINE=1"
echo ""
echo "运行评测命令..."
echo ""

python main.py -m \
  mode=local \
  model=custom_api_env \
  module.model_name="Qwen3-30B-A3B-Thinking-2507" \
  dataset=kuq,qasper,situated_qa \
  > dataset_debug11.log 2>&1

echo ""
echo "评测完成！日志已保存到 dataset_debug11.log"
