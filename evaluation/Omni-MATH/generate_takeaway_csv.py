#!/usr/bin/env python3
"""
将evaluation_pass_at_k.csv转换为takeaway版本
格式：每一列是一个模型，每一行是一个指标（pass@1, pass@2, pass@4, pass@8, length）
只保留overall的结果
"""

import csv
import argparse
from pathlib import Path


def convert_to_takeaway(input_csv, output_csv):
    """
    将原始CSV转换为takeaway格式
    
    Args:
        input_csv: 输入的CSV文件路径
        output_csv: 输出的CSV文件路径
    """
    # 读取原始CSV
    overall_data = {}
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 只处理overall的数据
            if row['difficulty_range'] == 'overall':
                method_name = row['method']
                overall_data[method_name] = {
                    'pass@1': row.get('pass@1', ''),
                    'pass@2': row.get('pass@2', ''),
                    'pass@4': row.get('pass@4', ''),
                    'pass@8': row.get('pass@8', ''),
                    'length': row.get('avg_length', '')
                }
    
    if not overall_data:
        print(f"警告：在 {input_csv} 中没有找到overall数据")
        return
    
    # 获取所有模型名称（列名）
    model_names = sorted(overall_data.keys())
    
    # 定义指标顺序
    metrics = ['pass@1', 'pass@2', 'pass@4', 'pass@8', 'length']
    
    # 构建新的数据行
    rows = []
    for metric in metrics:
        row = {'Metric': metric}
        for model in model_names:
            row[model] = overall_data[model].get(metric, '')
        rows.append(row)
    
    # 写入新的CSV文件
    fieldnames = ['Metric'] + model_names
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Takeaway CSV已生成: {output_csv}")
    print(f"包含 {len(model_names)} 个模型，{len(metrics)} 个指标")


def main():
    parser = argparse.ArgumentParser(
        description='将evaluation_pass_at_k.csv转换为takeaway版本'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='evaluation_pass_at_k.csv',
        help='输入的CSV文件路径（默认: evaluation_pass_at_k.csv）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_pass_at_k_takeaway.csv',
        help='输出的CSV文件路径（默认: evaluation_pass_at_k_takeaway.csv）'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input).exists():
        print(f"错误：输入文件不存在: {args.input}")
        return
    
    convert_to_takeaway(args.input, args.output)


if __name__ == "__main__":
    main()
