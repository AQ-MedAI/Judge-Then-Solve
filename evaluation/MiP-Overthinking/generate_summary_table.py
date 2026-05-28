#!/usr/bin/env python3
"""
自动总结评测结果脚本
生成包含所有模型和方法的评测结果表格（CSV + XLSX格式）

表格格式：
- 第1列：条件类型（条件缺失/Well defined）
- 第2列：数据量
- 第3列：指标名称（Abstention Rate/Length/Answer_rate/Correctness）
- 第4列及之后：各个模型的表现
"""

import os
import json
import csv
from collections import defaultdict
from pathlib import Path

try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    XLSX_AVAILABLE = True
except ImportError:
    XLSX_AVAILABLE = False
    print("Warning: openpyxl not installed. Only CSV will be generated.")
    print("Install with: pip install openpyxl")


def load_analysis_info(result_dir):
    """加载 analysis_info.json 文件"""
    analysis_path = os.path.join(result_dir, "analysis_info.json")
    if not os.path.exists(analysis_path):
        return None

    try:
        with open(analysis_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {analysis_path}: {e}")
        return None


def count_dataset_items(dataset_name):
    """统计数据集题目数量"""
    # 优先从 data/test 读取（测试集）
    data_path = f"data/test/{dataset_name}.json"
    if not os.path.exists(data_path):
        # 如果没有测试集，尝试从 data 读取（全量数据）
        data_path = f"data/{dataset_name}.json"

    if not os.path.exists(data_path):
        return 0

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return len(data)
            return 0
    except Exception as e:
        print(f"Warning: Failed to count items in {data_path}: {e}")
        return 0


def collect_all_results():
    """
    收集所有模型的评测结果

    返回格式：
    {
        'model_name': {
            'MiP': {
                'abstention_rate': float,
                'length': float,
                'n_items': int
            },
            'normal': {
                'answer_rate': float,
                'correctness': float,
                'length': float,
                'n_items': int
            }
        }
    }
    """
    results_dir = "results"
    datasets = ["gsm8k", "math", "svamp", "formula"]

    # 收集所有模型的结果
    model_results = defaultdict(lambda: {'MiP': {}, 'normal': {}})

    # 统计总数据量
    total_mip_items = 0
    total_normal_items = 0

    for dataset in datasets:
        # 统计数据量
        n_items = count_dataset_items(dataset)

        # MiP 版本（条件缺失）
        mip_dir = os.path.join(results_dir, dataset, "MiP")
        if os.path.exists(mip_dir):
            total_mip_items += n_items
            for model_name in os.listdir(mip_dir):
                model_path = os.path.join(mip_dir, model_name)
                if not os.path.isdir(model_path):
                    continue

                info = load_analysis_info(model_path)
                if info is None:
                    continue

                eval_results = info.get('evaluation_results', {})
                mean_length = info.get('mean_answer_length', 0)

                # 累加 MiP 结果
                if 'abstention_rate' not in model_results[model_name]['MiP']:
                    model_results[model_name]['MiP']['abstention_rate'] = 0
                    model_results[model_name]['MiP']['length'] = 0
                    model_results[model_name]['MiP']['n_items'] = 0

                abstention_rate = eval_results.get('insufficient_condition', 0)
                model_results[model_name]['MiP']['abstention_rate'] += abstention_rate * n_items
                model_results[model_name]['MiP']['length'] += mean_length * n_items
                model_results[model_name]['MiP']['n_items'] += n_items

        # Normal 版本（条件完整）- 只有 gsm8k 和 math 有
        if dataset in ["gsm8k", "math"]:
            normal_dir = os.path.join(results_dir, dataset, "normal")
            if os.path.exists(normal_dir):
                total_normal_items += n_items
                for model_name in os.listdir(normal_dir):
                    model_path = os.path.join(normal_dir, model_name)
                    if not os.path.isdir(model_path):
                        continue

                    info = load_analysis_info(model_path)
                    if info is None:
                        continue

                    eval_results = info.get('evaluation_results', {})
                    mean_length = info.get('mean_answer_length', 0)

                    # 累加 Normal 结果
                    if 'answer_rate' not in model_results[model_name]['normal']:
                        model_results[model_name]['normal']['answer_rate'] = 0
                        model_results[model_name]['normal']['correctness'] = 0
                        model_results[model_name]['normal']['length'] = 0
                        model_results[model_name]['normal']['n_items'] = 0

                    # Answer rate = 1 - abstention_rate
                    abstention_rate = eval_results.get('insufficient_condition', 0)
                    answer_rate = 1 - abstention_rate
                    correctness = eval_results.get('correct_answer', 0)

                    model_results[model_name]['normal']['answer_rate'] += answer_rate * n_items
                    model_results[model_name]['normal']['correctness'] += correctness * n_items
                    model_results[model_name]['normal']['length'] += mean_length * n_items
                    model_results[model_name]['normal']['n_items'] += n_items

    # 计算加权平均
    for model_name in model_results:
        # MiP 平均
        if model_results[model_name]['MiP'].get('n_items', 0) > 0:
            n = model_results[model_name]['MiP']['n_items']
            model_results[model_name]['MiP']['abstention_rate'] /= n
            model_results[model_name]['MiP']['length'] /= n

        # Normal 平均
        if model_results[model_name]['normal'].get('n_items', 0) > 0:
            n = model_results[model_name]['normal']['n_items']
            model_results[model_name]['normal']['answer_rate'] /= n
            model_results[model_name]['normal']['correctness'] /= n
            model_results[model_name]['normal']['length'] /= n

    return dict(model_results), total_mip_items, total_normal_items


def generate_csv(model_results, total_mip_items, total_normal_items, output_path):
    """生成 CSV 文件"""
    # 获取所有模型名称并排序
    model_names = sorted(model_results.keys())

    # 构建表格数据
    rows = []

    # 表头
    header = ["条件类型", "数据量", "指标"] + model_names
    rows.append(header)

    # 条件缺失部分
    rows.append([
        "条件缺失",
        str(total_mip_items),
        "Abstention Rate",
        *[f"{model_results[m]['MiP'].get('abstention_rate', 0)*100:.2f}%"
          if model_results[m]['MiP'].get('n_items', 0) > 0 else "N/A"
          for m in model_names]
    ])

    rows.append([
        "",
        "",
        "Length",
        *[f"{model_results[m]['MiP'].get('length', 0):.0f}"
          if model_results[m]['MiP'].get('n_items', 0) > 0 else "N/A"
          for m in model_names]
    ])

    # Well defined 部分
    rows.append([
        "Well defined",
        str(total_normal_items),
        "Answer_rate",
        *[f"{model_results[m]['normal'].get('answer_rate', 0)*100:.2f}%"
          if model_results[m]['normal'].get('n_items', 0) > 0 else "N/A"
          for m in model_names]
    ])

    rows.append([
        "",
        "",
        "Correctness",
        *[f"{model_results[m]['normal'].get('correctness', 0)*100:.2f}%"
          if model_results[m]['normal'].get('n_items', 0) > 0 else "N/A"
          for m in model_names]
    ])

    rows.append([
        "",
        "",
        "Length",
        *[f"{model_results[m]['normal'].get('length', 0):.0f}"
          if model_results[m]['normal'].get('n_items', 0) > 0 else "N/A"
          for m in model_names]
    ])

    # 写入 CSV
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"✓ CSV file saved to: {output_path}")
    return rows


def generate_xlsx(rows, output_path):
    """生成 XLSX 文件（带格式）"""
    if not XLSX_AVAILABLE:
        print("✗ XLSX generation skipped (openpyxl not installed)")
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Evaluation Summary"

    # 写入数据
    for row_idx, row_data in enumerate(rows, start=1):
        for col_idx, value in enumerate(row_data, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)

            # 表头格式
            if row_idx == 1:
                cell.font = Font(bold=True, size=11)
                cell.fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")
                cell.alignment = Alignment(horizontal='center', vertical='center')

            # 第一列（条件类型）格式
            elif col_idx == 1 and value:
                cell.font = Font(bold=True, size=11)
                cell.alignment = Alignment(horizontal='left', vertical='center')

            # 第三列（指标名称）格式
            elif col_idx == 3:
                cell.font = Font(bold=True, size=10)
                cell.alignment = Alignment(horizontal='left', vertical='center')

            # 数据单元格格式
            else:
                cell.alignment = Alignment(horizontal='center', vertical='center')

    # 调整列宽
    ws.column_dimensions['A'].width = 15
    ws.column_dimensions['B'].width = 10
    ws.column_dimensions['C'].width = 18
    for col_idx in range(4, len(rows[0]) + 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = 15

    # 合并单元格（条件类型和数据量）
    # 条件缺失：合并 A2:A3 和 B2:B3
    ws.merge_cells('A2:A3')
    ws.merge_cells('B2:B3')

    # Well defined：合并 A4:A6 和 B4:B6
    ws.merge_cells('A4:A6')
    ws.merge_cells('B4:B6')

    # 保存文件
    wb.save(output_path)
    print(f"✓ XLSX file saved to: {output_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("自动总结评测结果")
    print("=" * 80)
    print()

    # 自动检测项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir

    # 如果当前目录没有 results 文件夹，尝试查找
    if not os.path.exists(os.path.join(project_root, "results")):
        # 检查是否在 MiP-Overthinking 目录下
        if os.path.basename(project_root) == "MiP-Overthinking":
            pass  # 已经在正确目录
        else:
            # 尝试查找 MiP-Overthinking 目录
            possible_paths = [
                os.path.join(os.getcwd(), "MiP-Overthinking"),
                os.path.join(os.path.dirname(os.getcwd()), "MiP-Overthinking"),
            ]
            for path in possible_paths:
                if os.path.exists(os.path.join(path, "results")):
                    project_root = path
                    break

    # 切换到项目根目录
    original_dir = os.getcwd()
    os.chdir(project_root)
    print(f"工作目录: {project_root}")
    print()

    try:
        # 收集所有结果
        print("正在收集评测结果...")
        model_results, total_mip_items, total_normal_items = collect_all_results()

        if not model_results:
            print("✗ 未找到任何评测结果")
            return

        print(f"✓ 找到 {len(model_results)} 个模型的评测结果")
        print(f"  - 条件缺失题目总数: {total_mip_items}")
        print(f"  - 条件完整题目总数: {total_normal_items}")
        print()

        # 生成输出文件
        output_dir = "results/summary"
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, "evaluation_summary.csv")
        xlsx_path = os.path.join(output_dir, "evaluation_summary.xlsx")

        print("正在生成汇总表格...")
        rows = generate_csv(model_results, total_mip_items, total_normal_items, csv_path)
        generate_xlsx(rows, xlsx_path)

        print()
        print("=" * 80)
        print("汇总完成！")
        print("=" * 80)
        print()
        print("生成的文件：")
        print(f"  - CSV:  {os.path.abspath(csv_path)}")
        print(f"  - XLSX: {os.path.abspath(xlsx_path)}")
        print()

        # 打印预览
        print("表格预览：")
        print("-" * 80)
        for row in rows[:6]:  # 只显示前6行
            print("  ".join(f"{str(cell):>15}" for cell in row))
        print("-" * 80)

    finally:
        # 恢复原始工作目录
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
