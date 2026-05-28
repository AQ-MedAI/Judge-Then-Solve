"""
为已有的结果文件生成评估指标的脚本

自动扫描results目录下所有子目录，评估所有结果并合并保存到CSV文件。

使用方法:
    python evaluate_existing_results.py
"""

import os
import glob
from pathlib import Path
from analysis.load_results import Results
from analysis.tables import AbstentionF1ScoreTable, CorrectnessTable, DetailedAbstentionTable

def check_use_prompt_method(base_results_dir: Path, result_path: str) -> bool:
    """
    检查结果路径是否使用了提示词方法
    
    Args:
        base_results_dir: results目录的路径
        result_path: 结果路径，格式如 "DatasetName_ModelName/YYYY-MM-DD_HH-MM-SS"
    
    Returns:
        bool: 是否使用了提示词方法
    """
    config_path = base_results_dir / result_path / "config.json"
    if config_path.exists():
        try:
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # 检查 config.json 中的 use_prompt_method 字段
            # 或者检查 run_name 是否包含 _ours_prompt
            if config_dict.get("use_prompt_method", False):
                return True
            run_name = config_dict.get("run_name", "")
            if "_ours_prompt" in run_name:
                return True
        except Exception:
            pass
    
    # 如果无法读取 config.json，检查路径名是否包含 _ours_prompt
    if "_ours_prompt" in result_path:
        return True
    
    return False


def find_all_result_paths(base_results_dir: Path, final_file: str = "GroundTruthAbstentionEvaluator.json"):
    """
    自动发现results目录下所有包含评估结果文件的路径
    只选择每个数据集的最新时间戳

    Args:
        base_results_dir: results目录的路径
        final_file: 要查找的结果文件名

    Returns:
        list[str]: 相对路径列表，格式如 "DatasetName_ModelName/YYYY-MM-DD_HH-MM-SS"
    """
    from collections import defaultdict

    # 使用glob查找所有匹配的文件
    pattern = str(base_results_dir / "**" / final_file)
    found_files = glob.glob(pattern, recursive=True)

    # 按数据集名称和是否使用提示词方法分组，保存所有时间戳
    # 键格式: (dataset_model, use_prompt_method)
    dataset_timestamps = defaultdict(list)

    for file_path in found_files:
        file_path_obj = Path(file_path)
        # 获取相对于base_results_dir的路径
        relative_path = file_path_obj.relative_to(base_results_dir).parent

        # 分离数据集名称和时间戳
        # 格式: DatasetName_ModelName/YYYY-MM-DD_HH-MM-SS
        parts = str(relative_path).split('/')
        if len(parts) == 2:
            dataset_model = parts[0]  # 例如: ALCUNADataset_CustomOpenAIModelEnv
            timestamp = parts[1]       # 例如: 2026-01-13_20-32-34
            use_prompt_method = check_use_prompt_method(base_results_dir, str(relative_path))
            # 使用 (dataset_model, use_prompt_method) 作为键
            key = (dataset_model, use_prompt_method)
            dataset_timestamps[key].append((timestamp, str(relative_path)))

    # 对每个数据集（区分是否使用提示词方法），只保留最新的时间戳
    result_paths = []
    for (dataset_model, use_prompt_method), timestamps in dataset_timestamps.items():
        # 按时间戳排序，取最新的
        timestamps.sort(reverse=True)  # 降序排序，最新的在前
        latest_path = timestamps[0][1]
        result_paths.append(latest_path)

    return sorted(result_paths)

def main():
    # 设置结果文件路径
    repo_root = Path(__file__).parent
    base_results_dir = repo_root / "results"
    final_file = "GroundTruthAbstentionEvaluator.json"
    
    # 自动发现所有结果路径
    print("="*60)
    print("正在扫描results目录...")
    print("="*60)
    result_paths = find_all_result_paths(base_results_dir, final_file)
    
    if not result_paths:
        print(f"❌ 未找到任何结果文件！")
        print(f"   请确保results目录下存在包含{final_file}的子目录")
        return
    
    print(f"✅ 找到 {len(result_paths)} 个结果目录:")
    for i, path in enumerate(result_paths, 1):
        print(f"   {i}. {path}")
    
    # 加载所有结果
    print("\n" + "="*60)
    print("正在加载所有结果...")
    print("="*60)
    
    try:
        results = Results(
            base_results_dir=str(base_results_dir),
            result_path_names=result_paths,  # 加载所有找到的结果
            final_file=final_file,
            format=True,  # 格式化模型和数据集名称
            allow_missing_correctness_files=True,  # 允许缺少正确性评估文件
        )
        
        print(f"✅ 成功加载 {len(results.df)} 条结果记录")
        print(f"   包含 {len(set(results.dataset_names))} 个数据集")
        print(f"   包含 {len(set(results.model_names))} 个模型")
        
        # 显示数据集和模型列表
        print(f"\n数据集列表: {', '.join(sorted(set(results.dataset_names)))}")
        print(f"模型列表: {', '.join(sorted(set(results.model_names)))}")
        
    except Exception as e:
        print(f"❌ 加载结果时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 生成Abstention F1分数表
    print("\n" + "="*60)
    print("生成Abstention F1分数表...")
    print("="*60)
    abstention_f1_table = AbstentionF1ScoreTable(results=results)
    print(f"✅ 生成了 {len(abstention_f1_table.table_df)} 行数据")
    print("\n前5行预览:")
    print(abstention_f1_table.table_df.head())
    
    # 生成正确性评估表
    print("\n" + "="*60)
    print("生成正确性评估表...")
    print("="*60)
    correctness_table = CorrectnessTable(results=results)
    print(f"✅ 生成了 {len(correctness_table.table_df)} 行数据")
    print("\n前5行预览:")
    print(correctness_table.table_df.head())
    
    # 生成详细弃权统计表
    print("\n" + "="*60)
    print("生成详细弃权统计表...")
    print("="*60)
    detailed_table = DetailedAbstentionTable(results=results)
    print(f"✅ 生成了 {len(detailed_table.table_df)} 行数据")
    print("\n前5行预览:")
    print(detailed_table.table_df.head())
    
    # 检查结果中是否使用了提示词方法
    prompt_method_paths = []
    normal_paths = []
    for result_path in result_paths:
        if check_use_prompt_method(base_results_dir, result_path):
            prompt_method_paths.append(result_path)
        else:
            normal_paths.append(result_path)
    
    # 保存结果到CSV文件
    print("\n" + "="*60)
    print("保存结果到CSV文件...")
    print("="*60)
    
    # 保存到analysis目录，与abstention_performance.csv格式一致
    output_dir = repo_root / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    # 根据是否使用了提示词方法，生成不同的文件名
    # 如果所有结果都使用了提示词方法，使用带后缀的文件名
    # 如果所有结果都没有使用提示词方法，使用默认文件名
    # 如果混合了，分别处理
    if prompt_method_paths and not normal_paths:
        # 所有结果都使用了提示词方法
        f1_csv_path = output_dir / "abstention_performance_ours_prompt.csv"
        correctness_csv_path = output_dir / "correctness_metrics_ours_prompt.csv"
        detailed_csv_path = output_dir / "detailed_abstention_stats_ours_prompt.csv"
        print(f"ℹ️  所有结果都使用了提示词方法，使用带 '_ours_prompt' 后缀的文件名")
    elif normal_paths and not prompt_method_paths:
        # 所有结果都没有使用提示词方法
        f1_csv_path = output_dir / "abstention_performance.csv"
        correctness_csv_path = output_dir / "correctness_metrics.csv"
        detailed_csv_path = output_dir / "detailed_abstention_stats.csv"
        print(f"ℹ️  所有结果都没有使用提示词方法，使用默认文件名")
    else:
        # 混合了使用和不使用提示词方法的结果
        # 使用带后缀的文件名，并给出警告
        f1_csv_path = output_dir / "abstention_performance_ours_prompt.csv"
        correctness_csv_path = output_dir / "correctness_metrics_ours_prompt.csv"
        detailed_csv_path = output_dir / "detailed_abstention_stats_ours_prompt.csv"
        print(f"⚠️  警告：结果中混合了使用和不使用提示词方法的结果")
        print(f"   使用提示词方法的结果: {len(prompt_method_paths)} 个")
        print(f"   未使用提示词方法的结果: {len(normal_paths)} 个")
        print(f"   使用带 '_ours_prompt' 后缀的文件名保存")
    
    # 保存F1分数表（与abstention_performance.csv格式一致）
    abstention_f1_table.table_df.to_csv(f1_csv_path, index=False)
    print(f"✅ Abstention F1分数表已保存到: {f1_csv_path}")
    print(f"   共 {len(abstention_f1_table.table_df)} 行数据")
    
    # 保存正确性评估表
    correctness_table.table_df.to_csv(correctness_csv_path, index=False)
    print(f"✅ 正确性评估表已保存到: {correctness_csv_path}")
    print(f"   共 {len(correctness_table.table_df)} 行数据")
    
    # 保存详细弃权统计表（CSV格式）
    detailed_table.table_df.to_csv(detailed_csv_path, index=False)
    print(f"✅ 详细弃权统计表（CSV）已保存到: {detailed_csv_path}")
    print(f"   共 {len(detailed_table.table_df)} 行数据")
    
    # 保存详细弃权统计表（Excel格式）
    detailed_excel_path = detailed_csv_path.with_suffix('.xlsx')
    try:
        detailed_table.table_df.to_excel(detailed_excel_path, index=False, engine='openpyxl')
        print(f"✅ 详细弃权统计表（Excel）已保存到: {detailed_excel_path}")
    except ImportError:
        print(f"⚠️  无法保存Excel文件：需要安装 openpyxl 库")
        print(f"   请运行: pip install openpyxl")
    except Exception as e:
        print(f"⚠️  保存Excel文件时出错: {e}")
    
    # 显示汇总统计
    print("\n" + "="*60)
    print("汇总统计")
    print("="*60)
    print(f"总样本数: {len(results.df)}")
    print(f"应该abstain的样本数: {results.df['prompt_should_abstain'].sum()}")
    print(f"实际abstain的样本数: {results.df['is_abstention'].sum()}")
    
    if results.df['is_abstention_correct'].notna().any():
        print(f"Abstention准确率: {results.df['is_abstention_correct'].mean():.4f}")
    else:
        print(f"Abstention准确率: 无数据")
    
    should_not_abstain_df = results.df[~results.df['prompt_should_abstain']]
    if should_not_abstain_df['is_response_correct'].notna().any():
        accuracy = should_not_abstain_df['is_response_correct'].mean()
        print(f"响应正确率 (在应该回答的问题上): {accuracy:.4f}")
    else:
        print(f"响应正确率 (在应该回答的问题上): NaN (未评估)")
    
    print("\n" + "="*60)
    print("✅ 评估完成！")
    print("="*60)
    print(f"所有结果已合并并保存到:")
    print(f"  - {f1_csv_path}")
    print(f"  - {correctness_csv_path}")
    print(f"  - {detailed_csv_path}")
    if detailed_excel_path.exists():
        print(f"  - {detailed_excel_path}")

if __name__ == "__main__":
    main()
