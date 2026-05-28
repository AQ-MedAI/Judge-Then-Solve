"""
生成汇总统计表的脚本

生成两种格式的表格：
1. 汇总表：每个指标一行，每个方法一列（跨所有数据集）
2. 详细表：按数据集分类的相同指标

每个方法 = 模型名 + 是否使用提示词
"""
import os
import json
from pathlib import Path
import pandas as pd
from analysis.load_results import Results
from evaluate_existing_results import find_all_result_paths, check_use_prompt_method


def get_method_name(model_name: str, use_prompt: bool) -> str:
    """生成方法名称"""
    if use_prompt:
        return f"{model_name}_ours_prompt"
    return model_name


def calculate_metrics_for_method(df: pd.DataFrame) -> dict:
    """
    计算一个方法的所有指标
    
    Args:
        df: 该方法的DataFrame（可能包含多个数据集）
    
    Returns:
        dict: 包含所有指标的字典
    """
    # 1. 应该弃权的题目数量
    num_should_abstain = (df["prompt_should_abstain"] == True).sum()
    
    # 2. Abstention Rate（应该弃权中实际弃权的比例）
    should_abstain_df = df[df["prompt_should_abstain"] == True]
    if len(should_abstain_df) > 0:
        valid_mask = should_abstain_df["is_abstention"].notna()
        if valid_mask.sum() > 0:
            abstain_count = (should_abstain_df.loc[valid_mask, "is_abstention"] == True).sum()
            abstention_rate = abstain_count / valid_mask.sum()
        else:
            abstention_rate = 0.0
    else:
        abstention_rate = 0.0
    
    # 3. 应该回答的题目数量
    num_should_not_abstain = (df["prompt_should_abstain"] == False).sum()
    
    # 4. Answer Rate（不应该弃权中实际回答的比例）
    should_not_abstain_df = df[df["prompt_should_abstain"] == False]
    if len(should_not_abstain_df) > 0:
        answer_count = (should_not_abstain_df["is_abstention"] == False).sum()
        answer_rate = answer_count / len(should_not_abstain_df)
    else:
        answer_rate = 0.0
    
    # 5. Correctness（不应该弃权的题目中答对的比例）
    if len(should_not_abstain_df) > 0:
        correct_count = (should_not_abstain_df["is_response_correct"] == True).sum()
        correctness = correct_count / len(should_not_abstain_df)
    else:
        correctness = 0.0
    
    return {
        "num_should_abstain": num_should_abstain,
        "abstention_rate": abstention_rate,
        "num_should_not_abstain": num_should_not_abstain,
        "answer_rate": answer_rate,
        "correctness": correctness,
    }


def add_method_name_to_df(results: Results, result_paths: list, base_results_dir: Path):
    """
    为DataFrame添加method_name列
    
    通过匹配result_paths和DataFrame中的model_name、dataset_name来确定方法
    """
    # 如果method_name列已经存在，直接返回
    if 'method_name' in results.df.columns:
        return
    
    # 创建(model_name, dataset_name)到method_name的映射
    # 通过遍历result_paths，为每个路径确定对应的model_name、dataset_name和是否使用提示词
    method_map = {}  # key: (model_name, dataset_name), value: method_name
    
    for result_path in result_paths:
        use_prompt = check_use_prompt_method(base_results_dir, result_path)
        path_parts = result_path.split('/')
        if len(path_parts) >= 1:
            dataset_model = path_parts[0]
            # 从DataFrame中找到匹配的行来确定model_name和dataset_name
            # 使用dataset_name来匹配，因为路径格式是DatasetName_ModelName
            matching_rows = results.df[
                results.df.apply(
                    lambda r: dataset_model.startswith(r['dataset_name']) or 
                             r['dataset_name'] in dataset_model,
                    axis=1
                )
            ]
            if len(matching_rows) > 0:
                model_name = matching_rows.iloc[0]['model_name']
                dataset_name = matching_rows.iloc[0]['dataset_name']
                method_name = get_method_name(model_name, use_prompt)
                method_map[(model_name, dataset_name)] = method_name
    
    # 为DataFrame的每一行添加method_name
    def get_method_for_row(row):
        key = (row['model_name'], row['dataset_name'])
        if key in method_map:
            return method_map[key]
        # 如果没找到，根据是否有_ours_prompt判断
        model_name = row['model_name']
        has_prompt = any(
            check_use_prompt_method(base_results_dir, p) and 
            (model_name in p or row['dataset_name'] in p)
            for p in result_paths
        )
        return get_method_name(model_name, has_prompt)
    
    results.df['method_name'] = results.df.apply(get_method_for_row, axis=1)


def create_summary_table(results: Results, result_paths: list, base_results_dir: Path) -> pd.DataFrame:
    """
    创建汇总表：每个指标一行，每个方法一列
    
    Returns:
        pd.DataFrame: 汇总表
    """
    # 为DataFrame添加method_name列
    add_method_name_to_df(results, result_paths, base_results_dir)
    
    # 按方法分组计算指标（跨所有数据集）
    method_metrics = {}
    for method_name in results.df['method_name'].unique():
        method_df = results.df[results.df['method_name'] == method_name]
        method_metrics[method_name] = calculate_metrics_for_method(method_df)
    
    # 创建汇总表：每个指标一行，每个方法一列
    summary_data = {
        '指标': [
            '应该弃权的题目数量',
            'Abstention Rate',
            '应该回答的题目数量',
            'Answer Rate',
            'Correctness'
        ]
    }
    
    # 按方法名排序，确保顺序一致
    sorted_methods = sorted(method_metrics.keys())
    for method_name in sorted_methods:
        metrics = method_metrics[method_name]
        summary_data[method_name] = [
            metrics['num_should_abstain'],
            metrics['abstention_rate'],
            metrics['num_should_not_abstain'],
            metrics['answer_rate'],
            metrics['correctness'],
        ]
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df


def create_detailed_table(results: Results, result_paths: list, base_results_dir: Path) -> pd.DataFrame:
    """
    创建详细表：按数据集分类的相同指标
    
    Returns:
        pd.DataFrame: 详细表
    """
    # 为DataFrame添加method_name列（复用上面的逻辑）
    add_method_name_to_df(results, result_paths, base_results_dir)
    
    # 按方法和数据集分组计算指标
    detailed_rows = []
    for (method_name, dataset_name), group_df in results.df.groupby(['method_name', 'dataset_name_formatted']):
        metrics = calculate_metrics_for_method(group_df)
        detailed_rows.append({
            '方法': method_name,
            '数据集': dataset_name,
            '应该弃权的题目数量': metrics['num_should_abstain'],
            'Abstention Rate': metrics['abstention_rate'],
            '应该回答的题目数量': metrics['num_should_not_abstain'],
            'Answer Rate': metrics['answer_rate'],
            'Correctness': metrics['correctness'],
        })
    
    detailed_df = pd.DataFrame(detailed_rows)
    # 按方法和数据集排序
    detailed_df = detailed_df.sort_values(['方法', '数据集'])
    return detailed_df


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
        return
    
    print(f"✅ 找到 {len(result_paths)} 个结果目录")
    
    # 加载所有结果
    print("\n" + "="*60)
    print("正在加载所有结果...")
    print("="*60)
    
    try:
        results = Results(
            base_results_dir=str(base_results_dir),
            result_path_names=result_paths,
            final_file=final_file,
            format=True,
            allow_missing_correctness_files=True,
        )
        
        print(f"✅ 成功加载 {len(results.df)} 条结果记录")
        
    except Exception as e:
        print(f"❌ 加载结果时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 生成汇总表
    print("\n" + "="*60)
    print("生成汇总表...")
    print("="*60)
    summary_df = create_summary_table(results, result_paths, base_results_dir)
    print(f"✅ 生成了汇总表，包含 {len(summary_df)} 行，{len(summary_df.columns)-1} 个方法")
    print("\n汇总表预览:")
    print(summary_df)
    
    # 生成详细表
    print("\n" + "="*60)
    print("生成详细表...")
    print("="*60)
    detailed_df = create_detailed_table(results, result_paths, base_results_dir)
    print(f"✅ 生成了详细表，包含 {len(detailed_df)} 行")
    print("\n详细表预览:")
    print(detailed_df.head(10))
    
    # 保存结果
    print("\n" + "="*60)
    print("保存结果...")
    print("="*60)
    
    output_dir = repo_root / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    # 保存汇总表
    summary_csv_path = output_dir / "summary_statistics.csv"
    summary_excel_path = output_dir / "summary_statistics.xlsx"
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 汇总表（CSV）已保存到: {summary_csv_path}")
    
    try:
        summary_df.to_excel(summary_excel_path, index=False, engine='openpyxl')
        print(f"✅ 汇总表（Excel）已保存到: {summary_excel_path}")
    except ImportError:
        print(f"⚠️  无法保存Excel文件：需要安装 openpyxl 库")
    except Exception as e:
        print(f"⚠️  保存Excel文件时出错: {e}")
    
    # 保存详细表
    detailed_csv_path = output_dir / "detailed_statistics.csv"
    detailed_excel_path = output_dir / "detailed_statistics.xlsx"
    detailed_df.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 详细表（CSV）已保存到: {detailed_csv_path}")
    
    try:
        detailed_df.to_excel(detailed_excel_path, index=False, engine='openpyxl')
        print(f"✅ 详细表（Excel）已保存到: {detailed_excel_path}")
    except ImportError:
        print(f"⚠️  无法保存Excel文件：需要安装 openpyxl 库")
    except Exception as e:
        print(f"⚠️  保存Excel文件时出错: {e}")
    
    print("\n" + "="*60)
    print("✅ 完成！")
    print("="*60)


if __name__ == "__main__":
    main()
