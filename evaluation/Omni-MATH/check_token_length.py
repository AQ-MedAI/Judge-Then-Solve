#!/usr/bin/env python3
"""
统计 Omni-MATH 数据集中 problem 字段的 token 数量
找出超过指定 token 限制的题目
"""
import json
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import os


def load_dataset(input_file):
    """加载 JSONL 数据集"""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def count_tokens(text, tokenizer):
    """统计文本的 token 数量"""
    # 使用 encode 方法获取 token IDs，然后计算长度
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def main():
    parser = argparse.ArgumentParser(description="统计数据集中题目的 token 数量")
    parser.add_argument("--input", type=str, required=True,
                        help="输入的 JSONL 文件路径")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径或名称（用于加载 tokenizer）")
    parser.add_argument("--max_tokens", type=int, default=262144,
                        help="最大 token 限制（默认 262144）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出统计结果到文件（可选）")
    
    args = parser.parse_args()
    
    print(f"正在加载 tokenizer from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"加载 tokenizer 失败: {e}")
        print("请确保模型路径正确，或者使用 HuggingFace 模型名称")
        return
    
    print(f"正在加载数据集 from {args.input}...")
    data = load_dataset(args.input)
    print(f"已加载 {len(data)} 条数据")
    
    print(f"正在统计 token 数量（限制: {args.max_tokens} tokens）...")
    
    exceeded_count = 0
    exceeded_items = []
    token_counts = []
    max_tokens = 0
    max_tokens_item_idx = 0
    
    for idx, item in enumerate(tqdm(data, desc="处理中")):
        problem = item.get('problem', '')
        if not problem:
            continue
        
        token_count = count_tokens(problem, tokenizer)
        token_counts.append(token_count)
        
        if token_count > max_tokens:
            max_tokens = token_count
            max_tokens_item_idx = idx
        
        if token_count > args.max_tokens:
            exceeded_count += 1
            exceeded_items.append({
                'index': idx,
                'token_count': token_count,
                'problem_preview': problem[:200] + '...' if len(problem) > 200 else problem,
                'domain': item.get('domain', ''),
                'difficulty': item.get('difficulty', ''),
                'source': item.get('source', '')
            })
    
    # 统计信息
    total_count = len(data)
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0
    
    print("\n" + "="*60)
    print("统计结果:")
    print("="*60)
    print(f"总题目数: {total_count}")
    print(f"超过 {args.max_tokens} tokens 的题目数: {exceeded_count}")
    print(f"超过比例: {exceeded_count/total_count*100:.2f}%")
    print(f"\nToken 统计:")
    print(f"  最小 tokens: {min_tokens}")
    print(f"  最大 tokens: {max_tokens} (题目索引: {max_tokens_item_idx})")
    print(f"  平均 tokens: {avg_tokens:.2f}")
    print("="*60)
    
    if exceeded_count > 0:
        print(f"\n超过限制的题目详情（前10条）:")
        for i, item in enumerate(exceeded_items[:10]):
            print(f"\n[{i+1}] 索引: {item['index']}, Token数: {item['token_count']}")
            print(f"    Domain: {item['domain']}")
            print(f"    Difficulty: {item['difficulty']}")
            print(f"    Source: {item['source']}")
            print(f"    题目预览: {item['problem_preview']}")
        
        if len(exceeded_items) > 10:
            print(f"\n... 还有 {len(exceeded_items) - 10} 条超过限制的题目")
    
    # 保存结果到文件
    if args.output:
        result = {
            'total_count': total_count,
            'exceeded_count': exceeded_count,
            'exceeded_ratio': exceeded_count/total_count*100,
            'max_tokens': max_tokens,
            'max_tokens_index': max_tokens_item_idx,
            'avg_tokens': avg_tokens,
            'min_tokens': min_tokens,
            'exceeded_items': exceeded_items
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
