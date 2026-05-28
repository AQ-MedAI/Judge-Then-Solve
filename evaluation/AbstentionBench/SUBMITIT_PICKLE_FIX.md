# submitit Pickle 序列化错误修复

## 🔴 问题

### 错误信息
```
ModuleNotFoundError: No module named 'recipe'
```

发生在 pickle 反序列化时：
```python
File "submitit/core/utils.py", line 232, in pickle_load
    return pickle.load(ifile)
ModuleNotFoundError: No module named 'recipe'
```

## 🔍 根本原因

### submitit 工作流程

1. **主进程**调用 `executor.submit(main, config)`
2. submitit 使用 **pickle 序列化**整个任务对象（包括 `main` 函数和 `config`）
3. 序列化的对象保存到文件
4. **新进程**启动，读取并**反序列化**任务对象
5. 执行任务

### 问题所在

- `main` 函数导入了 `recipe` 模块（第 27-29 行）
- pickle 序列化时需要能够导入所有依赖模块
- `clean_env()` 会**清除所有环境变量**，包括 PYTHONPATH
- 在 `clean_env()` 内部调用 `executor.submit()` 时，PYTHONPATH 已被清除
- pickle 无法找到 `recipe` 模块 → ModuleNotFoundError

### 之前的修复为什么不起作用

之前我在 `make_submitit_executor()` 中通过 `slurm_setup` 设置 PYTHONPATH：

```python
slurm_launcher_configs['slurm_setup'].append(f'export PYTHONPATH={new_pythonpath}')
```

**问题**：这只在**新进程启动时**设置 PYTHONPATH，但 pickle 序列化发生在**提交任务时**（主进程中），此时 PYTHONPATH 已被 `clean_env()` 清除。

## ✅ 解决方案

### 修改位置

`main.py` 第 159-180 行

### 修复逻辑

```python
def launch_evaluation_in_separate_job_with_submitit(self):
    second_stage_config, second_stage_logs_dir = self.create_second_stage_config()
    executor = self.make_submitit_executor(
        second_stage_logs_dir, second_stage_config.abstention_detector_launcher
    )

    # 在 clean_env() 之前保存 PYTHONPATH
    project_root = os.path.dirname(os.path.abspath(__file__))
    saved_pythonpath = os.environ.get('PYTHONPATH', '')

    with clean_env():
        # 在 clean_env() 内部恢复 PYTHONPATH，用于 pickle 序列化
        if saved_pythonpath:
            os.environ['PYTHONPATH'] = f"{project_root}:{saved_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = project_root

        job = executor.submit(main, second_stage_config)
        logger.info(f"Launched eval job! job_id: {job.job_id}")
        output = job.result()
    logger.info("Eval job (stage 2) complete.")
```

### 关键点

1. **在 `clean_env()` 之前**保存项目根目录路径
2. **在 `clean_env()` 内部**恢复 PYTHONPATH
3. 这样 `executor.submit()` 调用时，PYTHONPATH 已经设置好
4. pickle 序列化可以正确找到 `recipe` 模块

## 📋 完整修复总结

现在有**两层 PYTHONPATH 设置**：

### 1. 主进程中（用于 pickle 序列化）
- **位置**: `launch_evaluation_in_separate_job_with_submitit()` 方法
- **时机**: 在 `executor.submit()` 之前
- **目的**: 让 pickle 能够序列化包含 `recipe` 模块的对象

### 2. 新进程中（用于任务执行）
- **位置**: `make_submitit_executor()` 方法
- **时机**: 新进程启动时通过 `slurm_setup`
- **目的**: 让新进程能够导入 `recipe` 模块执行任务

## 🚀 现在可以运行了

```bash
python main.py -m mode=local model=custom_api_env \
  module.model_name="Qwen3-30B-A3B-Thinking-2507" \
  dataset='glob(*,exclude=[dummy,freshqa,gsm8k,kuq,qasper,situated_qa,umwp])'
```

这个命令现在应该可以成功运行所有 14 个数据集的完整评估流程。
