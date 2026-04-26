import json
import subprocess
import tempfile
import os
import re
import shutil
from typing import Dict, List, Tuple, Any, Optional
import logging
from openai import OpenAI

logging.getLogger("httpx").setLevel(logging.WARNING)
# --- 辅助函数修改 ---

def extract_and_save_json_to_path(text: str, target_path: str) -> bool:
    """
    修改版：将 JSON 保存到指定的全路径 (target_path)，而不是默认文件名
    """
    pattern = re.compile(
        r'## Instance Data \(JSON format\):\n\s*```json\n(.*?)```', 
        re.DOTALL
    )
    match = pattern.search(text)
    
    if match:
        json_content_str = match.group(1).strip()
        try:
            parsed_json = json.loads(json_content_str)
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            # 生产环境建议使用 logging 而不是 print，避免刷屏
            # print(f"JSON提取或保存失败: {e}")
            return False
    return False

def replace_config_paths(text):
    """
    将代码中所有对 config.json 的引用强制改为文件名
    因为我们会设置 cwd，所以直接用文件名即可
    """
    # 匹配各种写法的 config.json (如 /data/config.json, ./config.json)
    # 替换为纯文件名 "config.json"
    pattern = r'["\'].*?config\.json["\']'
    return re.sub(pattern, '"config.json"', text)

import re

def extract_python_code(text: str) -> str:
    """
    从混合文本中提取最后一个 Python 代码块
    只支持 ```python ... ``` 格式
    """
    # 正则匹配 ```python ... ```
    pattern = re.compile(r'```python\s*(.*?)```', re.DOTALL)
    
    matches = pattern.findall(text)
    
    if matches:
        # 只返回最后一个匹配到的 Python 代码块
        return matches[-1]
    
    # 启发式检查：如果整段文本可能就是代码
    if 'import ' in text or 'from ' in text:
        return text
    
    # 没有匹配到时返回原始内容
    return text

# --- ModelingChecklist 类修改 ---

class ModelingChecklist:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        # ... (self.checklist 和 self.evaluation_prompt 部分保持不变，省略以节省空间) ...
        self.checklist = {
            "feasibility": [
                {"id": "F1", "question": "模型是否可行？生成的程序是否包含编译错误？", "weight": 1.0, "check_type": "code_execution"},
                {"id": "F2", "question": "优化模型是否可解？是否存在不可行的约束？", "weight": 1.0, "check_type": "solver_check"},
                {"id": "F3", "question": "变量定义是否正确？是否有未定义的变量？", "weight": 0.8, "check_type": "llm_judge"}
            ],
            "correctness": [
                {"id": "C1", "question": "模型是否提供了正确的答案？结果是否最优？", "weight": 2.0, "check_type": "solver_result"},
                {"id": "C2", "question": "目标函数是否正确建模？", "weight": 1.5, "check_type": "llm_judge"},
                {"id": "C3", "question": "约束条件是否完整覆盖问题要求？", "weight": 1.5, "check_type": "llm_judge"}
            ],
            "robustness": [
                {"id": "R1", "question": "约束是否适当紧致？是否存在冗余约束？", "weight": 0.8, "check_type": "llm_judge"},
                {"id": "R2", "question": "变量是否正确反映了整数性约束？", "weight": 1.0, "check_type": "llm_judge"},
                {"id": "R3", "question": "模型是否能处理边界情况？", "weight": 0.6, "check_type": "llm_judge"}
            ]
        }
        self.evaluation_prompt = """
        你是一个优化建模专家。请根据给定的问题、建模结果、求解器日志和参考答案，回答以下问题。

        原始问题：
        {problem}

        建模结果：
        {modeling_result}

        求解器日志：
        {solver_log}

        参考答案（如果有）：
        {ground_truth}

        请回答以下问题（只回答YES或NO）：
        {question}

        请以JSON格式回答：
        {{"answer": "YES/NO", "explanation": "简短解释"}}
        """

    def evaluate_model(self, 
                       problem: str, 
                       modeling_res: str, 
                       ground_truth: str = "", 
                       solver_log: str = "") -> Dict[str, Any]:
        
        results = {
            "feasibility": {}, "correctness": {}, "robustness": {},
            "total_reward": 0.0, "detailed_scores": {}
        }
        
        # 1. 预处理代码路径
        modeling_result = extract_python_code(modeling_res)
        modeling_result = replace_config_paths(modeling_result)

        # 2. 运行求解器 (修改点：传入 problem 用于提取数据)
        # 注意：这里修正了原代码的逻辑错误，必须在运行前准备好数据
        if not solver_log:
            solver_log = self._run_solver(modeling_result, problem)
        
        # 3. 评估每个维度 (循环逻辑保持不变)
        for dimension, questions in self.checklist.items():
            dimension_score = 0.0
            total_weight = 0.0
            for question_info in questions:
                score = self._evaluate_single_question(
                    question_info, problem, modeling_result, ground_truth, solver_log
                )
                dimension_score += score * question_info["weight"]
                total_weight += question_info["weight"]
                results["detailed_scores"][question_info["id"]] = {
                    "score": score, "weight": question_info["weight"], "question": question_info["question"]
                }
            if total_weight > 0:
                results[dimension]["score"] = dimension_score / total_weight
            else:
                results[dimension]["score"] = 0.0
        
        # 计算总分 (保持不变)
        results["total_reward"] = (
            results["feasibility"]["score"] * 0.3 +
            results["correctness"]["score"] * 0.5 +
            results["robustness"]["score"] * 0.2
        )
        return results

    # ... _evaluate_single_question 等中间函数保持不变 ...
    def _evaluate_single_question(self, question_info, problem, modeling_result, ground_truth, solver_log):
        # 只需要将原来的函数复制过来即可，不需要修改逻辑
        if question_info["check_type"] == "code_execution":
            return self._check_code_execution(modeling_result)
        elif question_info["check_type"] == "solver_check":
            return self._check_solver_feasibility(solver_log)
        elif question_info["check_type"] == "solver_result":
            return self._check_solver_result(solver_log, ground_truth)
        elif question_info["check_type"] == "llm_judge":
            return self._llm_judge(question_info["question"], problem, modeling_result, ground_truth, solver_log)
        return 0.0

    def _check_code_execution(self, modeling_result: str) -> float:
        """
        修改版：使用临时文件检查代码语法，避免 temp.py 冲突
        """
        try:
            # delete=True 会在关闭后自动删除文件，但在 Windows 上可能导致无法再次打开读取
            # 所以使用 TemporaryDirectory 更安全
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, 'check_syntax.py')
                with open(temp_file_path, 'w', encoding='utf-8') as f:
                    f.write(modeling_result)
                
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    compile(f.read(), temp_file_path, 'exec')
                return 1.0
        except SyntaxError:
            return 0.0
        except Exception:
            return 0.0

    # ... _check_solver_feasibility, _check_solver_result, _extract_objective_value, _llm_judge 保持不变 ...
    def _check_solver_feasibility(self, solver_log):
        # (同原代码)
        if not solver_log: return 0.0
        feasible = ["optimal", "feasible", "solution found"]
        infeasible = ["infeasible", "unbounded", "no solution"]
        log = solver_log.lower()
        if any(k in log for k in infeasible): return 0.0
        if any(k in log for k in feasible): return 1.0
        return 0.5

    def _check_solver_result(self, solver_log, ground_truth):
        # (同原代码)
        if not solver_log or not ground_truth: return 0.0
        try:
            val = self._extract_objective_value(solver_log)
            gt = self._extract_objective_value(ground_truth)
            print(f"gt = {gt}, val = {val}")
            if val is not None and gt is not None:
                err = abs(val - gt) / max(abs(gt), 1e-8)
                if err < 0.01: return 1.0
                if err < 0.05: return 0.8
                if err < 0.1: return 0.6
                return 0.2
        except: pass
        return 0

    def _extract_objective_value(self, text: Any) -> float:
        """
        从文本中提取目标函数值，支持字符串解析和直接数字输入
        """
        # === 修复点：直接处理浮点数/整数类型 ===
        if isinstance(text, (int, float)):
            return float(text)
        # ====================================

        if not isinstance(text, str):
            return None
        import re
        cleaned_text = text.strip()
        try:
            return float(cleaned_text)
        except:
            pass
        patterns = [r"objective[:\s].*?\(?([+-]?\d*\.?\d+(?:e[+-]?\d+)?)"]
        for p in patterns:
            m = re.search(p, text.lower())
            if m:
                try:
                    ans = float(m.group(1))
                    return ans
                except: continue
        return None

    def _llm_judge(self, question, problem, modeling_result, ground_truth, solver_log):
        # (同原代码)
        try:
            prompt = self.evaluation_prompt.format(problem=problem, modeling_result=modeling_result, solver_log=solver_log, ground_truth=ground_truth, question=question)
            response = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.5, timeout=180)
            content = response.choices[0].message.content.strip()
            if "YES" in content.upper(): return 1.0
            return 0.0
        except: return 0.5

    # --- 核心修改：_run_solver ---

    def _run_solver(self, modeling_result: str, problem_text: str = "") -> str:
        """
        运行求解器并返回日志 (隔离环境版)
        
        Args:
            modeling_result: Python 代码
            problem_text: 问题描述（包含 JSON 数据）
        """
        try:
            # 创建一个临时的、隔离的目录
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # 1. 尝试从 problem_text 中提取 config.json 并保存到该临时目录
                config_path = os.path.join(temp_dir, "config.json")
                has_config = False
                if problem_text:
                    has_config = extract_and_save_json_to_path(problem_text, config_path)
                
                code = modeling_result + "\n\nprint(f\"optimal objective: {solve()}\")"
                code = code.replace("glpk", "highs")
                # 2. 将 Python 代码保存到该临时目录
                script_path = os.path.join(temp_dir, 'solution.py')
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                # 3. 执行代码
                # cwd=temp_dir 确保代码运行时，当前目录是这个临时目录
                # 这样代码里的 open('config.json') 就能找到刚才生成的那个文件
                result = subprocess.run(
                    ['python', 'solution.py'],
                    cwd=temp_dir,           # <--- 关键：设置工作目录为沙盒目录
                    capture_output=True,
                    text=True,
                    timeout=30              # 防止代码死循环
                )
                
                output = result.stdout + "\n" + result.stderr
                # print("*" * 20)
                # print(output)

                # 可选：如果提取失败但代码尝试读取，可以在 log 里加提示
                if not has_config and "No such file" in output and "config.json" in output:
                    output += "\n[System Info]: Config.json not found in problem description."
                    
                return output

        except subprocess.TimeoutExpired:
            return "Error: Execution timed out."
        except Exception as e:
            return f"Error executing solver: {str(e)}"