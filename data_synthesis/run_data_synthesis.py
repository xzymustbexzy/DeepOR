#!/usr/bin/env python3
"""
DeepOR Cold-start Data Synthesis Pipeline

Orchestrates parallel CoT (Chain-of-Thought) generation using the Chain-of-Experts
framework, followed by a reviser that polishes multi-expert outputs into a single,
natural reasoning trace.

Usage:
    python run_data_synthesis.py

Environment variables:
    SYNTHESIS_MODEL: LLM used for Chain-of-Experts synthesis (default: from coe.llm)
    REVISER_MODEL: LLM used for revising CoT (default: same as SYNTHESIS_MODEL)
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from coe.main import chain_of_experts
from coe.llm import get_default_model
from reviser import revise_cot

SYNTHESIS_MODEL = os.getenv('SYNTHESIS_MODEL', get_default_model())
REVISER_MODEL = os.getenv('REVISER_MODEL', SYNTHESIS_MODEL)
SYNTHESIS_CONCURRENCY = 10

SEED_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'seed_problems.json')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthesis_examples')

with open(SEED_DATA_PATH, 'r') as f:
    data = json.load(f)

# Count completed and total
total_problems = len(data)
completed_problems = sum(1 for i in range(total_problems) if os.path.exists(os.path.join(OUTPUT_DIR, f'prob_{i}', 'cot.txt')))
print(f"Total problems: {total_problems}, Completed: {completed_problems}, Pending: {total_problems - completed_problems}")


def process_problem(i, item):
    try:
        cot_path = os.path.join(OUTPUT_DIR, f'prob_{i}', 'cot.txt')
        if os.path.exists(cot_path):
            return ('skipped', i, 0, None)

        description = item['input']
        problem = {'description': description}

        problem_start_time = time.time()
        cot = chain_of_experts(
            problem,
            max_collaborate_nums=4,
            model_name=SYNTHESIS_MODEL,
            enable_reflection=False,
            max_trials=1,
        )
        revised_cot = revise_cot(description, cot, model_name=REVISER_MODEL)

        os.makedirs(os.path.join(OUTPUT_DIR, f'prob_{i}'), exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, f'prob_{i}', 'cot.txt'), 'w') as f:
            f.write(cot)
        with open(os.path.join(OUTPUT_DIR, f'prob_{i}', 'revised_cot.txt'), 'w') as f:
            f.write(revised_cot)
        with open(os.path.join(OUTPUT_DIR, f'prob_{i}', 'description.txt'), 'w') as f:
            f.write(description)
        with open(os.path.join(OUTPUT_DIR, f'prob_{i}', 'answer.txt'), 'w') as f:
            f.write(str(item['answer']))

        return ('success', i, time.time() - problem_start_time, None)
    except Exception as e:
        return ('failed', i, 0, str(e))


# Timing
start_time = time.time()
processed_count = 0
failed_count = 0
time_records = []

pending_items = [
    (i, item)
    for i, item in enumerate(data)
    if not os.path.exists(os.path.join(OUTPUT_DIR, f'prob_{i}', 'cot.txt'))
]

with ThreadPoolExecutor(max_workers=SYNTHESIS_CONCURRENCY) as executor:
    futures = [executor.submit(process_problem, i, item) for i, item in pending_items]

    for future in as_completed(futures):
        status, i, problem_time, error = future.result()

        if status == 'success':
            processed_count += 1
            time_records.append(problem_time)

            elapsed_time = time.time() - start_time
            avg_time_per_problem = sum(time_records) / len(time_records)
            remaining_problems = len(pending_items) - processed_count - failed_count
            estimated_remaining_time = avg_time_per_problem * max(remaining_problems, 0)

            elapsed_str = str(timedelta(seconds=int(elapsed_time)))
            remaining_str = str(timedelta(seconds=int(estimated_remaining_time)))
            current_problem_str = str(timedelta(seconds=int(problem_time)))

            print(f"\n{'='*80}")
            print(f"Completed problem {completed_problems + processed_count}/{total_problems} (prob_{i})")
            print(f"Current problem time: {current_problem_str}")
            print(f"Average time per problem: {str(timedelta(seconds=int(avg_time_per_problem)))}")
            print(f"Elapsed time: {elapsed_str}")
            print(f"Estimated remaining: {remaining_str}")
            print(f"Success: {processed_count}, Failed: {failed_count}")
            print(f"{'='*80}\n")
        elif status == 'failed':
            failed_count += 1
            print(f"\n{'='*80}")
            print(f"prob_{i} failed: {error}")
            print(f"Skipping this problem, continuing with the next...")
            print(f"Success: {processed_count}, Failed: {failed_count}")
            print(f"{'='*80}\n")

# Final statistics
print(f"\n{'='*80}")
print(f"All tasks completed!")
print(f"Total time: {str(timedelta(seconds=int(time.time() - start_time)))}")
print(f"Successfully processed: {processed_count}")
print(f"Failed / skipped: {failed_count}")
print(f"Total completed: {completed_problems + processed_count}/{total_problems}")
print(f"{'='*80}\n")
