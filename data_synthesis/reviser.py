import os
from coe.llm import create_chat_completion, get_default_model

PROMPT_TEMPLATE = """
You are an expert in the field of operations research and optimization. Now you are faced with the following operations research problem:
{problem_description}

Regarding this operations research problem, there is a modeling approach for it. Modeling refers to the process of transforming the problem into a mathematical model and then converting it into a solution program. The thought process for this modeling approach is as follows:
{modeling_thought}

In this thought process, the content enclosed by <think></think> represents the intermediate reasoning process, followed by the final modeling result (which is a solution program written in Pyomo).
This intermediate reasoning process is composed of several expert models, each acting as a human expert, analyzing one aspect of the operations research modeling problem in a specific sequence. This approach makes the modeling process smoother and allows the problem to be solved step by step.
Now, you need to revise this thought process so that it appears fluent and natural, with clear logical flow, and as if it were written by a single expert.

Your output requirements include:
- You need to appropriately rewrite the thought process so that it reads like it was written by a person, rather than being a stack of comments from different experts (THIS IS IMPORTANT).
- Use split line '---' to separate each part.
- The transition between each aspect of the thought process should be smooth and logically coherent.
- When the reasoning involves reflection, you should introduce transitional logic, be willing to self-correct, and revise previous conclusions when necessary.
- You should preserve the original logic and answers as much as possible; only modify the core logic if there are obvious errors.
- Your output format should remain as <think>{{thought process}}</think>```python{{final python code}}```, and you should not add any extra text such as "OK" or "Sure."
- Your role is that of an operations research modeling expert.
- You can keep the content of code and code review (reflection) within the <think></think> tag.

Here is an example of a well-revised thought process:

{example}

Below is your revised output:
"""


def revise_cot(problem_description, cot, model_name=None):
    model_name = model_name or get_default_model()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    example_file_path = os.path.join(current_dir, 'cot_revised.txt')

    with open(example_file_path, 'r', encoding='utf-8') as f:
        example_content = f.read()

    # Escape braces to avoid format conflicts
    example_escaped = example_content.replace('{', '{{').replace('}', '}}')
    cot_escaped = cot.replace('{', '{{').replace('}', '}}')

    response = create_chat_completion(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    problem_description=problem_description,
                    modeling_thought=cot_escaped,
                    example=example_escaped
                )
            }
        ],
        temperature=0
    )

    output = response.choices[0].message.content
    return output
