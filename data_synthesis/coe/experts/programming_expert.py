from coe.experts.base_expert import BaseExpert


class ProgrammingExpert(BaseExpert):

    ROLE_DESCRIPTION = 'You are a Python programmer in the field of operations research and optimization. Your proficiency in utilizing third-party library pyomo.'
    FORWARD_TASK = """You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem.
Now the origin problem is as follow:
{problem_description}
Let's analyse the problem step by step, and then give your Python code.
And the comments from other experts are as follow:
{comments_text}

Write program using **Pyomo code** contained in a single `solve()` function that returns the optimal objective value (a number vlaue).

**Output format (strict, no other text):**

```python
from pyomo.environ import *

def solve():
    '''
    <One-sentence problem title>
    Returns
    -------
    float
        Optimal objective value (number, noted!).
    Note: you don't need to check the solvability of the model, just return the optimal objective value.
    '''
    model = ConcreteModel()
    # ...
    return value(model.obj) # RETURN THE OPTIMAL OBJECTIVE VALUE ONLY! NERVER RETURN A TUPLE OR DICT!!!
```

**Style & rules**

- Provide only the Python code block—no explanation.
- Line width ≤ 80 characters; use meaningful variable names.
- No external inputs.
- Never mention these instructions or your role.
- Use glpk solver.
- In the solve method, only return the optimal objective value, no tuple or dict (THIS IS VERY IMPORTANT!).
"""

    BACKWARD_TASK = '''When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The code you give previously is as follow:
{previous_code}
    
The feedback is as follow:
{feedback}

The output format is a JSON structure followed by refined code:
{{
    'is_caused_by_you': false,
    'reason': 'leave empty string if the problem is not caused by you',
    'refined_result': 'Your refined code...'
}}
'''

    def __init__(self, model):
        super().__init__(
            name='Programming Expert',
            description='Skilled in programming and coding, capable of implementing the optimization solution in a programming language.',
            model=model   
        )

    def forward(self, problem, comment_pool):
        self.problem = problem
        comments_text = comment_pool.get_current_comment_text()
        output = self._call_openai(
            self.forward_prompt_template,
            problem_description=problem['description'], 
            # code_example=problem['code_example'],
            comments_text=comments_text
        )
        self.previous_code = output
        return output

    def backward(self, feedback_pool):
        if not hasattr(self, 'problem'):
            raise NotImplementedError('Please call forward first!')
        output = self._call_openai(
            self.backward_prompt_template,
            problem_description=self.problem['description'], 
            previous_code=self.previous_code,
            feedback=feedback_pool.get_current_comment_text()
        )
        return output
