from coe.llm import create_chat_completion, get_default_model


class BaseExpert(object):

    def __init__(self, name, description, model):
        self.name = name
        self.description = description
        self.model = model or get_default_model()
        self.forward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.FORWARD_TASK
        if hasattr(self, 'BACKWARD_TASK'):
            self.backward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.BACKWARD_TASK
    
    def _call_openai(self, prompt, **kwargs):
        """统一的OpenAI调用方法"""
        # 格式化prompt模板
        formatted_prompt = prompt.format(**kwargs)
        
        response = create_chat_completion(
            model=self.model,
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0
        )
        
        # 获取返回的文本内容
        content = response.choices[0].message.content or ""
        
        # 如果返回文本被```json```包围，去掉这些标记
        if content.startswith('```json') and content.endswith('```'):
            content = content[7:-3]  # 去掉开头的```json和结尾的```
        elif content.startswith('```') and content.endswith('```'):
            content = content[3:-3]  # 去掉开头的```和结尾的```
        
        return content.strip()

    def forward(self):
        pass

    def backward(self):
        pass

    def __str__(self):
        return f'{self.name}: {self.description}'
