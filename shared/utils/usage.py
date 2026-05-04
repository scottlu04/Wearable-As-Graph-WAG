from typing import Dict
from datetime import datetime
import tiktoken
class usage_tracker:
    def __init__(self):
        self.usage = {
            'openai': {'calls': 0, 'tokens': 0, 'estimated_cost': 0},
            'serper': {'calls': 0, 'estimated_cost': 0},
            'umls': {'calls': 0}
        }
        
    def log_openai_usage(self, num_input_tokens: int,num_output_tokens: int, model: str = "gpt-4"):
        """Log OpenAI API usage and estimate costs"""
        estimated_cost = estimate_query_cost(model, num_input_tokens, num_output_tokens)
        self.usage['llm']['calls'] += 1
        self.usage['llm']['tokens'] += num_input_tokens + num_output_tokens
        self.usage['llm']['estimated_cost'] += estimated_cost
        
    def log_serper_call(self):
        """Log Google Serper API usage"""
        self.usage['serper']['calls'] += 1
        # Approximate cost per call - update as needed
        self.usage['serper']['estimated_cost'] += 0.01
        
    def log_umls_call(self):
        """Log UMLS API usage"""
        self.usage['umls']['calls'] += 1
        
    def get_usage_report(self) -> Dict:
        return self.usage



def estimate_query_cost(model, input_tokens, output_tokens):
    # per 1M tokens
    prices = {
        "gpt-4o":{"input": 2.50,
                "output": 10,
                "cache_input": 1.25,
                "batch_input": 1.25,
                "batch_output": 5,
        },  
        "gpt-4o-mini": {
            "input": 0.15, 
            "output": 0.6,
            "cache_input": 0.075,
            "batch_input": 0.075,
            "batch_output": 0.3,
        },
        "o1": {
            "input": 15, 
            "output": 60,
            "cache_input": 7.5,
            "batch_input": None,
            "batch_output": None,
        },
        "o1": {
            "input": 15, 
            "output": 60,
            "cache_input": 7.5,
            "batch_input": None,
            "batch_output": None,
        },
        "o1-mini": {
            "input": 3, 
            "output": 12,
            "cache_input": 1.5,
            "batch_input": None,
            "batch_output": None,
        },
        "claude-3-5-sonnet": {
        "input": 3, 
        "output": 15,
        "cache_input": 0.3,
        "cache_output": 3.75,
        },
        "claude-3-opus": {
            "input": 15, 
            "output": 75,
            "cache_input": 1.5,
            "cache_output": 18.75,
        },
        "claude-3-5-haiku": {
                "input": 0.8, 
                "output": 4,
                "cache_input": 0.08,
                "cache_output": 1,
        },
        "gemini-1.5 pro": {
            "input": 1.25, 
            "output": 5,
        },
        "deepseek-v3": {
            "input": 0.14,
            "output": 0.28,
            "cache_input": 0.014,
            
        }


    }


    if model not in prices:
        raise ValueError("Unsupported model. Please choose from: gpt-4o, gpt-4o-mini, o1, gpt-3.5-turbo")
    
    input_cost = (input_tokens / 1000000) * prices[model]["input"]
    output_cost = (output_tokens / 1000000) * prices[model]["output"]
    total_cost = input_cost + output_cost
    
    return total_cost


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
