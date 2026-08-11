"""
Turkish Chain of Thought Dataset
altaidevorg/gemini-turkish-thinking
"""

from datasets import load_dataset, concatenate_datasets
from tasks.common import Task

class TurkishThinking(Task):

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        
        assert split in ["train"]

        ds_v1 = load_dataset("altaidevorg/gemini-turkish-thinking", split=split)
        ds_v2 = load_dataset("altaidevorg/gemini-turkish-thinking-v2", split=split)
        
        combined_ds = concatenate_datasets([ds_v1, ds_v2])
        
        self.ds = combined_ds.shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        
        user_message = row['user']
        assistant_message = row['assistant'] 

        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        conversation = {
            "messages": messages,
            "ground_truth": assistant_message 
        }
        
        return conversation

    def evaluate(self, conversation, assistant_response):
        ground_truth = conversation['ground_truth']
        is_correct = (assistant_response.strip() == ground_truth.strip())
        return int(is_correct)