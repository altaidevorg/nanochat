"""
Bespoke-Stratos-17k dataset.
https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k

A dataset of questions with reasoning traces and answers across domains like code, mathematics, and scientific puzzles.
"""

from datasets import load_dataset
from tasks.common import Task


class MathThinking(Task):
    """
    Bespoke-Stratos-17k dataset for math and reasoning tasks.
    Contains questions, reasoning traces, and answers.
    """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "MathThinking split must be train"
        self.ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        question = row['question']
        reasoning = row.get('reasoning', '')
        answer = row['answer']
        
        if reasoning:
            assistant_content = f"{reasoning}\n\n{answer}"
        else:
            assistant_content = answer
        
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Compares the final answer from the assistant response with the ground truth answer.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        ground_truth_content = assistant_message['content']
        
        ground_truth_answer = ground_truth_content.split('\n\n')[-1].strip()
        if not ground_truth_answer:
            ground_truth_answer = ground_truth_content.strip()
        
        predicted_answer = assistant_response.split('\n\n')[-1].strip()
        if not predicted_answer:
            predicted_answer = assistant_response.strip()
        
        ground_truth_normalized = ground_truth_answer.lower().strip()
        predicted_normalized = predicted_answer.lower().strip()
        
        if ground_truth_normalized == predicted_normalized:
            return 1
        
        if ground_truth_normalized in predicted_normalized:
            return 1
        
        if predicted_normalized in ground_truth_normalized and len(predicted_normalized) > 0:
            return 1
        
        return 0

