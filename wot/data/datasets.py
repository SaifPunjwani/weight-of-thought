"""
Dataset implementations for Weight-of-Thought model.

This module provides dataset classes for loading and preprocessing
reasoning tasks for training and evaluation.
"""

import torch
from torch.utils.data import Dataset


class ReasoningDataset(Dataset):
    """
    Dataset for reasoning tasks used in the Weight-of-Thought model.
    
    This dataset handles various types of reasoning tasks, including:
    - Syllogistic reasoning (classification)
    - Mathematical sequence prediction (regression)
    - Algebraic problems (regression)
    - Combinatorial counting (regression)
    - Geometric reasoning (classification)
    
    It converts each task's question into tokens and prepares
    appropriate labels based on the task type.
    """
    
    def __init__(self, tasks, tokenizer, max_length=128):
        """
        Initialize the reasoning dataset.
        
        Args:
            tasks: List of task dictionaries, each containing 'question', 'answer', and 'type'
            tokenizer: Tokenizer to use for encoding questions
            max_length: Maximum length for tokenization (default: 128)
        """
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """Return the number of tasks in the dataset."""
        return len(self.tasks)
    
    def __getitem__(self, idx):
        """
        Get a dataset item by index.
        
        Args:
            idx: Index of the task
            
        Returns:
            Dictionary containing processed task data
        """
        task = self.tasks[idx]
        
        # Tokenize the question
        encoding = self.tokenizer(
            task['question'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get input IDs and attention mask
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Process the answer based on task type
        task_type = task['type']
        
        # Initialize both label types, but with placeholder values for the non-applicable type
        if task_type in ['syllogism', 'geometry']:
            # Binary classification (Yes/No)
            label = 1 if task['answer'] == 'Yes' else 0
            label = torch.tensor(label, dtype=torch.long)
            # Placeholder for numeric tasks
            numeric_label = torch.tensor(0.0, dtype=torch.float)
        else:
            # Numeric prediction
            numeric_label = float(task['answer'])
            numeric_label = torch.tensor(numeric_label, dtype=torch.float)
            # Placeholder for classification tasks
            label = torch.tensor(0, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label,
            'numeric_label': numeric_label,
            'task_type': task_type,
            'question': task['question'],
            'answer': task['answer']
        }


class AugmentedReasoningDataset(ReasoningDataset):
    """
    Extended reasoning dataset with data augmentation capabilities.
    
    This dataset extends the base ReasoningDataset with:
    - Synonym replacement for common words
    - Question paraphrasing
    - Random task combination (for more complex reasoning)
    - Noise injection for robustness
    """
    
    def __init__(self, tasks, tokenizer, max_length=128, augment_prob=0.3):
        """
        Initialize the augmented reasoning dataset.
        
        Args:
            tasks: List of task dictionaries
            tokenizer: Tokenizer to use for encoding questions
            max_length: Maximum length for tokenization (default: 128)
            augment_prob: Probability of applying augmentation (default: 0.3)
        """
        super().__init__(tasks, tokenizer, max_length)
        self.augment_prob = augment_prob
        self.synonyms = self._load_synonyms()
    
    def _load_synonyms(self):
        """
        Load synonym dictionary for word replacement.
        
        Returns:
            Dictionary of words and their synonyms
        """
        # Simple synonym map (would be more extensive in production)
        return {
            'definitely': ['certainly', 'surely', 'absolutely'],
            'possible': ['feasible', 'potential', 'conceivable'],
            'have': ['possess', 'own', 'hold'],
            'many': ['numerous', 'several', 'multiple'],
            'together': ['combined', 'jointly', 'in total'],
            'everyone': ['everybody', 'each person', 'all people'],
            'exactly': ['precisely', 'just', 'accurately'],
            'total': ['entire', 'complete', 'whole']
        }
    
    def _augment_question(self, question):
        """
        Apply text augmentation to a question.
        
        Args:
            question: Original question text
            
        Returns:
            Augmented question text
        """
        import random
        
        words = question.split()
        augmented_words = []
        
        for word in words:
            # Strip punctuation for checking
            clean_word = word.lower().strip('.,?!')
            
            # 20% chance to replace with synonym if available
            if clean_word in self.synonyms and random.random() < 0.2:
                replacement = random.choice(self.synonyms[clean_word])
                
                # Preserve capitalization and punctuation
                if word[0].isupper():
                    replacement = replacement.capitalize()
                
                # Add back any punctuation
                if not word[-1].isalnum():
                    replacement = replacement + word[-1]
                
                augmented_words.append(replacement)
            else:
                augmented_words.append(word)
        
        return ' '.join(augmented_words)
    
    def __getitem__(self, idx):
        """
        Get a dataset item with potential augmentation.
        
        Args:
            idx: Index of the task
            
        Returns:
            Dictionary containing processed task data
        """
        import random
        
        # Get the original item
        item = super().__getitem__(idx)
        
        # Apply augmentation with probability augment_prob
        if random.random() < self.augment_prob:
            augmented_question = self._augment_question(item['question'])
            
            # Re-tokenize the augmented question
            encoding = self.tokenizer(
                augmented_question,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Update input IDs and attention mask
            item['input_ids'] = encoding['input_ids'].squeeze(0)
            item['attention_mask'] = encoding['attention_mask'].squeeze(0)
            item['question'] = augmented_question
            item['augmented'] = True
        else:
            item['augmented'] = False
        
        return item