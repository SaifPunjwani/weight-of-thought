"""
Unit tests for the Weight-of-Thought model implementation.
"""

import os
import unittest
import tempfile

import torch
import numpy as np
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wot.models import WOTReasoner
from wot.models.encoders import LanguageEncoder, BertEncoder
from wot.data import ReasoningDataset
from wot.data.tasks import generate_all_tasks


class TestLanguageEncoder(unittest.TestCase):
    """Test the language encoder component."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        self.encoder = LanguageEncoder(model_name='gpt2')
        
        # Create a sample batch
        self.tokenizer = self.encoder.tokenizer
        self.sample_text = ["This is a test sentence", "Another example"]
        self.tokenized = self.tokenizer(
            self.sample_text, 
            padding='max_length', 
            truncation=True, 
            max_length=20, 
            return_tensors='pt'
        )
        
    def test_initialization(self):
        """Test encoder initialization."""
        self.assertIsNotNone(self.encoder)
        self.assertEqual(self.encoder.output_dim, 768)  # GPT-2 small has 768 dims
        
    def test_forward_pass(self):
        """Test encoder forward pass."""
        input_ids = self.tokenized['input_ids']
        attention_mask = self.tokenized['attention_mask']
        
        # Forward pass
        output = self.encoder(input_ids, attention_mask)
        
        # Check output shape
        self.assertEqual(output.shape, (len(self.sample_text), self.encoder.output_dim))
        
    def test_alternative_encoder(self):
        """Test alternative encoder (BERT)."""
        bert_encoder = BertEncoder(model_name='bert-base-uncased')
        
        # Tokenize with BERT tokenizer
        bert_tokenized = bert_encoder.tokenizer(
            self.sample_text, 
            padding='max_length', 
            truncation=True, 
            max_length=20, 
            return_tensors='pt'
        )
        
        input_ids = bert_tokenized['input_ids']
        attention_mask = bert_tokenized['attention_mask']
        
        # Forward pass
        output = bert_encoder(input_ids, attention_mask)
        
        # Check output shape (should be same dimension as BERT hidden size)
        self.assertEqual(output.shape, (len(self.sample_text), bert_encoder.output_dim))


class TestWeightOfThoughts(unittest.TestCase):
    """Test the WebOfThoughts model implementation."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Create a small model for testing
        self.reasoner = WOTReasoner(
            hidden_dim=64,  # Small for fast testing
            num_nodes=4,
            num_reasoning_steps=2,
            lr=1e-4
        )
        
        # Create a small dataset for testing
        tasks = generate_all_tasks(num_each=2)
        self.dataset = ReasoningDataset(tasks, self.reasoner.encoder.tokenizer)
        self.dataloader = DataLoader(self.dataset, batch_size=2)
        
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.reasoner)
        self.assertIsNotNone(self.reasoner.wot_model)
        self.assertEqual(self.reasoner.wot_model.hidden_dim, 64)
        self.assertEqual(self.reasoner.wot_model.num_nodes, 4)
        self.assertEqual(self.reasoner.wot_model.num_reasoning_steps, 2)
        
    def test_forward_pass(self):
        """Test model forward pass."""
        # Get a sample batch
        batch = next(iter(self.dataloader))
        input_ids = batch['input_ids'].to(self.reasoner.device)
        attention_mask = batch['attention_mask'].to(self.reasoner.device)
        
        # Encode text
        text_embedding = self.reasoner.encoder(input_ids, attention_mask)
        
        # Get first task type in the batch
        task_type = batch['task_type'][0]
        
        # Forward pass through WOT model
        class_logits, numeric_prediction = self.reasoner.wot_model(text_embedding, task_type)
        
        # Check output shapes based on task type
        if task_type in ['syllogism', 'geometry']:
            self.assertIsNotNone(class_logits)
            self.assertEqual(class_logits.shape[1], 2)  # Binary classification
        else:
            self.assertIsNotNone(numeric_prediction)
            self.assertEqual(numeric_prediction.shape[1], 1)  # Scalar prediction
        
    def test_inference(self):
        """Test model inference."""
        # Test syllogism inference
        syllogism_question = "If all Bloops are Razzies and all Razzies are Wazzies, are all Bloops definitely Wazzies? Answer with Yes or No."
        syllogism_answer = self.reasoner.infer(syllogism_question, "syllogism")
        self.assertIn(syllogism_answer, ["Yes", "No"])
        
        # Test math sequence inference
        math_question = "What is the next number in the sequence: 2, 4, 6, 8, 10, ...?"
        math_answer = self.reasoner.infer(math_question, "math_sequence")
        # Should be a string representing a number
        self.assertTrue(math_answer.isdigit() or (math_answer[0] == '-' and math_answer[1:].isdigit()))
    
    def test_save_load(self):
        """Test saving and loading the model."""
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # Save the model
            self.reasoner.save_model(tmp.name)
            
            # Create a new reasoner
            new_reasoner = WOTReasoner(
                hidden_dim=64,
                num_nodes=4,
                num_reasoning_steps=2
            )
            
            # Load the saved model
            new_reasoner.load_model(tmp.name)
            
            # Verify the parameters are the same
            # Check one parameter from the encoder
            param1 = self.reasoner.encoder.model.wte.weight
            param2 = new_reasoner.encoder.model.wte.weight
            self.assertTrue(torch.allclose(param1, param2))
            
            # Check one parameter from the WOT model
            param1 = self.reasoner.wot_model.embedding[0].weight
            param2 = new_reasoner.wot_model.embedding[0].weight
            self.assertTrue(torch.allclose(param1, param2))
    
    def test_train_step(self):
        """Test a single training step."""
        # Get optimizer and loss function references
        optimizer = self.reasoner.optimizer
        classification_loss = self.reasoner.classification_loss
        
        # Save initial parameters to check for updates
        initial_params = {}
        for name, param in self.reasoner.wot_model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.clone().detach()
        
        # Set models to training mode
        self.reasoner.encoder.train()
        self.reasoner.wot_model.train()
        
        # Get a batch
        batch = next(iter(self.dataloader))
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Get task-specific batches
        task_types = batch['task_type']
        unique_task_types = set(task_types)
        
        # Process each task type
        batch_loss = 0.0
        for task_type in unique_task_types:
            # Get indices for this task type
            indices = [i for i, t in enumerate(task_types) if t == task_type]
            
            # Get embeddings for this task type
            input_ids = batch['input_ids'][indices].to(self.reasoner.device)
            attention_mask = batch['attention_mask'][indices].to(self.reasoner.device)
            text_embedding = self.reasoner.encoder(input_ids, attention_mask)
            
            # Forward pass
            class_logits, numeric_prediction = self.reasoner.wot_model(text_embedding, task_type)
            
            # Loss calculation based on task type
            if task_type in ['syllogism', 'geometry']:
                labels = batch['label'][indices].to(self.reasoner.device)
                loss = classification_loss(class_logits, labels)
            else:
                numeric_labels = batch['numeric_label'][indices].to(self.reasoner.device).float()
                numeric_pred = numeric_prediction.view(-1)
                numeric_labels = numeric_labels.view(-1)
                loss = torch.nn.functional.mse_loss(numeric_pred, numeric_labels)
            
            batch_loss += loss
        
        # Backward pass
        batch_loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Check that parameters have been updated
        params_changed = False
        for name, param in self.reasoner.wot_model.named_parameters():
            if param.requires_grad:
                if not torch.allclose(initial_params[name], param):
                    params_changed = True
                    break
        
        self.assertTrue(params_changed)


if __name__ == '__main__':
    unittest.main()