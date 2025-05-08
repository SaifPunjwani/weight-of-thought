"""
Language encoders for the Weight-of-Thought model.

This module contains various language encoder implementations that can be used
to convert text inputs into embeddings for the WoT reasoning model.
"""

import torch
import torch.nn as nn
from transformers import (
    GPT2Model, GPT2Tokenizer, GPT2Config,
    BertModel, BertTokenizer, BertConfig,
    RobertaModel, RobertaTokenizer, RobertaConfig
)


class LanguageEncoder(nn.Module):
    """
    Base language encoder using GPT-2.
    
    This encoder uses a pre-trained GPT-2 model to convert text
    inputs into fixed-dimensional embeddings.
    """
    
    def __init__(self, model_name='gpt2', output_type='last_token'):
        """
        Initialize the language encoder.
        
        Args:
            model_name: Name of the pre-trained model (default: 'gpt2')
            output_type: Type of output to use ('last_token', 'mean', 'cls')
                         (default: 'last_token')
        """
        super(LanguageEncoder, self).__init__()
        
        # Load pre-trained model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure model
        config = GPT2Config.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name, config=config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.output_dim = config.n_embd
        self.output_type = output_type
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the encoder.
        
        Args:
            input_ids: Token IDs from tokenizer [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        if self.output_type == 'last_token':
            # Use the representation of the last token
            # For each sequence, get the last non-padded token
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(last_token_indices.size(0), device=last_hidden_state.device)
            sentence_embedding = last_hidden_state[batch_indices, last_token_indices]
        elif self.output_type == 'mean':
            # Mean pooling (average of all token embeddings)
            # Only consider non-padded tokens
            masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1)
            summed = masked_hidden.sum(dim=1)
            sentence_embedding = summed / attention_mask.sum(dim=1, keepdim=True)
        else:
            # Default to last token
            sentence_embedding = last_hidden_state[:, -1, :]
        
        return sentence_embedding


class BertEncoder(nn.Module):
    """
    Language encoder using BERT.
    
    This encoder uses a pre-trained BERT model to convert text
    inputs into fixed-dimensional embeddings.
    """
    
    def __init__(self, model_name='bert-base-uncased', output_type='cls'):
        """
        Initialize the BERT encoder.
        
        Args:
            model_name: Name of the pre-trained model (default: 'bert-base-uncased')
            output_type: Type of output to use ('cls', 'mean', 'pooler')
                         (default: 'cls')
        """
        super(BertEncoder, self).__init__()
        
        # Load pre-trained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Configure model
        config = BertConfig.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, config=config)
        
        self.output_dim = config.hidden_size
        self.output_type = output_type
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the encoder.
        
        Args:
            input_ids: Token IDs from tokenizer [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            return_dict=True
        )
        
        if self.output_type == 'cls':
            # Use the CLS token representation
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
        elif self.output_type == 'pooler':
            # Use the pooler output
            sentence_embedding = outputs.pooler_output
        elif self.output_type == 'mean':
            # Mean pooling (average of all token embeddings)
            masked_hidden = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            summed = masked_hidden.sum(dim=1)
            sentence_embedding = summed / attention_mask.sum(dim=1, keepdim=True)
        else:
            # Default to CLS token
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
        
        return sentence_embedding


class RobertaEncoder(nn.Module):
    """
    Language encoder using RoBERTa.
    
    This encoder uses a pre-trained RoBERTa model to convert text
    inputs into fixed-dimensional embeddings.
    """
    
    def __init__(self, model_name='roberta-base', output_type='cls'):
        """
        Initialize the RoBERTa encoder.
        
        Args:
            model_name: Name of the pre-trained model (default: 'roberta-base')
            output_type: Type of output to use ('cls', 'mean')
                         (default: 'cls')
        """
        super(RobertaEncoder, self).__init__()
        
        # Load pre-trained model and tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Configure model
        config = RobertaConfig.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name, config=config)
        
        self.output_dim = config.hidden_size
        self.output_type = output_type
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the encoder.
        
        Args:
            input_ids: Token IDs from tokenizer [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            return_dict=True
        )
        
        if self.output_type == 'cls':
            # Use the <s> token representation (equivalent to CLS in BERT)
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
        elif self.output_type == 'mean':
            # Mean pooling (average of all token embeddings)
            masked_hidden = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            summed = masked_hidden.sum(dim=1)
            sentence_embedding = summed / attention_mask.sum(dim=1, keepdim=True)
        else:
            # Default to <s> token
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
        
        return sentence_embedding


class EncoderFactory:
    """
    Factory class for creating different types of language encoders.
    """
    
    @staticmethod
    def create_encoder(encoder_type='gpt2', model_name=None, output_type=None):
        """
        Create a language encoder of the specified type.
        
        Args:
            encoder_type: Type of encoder ('gpt2', 'bert', 'roberta')
                          (default: 'gpt2')
            model_name: Name of the pre-trained model (default: None)
            output_type: Type of output embedding (default: None)
            
        Returns:
            Initialized language encoder
        """
        if encoder_type.lower() == 'gpt2':
            model_name = model_name or 'gpt2'
            output_type = output_type or 'last_token'
            return LanguageEncoder(model_name, output_type)
        elif encoder_type.lower() == 'bert':
            model_name = model_name or 'bert-base-uncased'
            output_type = output_type or 'cls'
            return BertEncoder(model_name, output_type)
        elif encoder_type.lower() == 'roberta':
            model_name = model_name or 'roberta-base'
            output_type = output_type or 'cls'
            return RobertaEncoder(model_name, output_type)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")