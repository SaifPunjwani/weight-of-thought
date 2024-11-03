import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

class NeuroSymbolicReasoner(nn.Module):
    def __init__(self, gpt2_model, num_classes, num_rules, hidden_size):
        super(NeuroSymbolicReasoner, self).__init__()
        self.gpt2 = gpt2_model
        self.num_rules = num_rules
        self.hidden_size = hidden_size

        # Ensure hidden_size is divisible by num_heads
        assert hidden_size % 8 == 0, "hidden_size must be divisible by num_heads"

        # Neural module
        self.neural_projection = nn.Linear(gpt2_model.config.n_embd, hidden_size)

        # Symbolic module
        self.rule_embeddings = nn.Parameter(torch.randn(num_rules, hidden_size))
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)

        # Output layer
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Neural module
        gpt2_output = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = gpt2_output.last_hidden_state  # Extract the last hidden state
        neural_output = self.neural_projection(last_hidden_state)  # [batch_size, seq_len, hidden_size]

        # Prepare batch size and sequence length
        batch_size = neural_output.size(0)
        seq_len = neural_output.size(1)

        # Prepare queries
        queries = neural_output.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]

        # Expand rule embeddings to match batch size
        keys_values = self.rule_embeddings.unsqueeze(1).expand(-1, batch_size, -1)  # [num_rules, batch_size, hidden_size]

        # Symbolic module
        rule_attention_output, _ = self.attention(
            queries,        # [seq_len, batch_size, hidden_size]
            keys_values,    # [num_rules, batch_size, hidden_size]
            keys_values     # [num_rules, batch_size, hidden_size]
        )
        rule_attention_output = rule_attention_output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size]

        # Combine neural and symbolic outputs
        combined_output = neural_output + rule_attention_output

        # Classification
        logits = self.classifier(combined_output[:, -1, :])  # Use the last token's representation
        return logits

class HybridModel(nn.Module):
    def __init__(self, gpt2_model, num_classes, num_rules, hidden_size):
        super(HybridModel, self).__init__()
        self.reasoner = NeuroSymbolicReasoner(gpt2_model, num_classes, num_rules, hidden_size)

    def forward(self, input_ids, attention_mask):
        return self.reasoner(input_ids, attention_mask)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Val Accuracy: {100 * correct / total:.2f}%")
        print()

def generate_reasoning(model, tokenizer, premise, device):
    model.eval()
    input_text = f"Premise: {premise}\nConclusion:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)

    validity = "Valid" if predicted.item() == 1 else "Invalid"
    return f"The argument is {validity}."

def main():
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token
    gpt2_config = GPT2Config.from_pretrained("gpt2", output_hidden_states=False)
    gpt2_model = GPT2Model.from_pretrained("gpt2", config=gpt2_config)
    gpt2_model.resize_token_embeddings(len(tokenizer))  # Resize embeddings for the new pad token

    # Prepare dataset
    premises = [
        "All men are mortal. Socrates is a man.",  # Valid
        "If it rains, the grass gets wet. It rained last night.",  # Valid
        "All cats have tails. Fluffy is a cat.",  # Valid
        "If the sun is shining, it's daytime. The sun is shining.",  # Valid
        "All birds can fly. Penguins are birds.",  # Invalid
        "If it's raining, the streets are wet. The streets are wet.",  # Invalid
        "All dogs are animals. All animals have four legs. Therefore, all dogs have four legs.",  # Valid
        "Some fruits are sweet. Lemons are fruits.",  # Invalid
        "If you study, you pass the test. You passed the test.",  # Invalid
        "All cars are vehicles. A bike is a vehicle.",  # Invalid
    ]

    labels = [1, 1, 1, 1, 0, 0, 1, 0, 0, 0]  # 1: Valid, 0: Invalid

    # Tokenize inputs
    tokenized_inputs = tokenizer(premises, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        list(zip(input_ids, attention_mask)), labels, test_size=0.2, random_state=42
    )

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.stack([x[0] for x in X_train]),
        torch.stack([x[1] for x in X_train]),
        torch.tensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.stack([x[0] for x in X_val]),
        torch.stack([x[1] for x in X_val]),
        torch.tensor(y_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    # Initialize model, loss function, and optimizer
    num_rules = 10  # Number of symbolic rules
    hidden_size = 512  # Adjusted hidden size
    model = HybridModel(gpt2_model, num_classes=2, num_rules=num_rules, hidden_size=hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=40)

    # Generate reasoning for new premises
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    new_premises = [
        "All mammals are warm-blooded. Whales are mammals.",  # Valid
        "If you eat well, you'll be healthy. You are healthy.",  # Invalid
        "Every student must submit homework. John is a student.",  # Valid
        "Birds can fly. Ostriches are birds.",  # Invalid
    ]
    for premise in new_premises:
        conclusion = generate_reasoning(model, tokenizer, premise, device)
        print(f"Premise: {premise}")
        print(f"Generated Conclusion: {conclusion}")
        print()

if __name__ == "__main__":
    main()

# This code implements a hybrid neuro-symbolic reasoning system using PyTorch and the GPT-2 language model.
# It combines neural networks (GPT-2) with a symbolic module for enhanced reasoning capabilities.
# 
# The HybridModel class integrates a neural module and a symbolic module within the NeuroSymbolicReasoner.
# This allows the model to leverage both neural language understanding and symbolic reasoning.
# 
# The NeuroSymbolicReasoner uses attention mechanisms to incorporate rule-based knowledge into the reasoning process.
# 
# The system is trained on a balanced dataset of logical premises and their validity.
# After training, it can classify the validity of new logical arguments and provide reasoning.
# 
# This approach demonstrates how to combine neural and symbolic AI techniques for more robust reasoning tasks,
# as inspired by the paper "https://arxiv.org/pdf/2309.13339".
