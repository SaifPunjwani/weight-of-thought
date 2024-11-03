import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Generate a larger set of entities
num_entities = 50
entities = [f'Entity_{i}' for i in range(num_entities)]

# Relations
relations = ['Parent', 'Grandparent', 'Sibling', 'Spouse', 'Ancestor']

# Mapping entities and relations to indices
entity2idx = {entity: idx for idx, entity in enumerate(entities)}
relation2idx = {relation: idx for idx, relation in enumerate(relations)}

# Synthetic data generation
np.random.seed(42)
facts = []

# Create parent relationships
for _ in range(100):
    parent = np.random.choice(entities)
    child = np.random.choice(entities)
    if parent != child:
        facts.append(('Parent', parent, child, 1.0))

# Negative samples
for _ in range(200):
    parent = np.random.choice(entities)
    child = np.random.choice(entities)
    if parent != child:
        facts.append(('Parent', parent, child, 0.0))

# Similar process for other relations (Sibling, Spouse)
# ...

class NeuralUnifier(nn.Module):
    def __init__(self, embedding_dim, num_entities):
        super(NeuralUnifier, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
    
    def forward(self, x, y):
        # Unification score between two entities
        x_emb = self.entity_embeddings(x)
        y_emb = self.entity_embeddings(y)
        score = -torch.norm(x_emb - y_emb, p=2, dim=1)
        return score

class NeuralTheoremProver(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100, max_depth=2):
        super(NeuralTheoremProver, self).__init__()
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.max_depth = max_depth
        self.unifier = NeuralUnifier(embedding_dim, num_entities)
        self.rule_embeddings = nn.Parameter(torch.randn(num_relations, embedding_dim))
    
    def prove(self, query_relation, head, tail, depth):
        if depth == 0:
            # Base case: direct fact retrieval
            return self.base_case(query_relation, head, tail)
        else:
            # Recursive case: apply rules
            score = torch.zeros(head.size(0), device=head.device)
            for relation in range(self.num_relations):
                # For simplicity, consider binary predicates
                score += self.apply_rule(query_relation, relation, head, tail, depth)
            return score
    
    def base_case(self, query_relation, head, tail):
        # Scoring function for facts
        head_emb = self.unifier.entity_embeddings(head)
        tail_emb = self.unifier.entity_embeddings(tail)
        relation_emb = self.rule_embeddings[query_relation]
        score = -torch.norm(head_emb + relation_emb - tail_emb, p=2, dim=1)
        return score
    
    def apply_rule(self, query_relation, relation, head, tail, depth):
        # Example rule: If R(head, z) and S(z, tail), then T(head, tail)
        z = torch.arange(self.unifier.entity_embeddings.num_embeddings, device=head.device)
        head = head.unsqueeze(1).repeat(1, z.size(0))
        tail = tail.unsqueeze(1).repeat(1, z.size(0))
        z = z.unsqueeze(0).repeat(head.size(0), 1)
        
        # Flatten tensors for scoring
        head_flat = head.contiguous().view(-1)
        tail_flat = tail.contiguous().view(-1)
        z_flat = z.contiguous().view(-1)
        
        # Recursive proof for subgoals
        score1 = self.prove(torch.full_like(head_flat, relation), head_flat, z_flat, depth - 1)
        score2 = self.prove(query_relation.repeat(z_flat.size(0)), z_flat, tail_flat, depth - 1)
        
        # Combine scores using a t-norm (e.g., product)
        combined_score = score1 * score2
        
        # Reshape combined_score back to (batch_size, num_entities)
        combined_score = combined_score.view(head.size(0), -1)
        
        # Return mean score over all possible z
        return combined_score.mean(dim=1)

# Prepare data loader
def get_data_loader(facts, batch_size=32):
    data = []
    for relation, head, tail, label in facts:
        data.append((
            torch.tensor([relation2idx[relation]]),
            torch.tensor([entity2idx[head]]),
            torch.tensor([entity2idx[tail]]),
            torch.tensor([label])
        ))
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
embedding_dim = 100
max_depth = 2
model = NeuralTheoremProver(num_entities, len(relations), embedding_dim, max_depth)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
losses = []

data_loader = get_data_loader(facts)

for epoch in range(num_epochs):
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        relation_idx, head_idx, tail_idx, label = batch
        label = label.float()
        scores = model.prove(relation_idx.squeeze(), head_idx.squeeze(), tail_idx.squeeze(), max_depth)
        loss = F.binary_cross_entropy_with_logits(scores, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Evaluate on some queries
def evaluate_query(model, query_relation, head_entity, tail_entity):
    with torch.no_grad():
        relation_idx = torch.tensor([relation2idx[query_relation]])
        head_idx = torch.tensor([entity2idx[head_entity]])
        tail_idx = torch.tensor([entity2idx[tail_entity]])
        score = model.prove(relation_idx, head_idx, tail_idx, max_depth)
        prob = torch.sigmoid(score)
        return prob.item()

# Example queries
print("Evaluating Queries:")
queries = [
    ('Grandparent', 'Entity_0', 'Entity_10'),
    ('Ancestor', 'Entity_5', 'Entity_25'),
    ('Sibling', 'Entity_2', 'Entity_3'),
    # Add more queries as needed
]

for query_relation, head_entity, tail_entity in queries:
    prob = evaluate_query(model, query_relation, head_entity, tail_entity)
    print(f"{query_relation}({head_entity}, {tail_entity}) = {prob:.4f}")

# Plotting Training Loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
