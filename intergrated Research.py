'''
Multi-Modal Research Agent: CNN + Transformer + RL

This script defines a simplified research agent that:
1. Uses a CNN-based VisionEncoder to extract features from document figures/images.
2. Uses a Transformer-based TextEncoder (HuggingFace) for text encoding.
3. Combines these embeddings in a PolicyNetwork (an actor-critic RL agent) to decide research actions.
4. Interacts with a stubbed ResearchEnvironment to simulate research tasks.

Dependencies:
- torch, torchvision
- transformers (HuggingFace)
- gym (for environment abstraction)
'''
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.models import resnet18
from transformers import BertModel, BertTokenizer
from torch.distributions import Categorical

# ---- Vision Encoder (CNN) ----
class VisionEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        # Pretrained ResNet18 backbone
        self.backbone = resnet18(pretrained=True)
        # Replace final layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, output_dim)
    def forward(self, image):
        # image: (B,3,H,W)
        return self.backbone(image)  # (B, output_dim)

# ---- Text Encoder (Transformer) ----
class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', output_dim=256):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, output_dim)
    def encode(self, texts):
        # texts: list[str]
        enc = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert(**enc)
        # Use [CLS] token embedding
        cls = outputs.last_hidden_state[:,0]  # (B, hidden)
        return self.proj(cls)                # (B, output_dim)

# ---- Research Environment (Stub) ----
class ResearchEnv(gym.Env):
    """
    Stub environment where agent chooses actions:
    0=Read Text, 1=Analyze Figure, 2=Summarize, 3=Query Web
    Returns multimodal observations and rewards.
    """
    def __init__(self, documents):
        super().__init__()
        self.documents = documents  # list of {'text':str,'image':PIL}
        self.idx = 0
        self.action_space = gym.spaces.Discrete(4)
        # obs: dict with 'text' and 'image'
        self.observation_space = gym.spaces.Dict({
            'text': gym.spaces.Text(max_length=512),
            'image': gym.spaces.Box(low=0, high=1, shape=(3,224,224))
        })
    def reset(self):
        self.idx = 0
        return self._obs()
    def step(self, action):
        # reward logic stub
        reward = 0.0
        if action == 0 and 'text' in self.documents[self.idx]: reward = 0.1
        if action == 1 and 'image' in self.documents[self.idx]: reward = 0.1
        if action == 2: reward = 0.2
        if action == 3: reward = 0.3
        self.idx += 1
        done = self.idx >= len(self.documents)
        return (self._obs(), reward, done, {})
    def _obs(self):
        doc = self.documents[self.idx]
        img = doc.get('image')
        if img is not None:
            transform = T.Compose([T.Resize((224,224)), T.ToTensor()])
            img = transform(img)
        else:
            img = torch.zeros(3,224,224)
        return {'text': doc.get('text',''), 'image': img}

# ---- Policy Network (Actor-Critic) ----
class ResearchAgent(nn.Module):
    def __init__(self, vision_dim=256, text_dim=256, hidden=128, n_actions=4):
        super().__init__()
        self.vision_enc = VisionEncoder(output_dim=vision_dim)
        self.text_enc   = TextEncoder(output_dim=text_dim)
        # shared MLP
        self.shared = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic= nn.Linear(hidden, 1)
    def forward(self, obs):
        # obs: dict
        v_emb = self.vision_enc(obs['image'])    # (B, vision_dim)
        t_emb = self.text_enc.encode([obs['text']])  # (1, text_dim)
        x = torch.cat([v_emb, t_emb], dim=1)
        h = self.shared(x)
        return Categorical(logits=self.actor(h)), self.critic(h)

# ---- Training Loop (A2C-style) ----
def train_agent(agent, env, episodes=1000, gamma=0.99, lr=1e-4):
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    for ep in range(episodes):
        obs = env.reset()
        done = False
        log_probs, values, rewards = [], [], []
        while not done:
            dist, value = agent(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            obs_next, reward, done, _ = env.step(action.item())
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float))
            obs = obs_next
        # compute returns and advantage
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.cat(returns)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        advantage = returns - values.squeeze()
        # losses
        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % 100 == 0:
            print(f"Episode {ep}, Return: {returns.sum().item():.2f}")
    return agent

# ---- Example Usage ----
if __name__ == '__main__':
    from PIL import Image
    # Stub documents: replace with real PDFs figures/texts
    docs = [
        {'text':'Introduction to deep learning.', 'image':Image.new('RGB',(224,224))},
        {'text':'Convolutional networks and vision.', 'image':Image.new('RGB',(224,224))},
        {'text':'Transformer architectures.', 'image':Image.new('RGB',(224,224))}
    ]
    env = ResearchEnv(docs)
    agent = ResearchAgent()
    trained = train_agent(agent, env)
    print("Training complete.")
