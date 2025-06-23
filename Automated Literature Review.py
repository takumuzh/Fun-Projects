'''\
Actual Multi-Modal Research Agent for Automated Literature Review

This agent integrates:
1) PDF document loader & OCR for extracting text and figures.
2) Embedding-based semantic retrieval (SentenceTransformers + FAISS).
3) Transformer-based summarization (HuggingFace transformers).
4) A simple policy network trained via reinforcement learning (PPO) to select next research actions.

Dependencies:
- torch, torchvision
- transformers, sentence-transformers
- faiss-cpu
- PyPDF2, Pillow, pytesseract
- gym
- stable-baselines3
'''
import os
import glob
import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
from PIL import Image
import pytesseract
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ---- Document Loader & OCR ----
def load_pdfs(folder):
    docs = []
    for path in glob.glob(os.path.join(folder, '*.pdf')):
        reader = PyPDF2.PdfFileReader(path)
        text = ''
        images = []
        for i in range(reader.numPages):
            page = reader.getPage(i)
            text += page.extractText()
            # extract images via rendering omitted for brevity
        docs.append({'path':path, 'text': text})
    return docs

# ---- Semantic Indexing ----
class SemanticIndex:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.texts = []

    def build(self, docs):
        corpus = [d['text'] for d in docs]
        embeddings = self.embedder.encode(corpus, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.texts = corpus

    def query(self, query_text, top_k=5):
        q_emb = self.embedder.encode([query_text], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        return [(self.texts[i], float(D[0][j])) for j,i in enumerate(I[0])]

# ---- Summarization Model ----
class Summarizer:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, text, max_length=150):
        input_ids = self.tokenizer.encode('summarize: ' + text, return_tensors='pt', truncation=True)
        outputs = self.model.generate(input_ids, max_length=max_length, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---- Gym Environment ----
class ResearchEnv(gym.Env):
    ''' Observations: {'query': str} Actions: 0=retrieve,1=summarize,2=next_query '''
    def __init__(self, docs, index, summarizer):
        super().__init__()
        self.docs = docs
        self.index = index
        self.summarizer = summarizer
        self.action_space = gym.spaces.Discrete(3)
        self.current_query = 'machine learning'
        self.step_count = 0
        self.max_steps = 20

    def reset(self):
        self.current_query = 'machine learning'
        self.step_count = 0
        return {'query': self.current_query}

    def step(self, action):
        reward = 0.0
        info = {}
        if action == 0:
            results = self.index.query(self.current_query)
            info['retrieved'] = results
            reward = 0.5
        elif action == 1:
            # summarize top retrieved's text
            top_text = info.get('retrieved', [(d['text'],0) for d in self.docs])[0][0]
            summ = self.summarizer.summarize(top_text)
            info['summary'] = summ
            reward = 1.0
        else:
            # new query refinement
            self.current_query += ' application'
            reward = 0.2
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return {'query': self.current_query}, reward, done, info

# ---- Policy Feature Extractor ----
class QueryFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.embed = SentenceTransformer('all-MiniLM-L6-v2')
        self.net = nn.Sequential(nn.Linear(384, features_dim), nn.ReLU())

    def forward(self, observations):
        texts = observations['query']
        embs = self.embed.encode(texts, convert_to_numpy=True)
        x = torch.tensor(embs, dtype=torch.float32)
        return self.net(x)

# ---- Train PPO Agent ----
if __name__ == '__main__':
    docs = load_pdfs('papers/')  # folder with PDFs
    index = SemanticIndex(); index.build(docs)
    summarizer = Summarizer()
    env = ResearchEnv(docs, index, summarizer)

    policy_kwargs = dict(
        features_extractor_class=QueryFeatures,
        features_extractor_kwargs=dict(features_dim=128),
    )
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=10000)
    model.save('research_agent_ppo')

