import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rewards.authenticity_reward import AuthenticityReward

df = pd.read_json('models/BoN_round1/accepted.jsonl', lines=True)
auth = AuthenticityReward(device='cuda')
scores = auth.score_batch(df.completion.tolist(), batch_size=8, normalize=False)
df['auth_reward_abs'] = scores
df.to_json('models/BoN_round1/accepted.jsonl', orient='records', lines=True)
print(f"round1 mean_auth_abs = {sum(scores)/len(scores):.4f}")
