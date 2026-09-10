import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.training.ordinal_loss import OrdinalCrossEntropy, compute_class_weights
from src.training.assessor import build_assessor_dataset, LABEL_MAP
import pandas as pd

base = "models/_merged_domain_lm"
tok = AutoTokenizer.from_pretrained(base)
if tok.pad_token is None: tok.pad_token = tok.eos_token

m = AutoModelForSequenceClassification.from_pretrained(
    base, num_labels=3, dtype=torch.bfloat16, device_map="cuda")
m.config.pad_token_id = tok.pad_token_id
m.eval()

df = pd.read_json("models/BoN_round1/accepted.jsonl", lines=True).head(8)
texts = [f"{e}: {t}" for e, t in zip(df.element, df.completion)]
labels = torch.tensor([LABEL_MAP[int(s)] for s in df.target_score]).cuda()

enc = tok(texts, return_tensors="pt", truncation=True,
          max_length=1024, padding="max_length").to("cuda")
with torch.no_grad():
    logits = m(**enc).logits.float()

print("logits shape:", tuple(logits.shape))
print("logits min/max/mean/std: "
      f"{logits.min():.2f} {logits.max():.2f} {logits.mean():.2f} {logits.std():.2f}")
print("sample logits:\n", logits[:4])
print()
print("plain CE       :", F.cross_entropy(logits, labels).item())
cw = compute_class_weights([0,0,1,1,1,2,2,2])
print("ordinal loss   :", OrdinalCrossEntropy(class_weights=cw)(logits, labels).item())
print("ordinal no-wgt :", OrdinalCrossEntropy()(logits, labels).item())
print()
print("score head weight std:", m.score.weight.std().item())
print("score head weight max:", m.score.weight.abs().max().item())
