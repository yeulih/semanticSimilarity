from transformers import AutoTokenizer, AutoModel
import torch

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").cuda()

inputs = tok("Hello from BERT on GPU!", return_tensors="pt").to("cuda")
outputs = model(**inputs)

print("BERT ran successfully on:", outputs.last_hidden_state.device)
