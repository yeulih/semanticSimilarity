import torch
from transformers import AutoTokenizer, AutoModel
from pypdf import PdfReader
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Extract text from a PDF
# ---------------------------------------------------------
def extract_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# ---------------------------------------------------------
# 2. Chunk text into ~350-token segments for BERT
# ---------------------------------------------------------
def chunk_text(text, tokenizer, max_tokens=350):
    words = text.split()
    chunks = []
    current = []

    for word in words:
        current.append(word)
        tokens = tokenizer(" ".join(current), return_tensors="pt", truncation=True)
        if tokens.input_ids.shape[1] >= max_tokens:
            chunks.append(" ".join(current[:-1]))
            current = [word]

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------
# 3. Embed a chunk using BERT (CLS embedding)
# ---------------------------------------------------------
def embed_chunk(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu()  # CLS token


# ---------------------------------------------------------
# 4. Build a document-level embedding by averaging chunk embeddings
# ---------------------------------------------------------
def embed_document(pdf_path, tokenizer, model):
    print(f"\nProcessing: {pdf_path}")

    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text, tokenizer)

    print(f" - Extracted {len(chunks)} chunks")

    embeddings = []
    for chunk in chunks:
        emb = embed_chunk(chunk, tokenizer, model)
        embeddings.append(emb)

    doc_embedding = torch.stack(embeddings).mean(dim=0)
    print(" - Document embedding shape:", doc_embedding.shape)

    return doc_embedding


# ---------------------------------------------------------
# 5. Compute cosine similarity between two document embeddings
# ---------------------------------------------------------
def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load BERT on GPU
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").cuda()

    # Paths to your PDFs
    pdf1 = "paper1.pdf"
    pdf2 = "paper2.pdf"

    # Generate document embeddings
    emb1 = embed_document(pdf1, tokenizer, model)
    emb2 = embed_document(pdf2, tokenizer, model)

    # Compute similarity
    sim = cosine_similarity(emb1, emb2)

    print("\n====================================")
    print(f"Similarity between PDFs: {sim:.4f}")
    print("====================================")
