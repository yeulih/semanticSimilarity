import json
import argparse
from sentence_transformers import SentenceTransformer, util


REMOVE_FIELDS = {"doi", "url"}


def remove_fields_case_insensitive(obj):
    if isinstance(obj, dict):
        return {
            k: remove_fields_case_insensitive(v)
            for k, v in obj.items()
            if k.lower() not in REMOVE_FIELDS
        }

    if isinstance(obj, list):
        return [remove_fields_case_insensitive(x) for x in obj]

    return obj


def flatten_to_text(obj):
    if isinstance(obj, dict):
        parts = []

        for key, value in obj.items():
            if value is not None:
                parts.append(f"{key}: {flatten_to_text(value)}")

        return " ".join(parts)

    if isinstance(obj, list):
        return " ".join(flatten_to_text(x) for x in obj)

    return str(obj)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_records(data):
    if isinstance(data, list):
        return data

    return [data]


def main():
    parser = argparse.ArgumentParser(
        description="Compute unordered semantic similarity between two JSON sets."
    )

    parser.add_argument("file1")
    parser.add_argument("file2")

    parser.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name or local path",
    )

    args = parser.parse_args()

    model = SentenceTransformer(args.model)

    data1 = remove_fields_case_insensitive(load_json(args.file1))
    data2 = remove_fields_case_insensitive(load_json(args.file2))

    records1 = normalize_records(data1)
    records2 = normalize_records(data2)

    texts1 = [flatten_to_text(r) for r in records1]
    texts2 = [flatten_to_text(r) for r in records2]

    embeddings1 = model.encode(
        texts1,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    embeddings2 = model.encode(
        texts2,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarity_matrix = util.cos_sim(embeddings1, embeddings2)

    # Best match from A -> B
    best_a_to_b = similarity_matrix.max(dim=1).values

    # Best match from B -> A
    best_b_to_a = similarity_matrix.max(dim=0).values

    # Symmetric unordered set similarity
    overall_similarity = (
        best_a_to_b.mean() + best_b_to_a.mean()
    ) / 2

    print(
        f"\nSemantic similarity between JSONs: "
        f"{overall_similarity.item():.4f}"
    )


if __name__ == "__main__":
    main()
