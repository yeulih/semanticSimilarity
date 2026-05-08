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


def get_title(record, index):
    if isinstance(record, dict):
        return record.get("title") or record.get("Title") or f"Record {index + 1}"
    return f"Record {index + 1}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare two JSON files for unordered semantic similarity."
    )

    parser.add_argument("file1", help="First JSON file")
    parser.add_argument("file2", help="Second JSON file")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name or local model path",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of best matches to print",
    )

    args = parser.parse_args()

    model = SentenceTransformer(args.model)

    data1 = remove_fields_case_insensitive(load_json(args.file1))
    data2 = remove_fields_case_insensitive(load_json(args.file2))

    records1 = normalize_records(data1)
    records2 = normalize_records(data2)

    texts1 = [flatten_to_text(record) for record in records1]
    texts2 = [flatten_to_text(record) for record in records2]

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

    best_1_to_2 = similarity_matrix.max(dim=1).values
    best_2_to_1 = similarity_matrix.max(dim=0).values

    overall_score = ((best_1_to_2.mean() + best_2_to_1.mean()) / 2).item()

    print(f"\nOverall unordered semantic similarity: {overall_score:.4f}")

    print(f"\nTop {args.top} best matches from file 1 to file 2:")

    match_rows = []

    for i in range(len(records1)):
        best_j = similarity_matrix[i].argmax().item()
        score = similarity_matrix[i][best_j].item()

        match_rows.append((score, i, best_j))

    match_rows.sort(reverse=True, key=lambda x: x[0])

    for score, i, j in match_rows[: args.top]:
        title1 = get_title(records1[i], i)
        title2 = get_title(records2[j], j)

        print("\n---")
        print(f"File 1 item: {title1}")
        print(f"Best file 2 match: {title2}")
        print(f"Similarity between JSONs: {score:.4f}")


if __name__ == "__main__":
    main()
