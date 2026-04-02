import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = Path("data/raw/it_simplification.csv")
GLOSSARY_PATH = Path("data/raw/glossary.csv")
OUTPUT_DIR = Path("data/processed")

BASE_SYSTEM_PROMPT = (
    "You are an educational assistant specialized in simplifying IT and computer science "
    "content into clear Modern Standard Arabic for Moroccan learners. "
    "Preserve the exact technical meaning. "
    "Use natural, grammatically correct Arabic. "
    "Avoid mixing Arabic with French, English, or other languages unless the technical term must be kept. "
    "When technical terminology is needed, prefer the provided Arabic glossary terms. "
    "Keep the explanation short, simple, and pedagogically clear."
)


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def singularize_french_token(token: str) -> str:
    token = token.strip().lower()
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def clean_mixed_text(text: str) -> str:
    """
    Very light cleanup for obviously mixed or noisy input.
    Keeps the sentence mostly intact, without aggressive rewriting.
    """
    text = str(text).strip()

    replacements = {
        " عبر ": " à travers ",
        " عدة ": " plusieurs ",
        " خوادم": " serveurs",
        " المستخدم ": " utilisateur ",
        " البيانات ": " données ",
        " الإنترنت": " Internet",
        " الشبكة": " réseau",
        " الخادم": " serveur",
        " المتصفح": " navigateur",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_glossary(glossary_path: Path) -> list[dict]:
    if not glossary_path.exists():
        print(f"Glossary file not found: {glossary_path}")
        return []

    df = pd.read_csv(glossary_path).fillna("")

    required_cols = {"term_fr", "term_ar"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Glossary file is missing columns: {missing}")

    entries = []
    for _, row in df.iterrows():
        term_fr = str(row["term_fr"]).strip()
        term_ar = str(row["term_ar"]).strip()

        if term_fr and term_ar:
            entries.append(
                {
                    "term_fr": term_fr,
                    "term_ar": term_ar,
                    "term_fr_norm": normalize_text(term_fr),
                }
            )

    return entries


def extract_glossary_for_row(
    original_text: str,
    keywords: str,
    glossary_entries: list[dict],
    max_terms: int = 12,
) -> list[dict]:
    original_norm = normalize_text(original_text)
    keywords_norm = normalize_text(str(keywords).replace("|", " "))

    original_tokens = set(original_norm.split())
    original_tokens_singular = {singularize_french_token(t) for t in original_tokens}

    keywords_tokens = set(keywords_norm.split())
    keywords_tokens_singular = {singularize_french_token(t) for t in keywords_tokens}

    matched = []
    seen = set()

    for entry in glossary_entries:
        fr_raw = entry["term_fr"]
        fr_norm = entry["term_fr_norm"]
        fr_tokens = fr_norm.split()

        phrase_match = fr_norm in original_norm or fr_norm in keywords_norm

        token_match = all(
            (
                t in original_tokens
                or t in keywords_tokens
                or singularize_french_token(t) in original_tokens_singular
                or singularize_french_token(t) in keywords_tokens_singular
            )
            for t in fr_tokens
        )

        single_token_match = False
        if len(fr_tokens) == 1:
            t = fr_tokens[0]
            ts = singularize_french_token(t)
            single_token_match = (
                t in original_tokens
                or t in keywords_tokens
                or ts in original_tokens_singular
                or ts in keywords_tokens_singular
            )

        if phrase_match or token_match or single_token_match:
            key = (entry["term_fr"], entry["term_ar"])
            if key not in seen:
                matched.append(entry)
                seen.add(key)

        if len(matched) >= max_terms:
            break

    return matched


def build_glossary_block(matched_terms: list[dict]) -> str:
    if not matched_terms:
        return ""

    lines = ["Use the following Arabic technical terminology when relevant:"]
    for item in matched_terms:
        lines.append(f"- {item['term_fr']} = {item['term_ar']}")
    return "\n".join(lines)


def build_user_prompt(original_text: str, glossary_block: str) -> str:
    prompt = (
        "Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. "
        "Preserve the meaning and avoid mixing languages.\n\n"
    )

    if glossary_block:
        prompt += glossary_block + "\n\n"

    prompt += f"Text:\n{original_text}"
    return prompt


def make_sample(row, glossary_entries: list[dict]) -> dict:
    cleaned_original = clean_mixed_text(row["original_text"])

    matched_terms = extract_glossary_for_row(
        original_text=cleaned_original,
        keywords=row.get("keywords", ""),
        glossary_entries=glossary_entries,
    )

    glossary_block = build_glossary_block(matched_terms)
    user_prompt = build_user_prompt(cleaned_original, glossary_block)

    return {
        "messages": [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": row["simplified_arabic"]},
        ]
    }


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_train_preview(path: Path, train_df: pd.DataFrame, glossary_entries: list[dict]) -> None:
    preview_rows = []

    for _, row in train_df.iterrows():
        cleaned_original = clean_mixed_text(row["original_text"])
        matched_terms = extract_glossary_for_row(
            original_text=cleaned_original,
            keywords=row.get("keywords", ""),
            glossary_entries=glossary_entries,
        )
        glossary_block = build_glossary_block(matched_terms)
        user_prompt = build_user_prompt(cleaned_original, glossary_block)

        preview_rows.append(
            {
                "id": row["id"],
                "topic": row["topic"],
                "source_lang": row["source_lang"],
                "original_text": row["original_text"],
                "cleaned_original_text": cleaned_original,
                "matched_terms": " | ".join(
                    [f"{m['term_fr']}={m['term_ar']}" for m in matched_terms]
                ),
                "user_prompt": user_prompt,
                "assistant_target": row["simplified_arabic"],
            }
        )

    preview_df = pd.DataFrame(preview_rows)
    preview_df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH).fillna("")

    required_cols = {
        "id",
        "topic",
        "source_lang",
        "original_text",
        "simplified_arabic",
        "keywords",
        "difficulty",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset file is missing columns: {missing}")

    glossary_entries = load_glossary(GLOSSARY_PATH)
    print(f"Loaded glossary entries: {len(glossary_entries)}")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["topic"],
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        shuffle=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_samples = [make_sample(r, glossary_entries) for _, r in train_df.iterrows()]
    valid_samples = [make_sample(r, glossary_entries) for _, r in valid_df.iterrows()]
    test_samples = [make_sample(r, glossary_entries) for _, r in test_df.iterrows()]

    save_jsonl(OUTPUT_DIR / "train.jsonl", train_samples)
    save_jsonl(OUTPUT_DIR / "valid.jsonl", valid_samples)
    save_jsonl(OUTPUT_DIR / "test.jsonl", test_samples)

    save_train_preview(OUTPUT_DIR / "train_preview.csv", train_df, glossary_entries)

    print("Dataset prepared successfully")
    print("Train:", len(train_samples))
    print("Valid:", len(valid_samples))
    print("Test:", len(test_samples))

    print("\nTrain topic distribution:")
    print(train_df["topic"].value_counts().to_dict())

    print("\nValid topic distribution:")
    print(valid_df["topic"].value_counts().to_dict())

    print("\nTest topic distribution:")
    print(test_df["topic"].value_counts().to_dict())

    print("\nGlossary debug preview:\n")
    for _, r in train_df.head(10).iterrows():
        cleaned_original = clean_mixed_text(r["original_text"])
        matched = extract_glossary_for_row(
            original_text=cleaned_original,
            keywords=r.get("keywords", ""),
            glossary_entries=glossary_entries,
        )
        print("TEXT:", cleaned_original)
        print("MATCHED:", [(m["term_fr"], m["term_ar"]) for m in matched])
        print("-" * 80)

    if train_samples:
        print("\nSample training prompt preview:\n")
        print(json.dumps(train_samples[0], ensure_ascii=False, indent=2))

    print("\nSaved preview file:", OUTPUT_DIR / "train_preview.csv")


if __name__ == "__main__":
    main()