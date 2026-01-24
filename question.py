import os
import yaml
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")

CTX_SIZE = 4096
MAX_TOKENS = 800

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_SIZE,
    n_threads=12,
    n_gpu_layers=35,
    verbose=False,
)


def build_context(data: dict) -> str:
    title = data.get("title", "")
    summary = data.get("summary", "")
    key_points = data.get("key_points", []) or []
    questions = data.get("questions_to_think", []) or []

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary: {summary}")
    if key_points:
        parts.append("Key points:")
        parts.extend([f"- {p}" for p in key_points])
    if questions:
        parts.append("Questions to think:")
        parts.extend([f"- {q}" for q in questions])
    return "\n".join(parts).strip()


def question_prompt(context: str, level: str) -> str:
    return f"""You are creating Duolingo-style practice questions for a {level} learner.
Generate exactly ONE question for EACH type below, with answers.

Types required (one each):
1) fill_blank_mcq
2) drag_drop
3) case_study_mcq
4) true_false

RULES:
- Output valid YAML only (no extra text).
- Each question must be a single sentence.
- Keep language beginner-friendly.
- For fill_blank_mcq, include "___" in the question and 4 options.
- For drag_drop, include 3 pairs.
- For case_study_mcq, include a short scenario AND a question, with 4 options.
- For true_false, answer must be "true" or "false".

YAML format:
questions:
  - type: fill_blank_mcq
    question: <string with ___>
    options: [A, B, C, D]
    answer: <A/B/C/D>
  - type: drag_drop
    question: <string>
    drag_drop:
      prompt: <string>
      pairs:
        - left: <string>
          right: <string>
        - left: <string>
          right: <string>
        - left: <string>
          right: <string>
    answer: <string>
  - type: case_study_mcq
    scenario: <string>
    question: <string>
    options: [A, B, C, D]
    answer: <A/B/C/D>
  - type: true_false
    question: <string>
    answer: <true/false>

Context:
{context}
"""


def generate_questions(context: str, level: str) -> str:
    prompt = question_prompt(context, level)
    output = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )
    if isinstance(output, dict):
        return output.get("choices", [{}])[0].get("text", "").strip()
    return str(output).strip()


def output_path_for(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_questions_by_level{ext}"


def process_one_file(path: str, level: str):
    if not os.path.isfile(path):
        raise RuntimeError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    data = yaml.safe_load(content) or {}
    context = build_context(data)

    questions_yaml = generate_questions(context, level)
    return questions_yaml


def main():
    if not MODEL_PATH:
        raise RuntimeError("MODEL_PATH is not set in .env")

    target = "data/processed/How Stock Markets Work/algorithmic_trading,_high-frequency_trading_(HFT),_and_market_microstructure/chunks.yaml"
    sample_users = [
        {"id": "u001", "name": "Ava", "level": "beginner"},
        {"id": "u002", "name": "Ben", "level": "intermediate"},
        {"id": "u003", "name": "Cara", "level": "advanced"},
    ]
    results = {}
    for user in sample_users:
        results[user["level"]] = process_one_file(target, user["level"])

    out_path = output_path_for(target)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("by_level:\n")
        for level, yaml_text in results.items():
            f.write(f"  {level}: |-\n")
            for line in (yaml_text or "").splitlines():
                f.write(f"    {line}\n")

    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
