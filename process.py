import os
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = os.getenv("RAW_DIR")
PROCESSED_DIR = os.getenv("PROCESSED_DIR")
MODEL_PATH = os.getenv("MODEL_PATH")

# ===============================
# CONFIG
# ===============================

CTX_SIZE = 2048
MAX_MICRO_TOKENS = 300
MAX_FINAL_TOKENS = 1500
MAX_CHUNK_CHARS = 1000
MAX_NOTES_CHARS = 3000

# ===============================
# INITIALIZE MODEL
# ===============================
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_SIZE,
    n_threads=8,
    verbose=False
)

# ===============================
# PROMPTS
# ===============================
def micro_summary_prompt(text):
    return f"""
Summarize the text below into concise study notes.
Keep only definitions, key ideas, and examples.
No fluff.

Text:
{text}

Notes:
""".strip()

def final_summary_prompt(notes, topic, subtopic, source):
    return f"""
You are an expert trading educator.

Using the summarized notes below, create a FINAL structured lesson
for an educational app frontend.

Requirements:
1. title
2. summary: concise explanation (7-10 sentences)
3. key_points (3–7 bullets)
4. examples (1–3)
5. definitions (simple explanations, can include analogies)
6. common_mistakes (1–2)
7. questions_to_think (1–2)
8. source: cite the original source of the content.

Notes:
{notes}

Topic: {topic}
Subtopic: {subtopic}
Source: {source}

Output YAML ONLY.
""".strip()

# ===============================
# HELPERS
# ===============================
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def save_yaml(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip())

def safe_truncate(text, max_chars):
    return text[:max_chars]

def generate(prompt, max_tokens):
    response = llm(prompt, max_tokens=max_tokens)
    text = response["choices"][0]["text"].strip()
    if not text:
        print("Warning: LLM returned empty output!")
    return text

# ===============================
# PIPELINE: ONE-TO-ONE CHUNKS
# ===============================
for main_topic in os.listdir(RAW_DIR):
    main_path = os.path.join(RAW_DIR, main_topic)
    if not os.path.isdir(main_path):
        continue

    for subtopic in os.listdir(main_path):
        sub_path = os.path.join(main_path, subtopic)
        if not os.path.isdir(sub_path):
            continue

        print(f"\nProcessing: {main_topic} → {subtopic}")

        for file in sorted(os.listdir(sub_path)):
            if not file.endswith(".txt"):
                continue

            raw_path = os.path.join(sub_path, file)
            raw_text = read_file(raw_path)
            if not raw_text:
                print(f"Skipping empty file: {file}")
                continue

            raw_text = safe_truncate(raw_text, MAX_CHUNK_CHARS)
            print(f"Processing file: {raw_path} (length {len(raw_text)})")

            # Micro summary
            micro = generate(micro_summary_prompt(raw_text), MAX_MICRO_TOKENS)
            if not micro:
                print(f"Skipping {file} — empty micro summary")
                continue
            #print(f"Micro summary preview: {micro[:100]}...")

            micro = safe_truncate(micro, MAX_NOTES_CHARS)

            # Final YAML
            final_yaml = generate(final_summary_prompt(micro, main_topic, subtopic, "Various sources"), MAX_FINAL_TOKENS)
            if not final_yaml:
                print(f"Skipping {file} — empty final YAML")
                continue

            out_path = os.path.join(PROCESSED_DIR, main_topic, subtopic, f"{os.path.splitext(file)[0]}_summary.yaml")
            save_yaml(final_yaml, out_path)
            print(f"✅ Saved summary for {file} → {out_path} (length {len(final_yaml)} chars)")

print("\n🎉 Completed all processing.")
