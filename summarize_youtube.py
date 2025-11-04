import os
import yt_dlp
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Step 1: Ask for a YouTube link ---
url = input("Paste a YouTube video URL: ").strip()

if not url:
    print("No URL entered. Exiting program.")
    raise SystemExit
if not url.startswith("http"):
    print("Please include http:// or https:// in the URL.")
    raise SystemExit

# --- Step 2: Download captions with yt-dlp ---
print("📥 Downloading captions...")

ydl_opts = {
    "skip_download": True,
    "writesubtitles": True,
    "writeautomaticsub": True,
    "subtitleslangs": ["en"],
    "subtitlesformat": "srt",
    "outtmpl": "%(id)s.%(ext)s",
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=True)
    video_id = info.get("id")
    srt_file = f"{video_id}.en.srt"

if not os.path.exists(srt_file):
    print("❌ No captions found for this video.")
    raise SystemExit

# --- Step 3: Clean captions ---
def clean_srt(srt_text: str) -> str:
    lines = []
    for line in srt_text.splitlines():
        if line.strip().isdigit():
            continue
        if "-->" in line:
            continue
        if line.strip():
            # filter out filler words
            if line.lower() in ["amen", "thanks guys", "thank you"]:
                continue
            lines.append(line.strip())
    return " ".join(lines)

with open(srt_file, "r", encoding="utf-8") as f:
    raw = f.read()
cleaned_text = clean_srt(raw)

print(f"🧹 Transcript cleaned. Length: {len(cleaned_text)} characters")

# --- Step 4: Chunking ---
def chunk_text(text, max_length=2500):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunks.append(text[start:end])
        start = end
    return chunks

chunks = chunk_text(cleaned_text)

# --- Step 5: Ollama API call ---
def ollama_generate(prompt, model="mistral"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt}
    )
    text = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")
            if '"response":"' in data:
                text += data.split('"response":"')[1].split('"')[0]
    return text.strip()

# --- Step 6: Summarization prompts ---
def summarize_batch(chunk_group):
    prompt = (
        "For EACH transcript section below, extract ONLY unique, actionable insights. "
        "Ignore filler, personal remarks, religious content, or thanks. "
        "Do NOT narrate events or repeat ideas. "
        "Each bullet must be under 12 words. "
        "Exclude references to paid tiers, subscriptions, or pricing.\n\n"
        + "\n\n---\n\n".join(chunk_group)
    )
    return ollama_generate(prompt)

def merge_summaries(batch_summaries):
    prompt = (
        "Here are partial insights from a video:\n\n"
        + "\n".join(batch_summaries)
        + "\n\n---\n\n"
        "TASK: Merge into concise, non-redundant bullet points. "
        "Output ONLY a flat bullet list. "
        "Remove duplicates, overlapping ideas, filler, irrelevant details, "
        "and any mention of paid tiers or subscriptions. "
        "Each bullet under 15 words."
    )
    return ollama_generate(prompt)

def final_summaries(merged_summary):
    quick_prompt = (
        "From the following insights, select the 5 MOST IMPORTANT lessons. "
        "Keep them concise, actionable, and non-redundant. "
        "Each bullet under 15 words. "
        "Do not repeat ideas. Exclude personal, religious, or subscription-related remarks. "
        "Output ONLY a flat bullet list:\n\n"
        + merged_summary
    )
    extended_prompt = (
        "From the following insights, produce EXACTLY 12 concise, actionable lessons. "
        "Each bullet under 15 words. "
        "Do not repeat ideas. Exclude personal, religious, or subscription-related remarks. "
        "Output ONLY a flat bullet list:\n\n"
        + merged_summary
    )
    return ollama_generate(quick_prompt), ollama_generate(extended_prompt)

# --- Step 7: Multi-worker execution with progress bar ---
def run_summarizer(chunks, batch_size=3, workers=3):
    batch_groups = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
    batch_summaries = []

    print(f"⚡ Summarizing {len(batch_groups)} batches with {workers} workers...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(summarize_batch, group): idx for idx, group in enumerate(batch_groups)}

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing batches")):
            result = future.result()
            batch_summaries.append(result)
            print(f"✅ Batch {i+1}/{len(futures)} complete")

    merged = merge_summaries(batch_summaries)
    quick, extended = final_summaries(merged)
    return quick, extended

# --- Step 8: Run everything ---
try:
    quick_summary, extended_summary = run_summarizer(chunks)

    print("\n✅ --- Quick Digest (5 bullets) ---\n")
    print(quick_summary)

    print("\n✅ --- Extended Digest (12 bullets) ---\n")
    print(extended_summary)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")

