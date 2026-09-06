import time
import re
import runpod
from vllm import LLM, SamplingParams

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

MODEL_PATH = "/models/hf/translategemma"
llm_engine = None
tokenizer = None

# =====================================================
# Language name → ISO 639-1 code mapping
# =====================================================
LANG_CODE_MAP = {
    # English
    "english": "en",
    # Spanish
    "spanish": "es", "español": "es", "espanol": "es",
    # French
    "french": "fr", "français": "fr", "francais": "fr",
    # German
    "german": "de", "deutsch": "de",
    # Arabic
    "arabic": "ar", "العربية": "ar", "عربي": "ar",
    # Russian
    "russian": "ru", "русский": "ru",
    # Chinese
    "chinese": "zh", "中文": "zh",
    # Portuguese
    "portuguese": "pt", "português": "pt",
    # Italian
    "italian": "it", "italiano": "it",
    # Turkish
    "turkish": "tr", "türkçe": "tr",
    # Japanese
    "japanese": "ja", "日本語": "ja",
    # Korean
    "korean": "ko", "한국어": "ko",
    # Dutch
    "dutch": "nl", "nederlands": "nl",
    # Polish
    "polish": "pl", "polski": "pl",
    # Hindi
    "hindi": "hi", "हिन्दी": "hi",
    # Greek
    "greek": "el", "ελληνικά": "el",
    # Czech
    "czech": "cs", "čeština": "cs",
    # Romanian
    "romanian": "ro", "română": "ro",
    # Hungarian
    "hungarian": "hu", "magyar": "hu",
    # Swedish
    "swedish": "sv", "svenska": "sv",
    # Bulgarian
    "bulgarian": "bg", "български": "bg",
    # Ukrainian
    "ukrainian": "uk", "українська": "uk",
    # Hebrew
    "hebrew": "he", "עברית": "he",
    # Thai
    "thai": "th", "ไทย": "th",
    # Vietnamese
    "vietnamese": "vi", "tiếng việt": "vi",
    # Indonesian
    "indonesian": "id", "bahasa indonesia": "id",
    # Malay
    "malay": "ms", "bahasa melayu": "ms",
    # Persian / Farsi
    "persian": "fa", "farsi": "fa", "فارسی": "fa",
}

def get_lang_code(language_name: str) -> str:
    """Convert a language name to its ISO 639-1 code."""
    normalized = language_name.strip().lower()
    # If it's already a 2-letter code, return it
    if len(normalized) == 2 and normalized.isascii():
        return normalized
    # If it's a known language name, map it
    if normalized in LANG_CODE_MAP:
        return LANG_CODE_MAP[normalized]
    # Fallback: return as-is (TranslateGemma may still recognize it)
    log(f"WARNING: Unknown language '{language_name}', passing as-is")
    return normalized


# =====================================================
# Summary prompt (kept for summarization)
# =====================================================
DEFAULT_SUMMARY_PROMPT = (
    "You are a professional legal assistant.\n"
    "Produce a single-paragraph summary of the ENTIRE document in clear English.\n"
    "STRICT RULES:\n"
    "- Output MUST be one paragraph only\n"
    "- Do NOT use headings, titles, bullet points, or lists\n"
    "- Do NOT classify the document type unless explicitly stated in the text\n"
    "- Do NOT invent or infer information\n"
    "- Mention only facts that are explicitly present in the document\n"
    "- Cover all major sections evenly if the document is long\n"
    "- Focus on parties, purpose, key obligations, payments, terms, penalties, and dispute resolution if present\n"
    "- Ignore layout, tables, formatting, and section numbering\n"
    "- Write in neutral legal English\n\n"
)


# =====================================================
# Load model with vLLM — auto-detects GPU
# =====================================================
def load_model():
    global llm_engine, tokenizer
    if llm_engine is not None:
        return

    # IMPORTANT: Do NOT call torch.cuda.* before vLLM init!
    # It initializes CUDA which forces 'spawn' multiprocessing and crashes.
    log("Loading TranslateGemma-4b-it with vLLM engine...")
    t0 = time.time()

    llm_engine = LLM(
        model=MODEL_PATH,
        dtype="auto",                    # auto-selects BF16 on Ampere+
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )

    tokenizer = llm_engine.get_tokenizer()

    # Log GPU info AFTER vLLM has initialized CUDA
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")

    log(f"vLLM engine ready in {time.time()-t0:.1f}s")


# =====================================================
# Helper: build prompt from messages
# =====================================================
def build_prompt(messages):
    """Build prompt using tokenizer's chat template."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        log(f"WARNING: Chat template failed ({e}), using fallback format")
        # Fallback for TranslateGemma: simple turn-based format
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if isinstance(content, list):
                # Structured content for translation
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_part = item.get("text", "")
                        src = item.get("source_lang_code", "")
                        tgt = item.get("target_lang_code", "")
                        parts.append(f"<start_of_turn>{role}\n"
                                     f"Translate from {src} to {tgt}:\n{text_part}"
                                     f"<end_of_turn>")
            else:
                parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>")
        parts.append("<start_of_turn>model\n")
        return "\n".join(parts)


# =====================================================
# Helper: build translation messages for TranslateGemma
# =====================================================
def build_translation_messages(text: str, target_lang_code: str) -> list:
    """Build TranslateGemma-style structured messages for translation."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "auto",
                    "target_lang_code": target_lang_code,
                    "text": text
                }
            ]
        }
    ]


# =====================================================
# Text helpers
# =====================================================

def limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    if truncated.rstrip().endswith("."):
        return truncated.rstrip()
    last_period = max(truncated.rfind(". "), truncated.rfind(".\n"))
    last_excl = truncated.rfind("! ")
    last_quest = truncated.rfind("? ")
    best = max(last_period, last_excl, last_quest)
    if best > len(truncated) * 0.6:
        return truncated[:best + 1].strip()
    return truncated.rstrip()

def clean_output(decoded: str) -> str:
    decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
    decoded = re.sub(r"<\|.*?\|>", "", decoded).strip()
    # Clean up any turn markers that might leak through
    decoded = re.sub(r"<start_of_turn>.*?<end_of_turn>", "", decoded, flags=re.DOTALL).strip()
    decoded = re.sub(r"<end_of_turn>", "", decoded).strip()
    return decoded


# =====================================================
# Token limits and helpers for context-aware chunking
# =====================================================
MAX_PROMPT_TOKENS = 6000  # Conservative limit for 4B model with 8192 context

def _estimate_tokens(text):
    """Rough token estimate: ~1 token per 3.5 characters for mixed content."""
    return len(text) // 3


# =====================================================
# TRANSLATION — TranslateGemma structured format
# =====================================================
def translate_text_batch(texts, target_language="English"):
    target_code = get_lang_code(target_language)
    log(f"Target language: {target_language} → ISO code: {target_code}")

    prompts = []
    valid_indices = []
    results = [""] * len(texts)

    for idx, text in enumerate(texts):
        stripped = (text or "").strip()
        if not stripped or len(re.findall(r"[^\W\d_]", stripped, re.UNICODE)) < 5:
            results[idx] = text or ""
            continue

        messages = build_translation_messages(stripped, target_code)
        prompt = build_prompt(messages)

        # Safety check: if a single page exceeds context, split it
        if _estimate_tokens(prompt) > MAX_PROMPT_TOKENS:
            words = stripped.split()
            mid = len(words) // 2
            for half in [" ".join(words[:mid]), " ".join(words[mid:])]:
                half_msgs = build_translation_messages(half, target_code)
                prompts.append(build_prompt(half_msgs))
                valid_indices.append(idx)  # both halves map to same index
        else:
            prompts.append(prompt)
            valid_indices.append(idx)

    if not prompts:
        return results

    log(f"Translating {len(prompts)} pages in parallel with vLLM...")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=4096,
    )

    t0 = time.time()
    outputs = llm_engine.generate(prompts, sampling_params)
    gen_time = time.time() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    log(f"Translation: {total_tokens} tokens in {gen_time:.1f}s "
        f"({total_tokens/gen_time:.1f} tok/s effective)")

    for i, output in enumerate(outputs):
        translated = clean_output(output.outputs[0].text)
        idx = valid_indices[i]
        if results[idx]:
            results[idx] += "\n" + translated  # Concatenate split-page halves
        else:
            results[idx] = translated

    return results


# =====================================================
# SUMMARY — chunked to fit context window
# =====================================================

def _build_summary_prompt(text_block, target_words, system_prompt):
    """Build a summary prompt from a text block and return the formatted string."""
    user_content = (
        f"{system_prompt}\n\n"
        f"Summarize the following document in approximately {target_words} words. "
        f"Make sure to complete all sentences properly.\n\n"
        f"DOCUMENT:\n{text_block}"
    )
    # TranslateGemma uses user/model roles (no system role)
    messages = [
        {"role": "user", "content": user_content}
    ]
    return build_prompt(messages)

def _chunk_pages_by_tokens(cleaned_pages, max_tokens):
    """Split cleaned page texts into chunks that fit within max_tokens."""
    chunks = []
    current_chunk = []
    current_tokens = 0

    for page_text in cleaned_pages:
        page_tokens = _estimate_tokens(page_text)
        # If a single page exceeds the limit, truncate it
        if page_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            # Truncate to fit
            char_limit = max_tokens * 3
            chunks.append(page_text[:char_limit])
            continue

        if current_tokens + page_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(page_text)
        current_tokens += page_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

def summarize_all_pages(pages, max_words, system_prompt):
    # Collect all page texts
    cleaned_pages = []
    for p in pages:
        text = (p["text"] or "").strip()
        if text and len(re.findall(r"[^\W\d_]", text, re.UNICODE)) > 20:
            cleaned_pages.append(text)

    if not cleaned_pages:
        log("ERROR: No valid text found for summary")
        return ""

    full_text = "\n\n".join(cleaned_pages)
    doc_word_count = len(full_text.split())
    actual_target = max(50, min(max_words, doc_word_count // 3))
    log(f"Summary target: {actual_target} words (doc has {doc_word_count} words)")

    # Check if the full text fits in one prompt
    test_prompt = _build_summary_prompt(full_text, actual_target, system_prompt)
    prompt_tokens = _estimate_tokens(test_prompt)

    if prompt_tokens <= MAX_PROMPT_TOKENS:
        # Single-shot: fits in context
        log("Summary: single-shot (fits in context)")
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=min(actual_target * 5, 4096),
        )
        t0 = time.time()
        outputs = llm_engine.generate([test_prompt], sampling_params)
        gen_time = time.time() - t0
        decoded = clean_output(outputs[0].outputs[0].text)
        result = limit_words(decoded, actual_target)
        log(f"Summary: {len(result.split())} words in {gen_time:.1f}s")
        return result

    # Chunked summarization: split pages into token-safe groups
    chunks = _chunk_pages_by_tokens(cleaned_pages, MAX_PROMPT_TOKENS)
    log(f"Summary: document too large, splitting into {len(chunks)} chunks")

    # Phase 1: Summarize each chunk
    words_per_chunk = max(100, actual_target // len(chunks) + 50)
    chunk_prompts = []
    for i, chunk_text in enumerate(chunks):
        chunk_prompts.append(
            _build_summary_prompt(chunk_text, words_per_chunk, system_prompt)
        )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=min(words_per_chunk * 5, 4096),
    )

    t0 = time.time()
    chunk_outputs = llm_engine.generate(chunk_prompts, sampling_params)
    phase1_time = time.time() - t0
    log(f"Summary phase 1: {len(chunks)} chunks summarized in {phase1_time:.1f}s")

    chunk_summaries = []
    for output in chunk_outputs:
        chunk_summaries.append(clean_output(output.outputs[0].text))

    # Phase 2: Combine chunk summaries into final summary
    combined = "\n\n".join(
        f"[Part {i+1}]: {s}" for i, s in enumerate(chunk_summaries)
    )

    combine_user = (
        f"{system_prompt}\n\n"
        f"Below are summaries of different sections of a single document. "
        f"Combine them into ONE coherent summary of approximately {actual_target} words. "
        f"Make sure to complete all sentences properly. "
        f"Do NOT list the parts separately — write a single unified paragraph.\n\n"
        f"{combined}"
    )
    combine_messages = [
        {"role": "user", "content": combine_user}
    ]
    combine_prompt = build_prompt(combine_messages)

    sampling_params_final = SamplingParams(
        temperature=0,
        max_tokens=min(actual_target * 5, 4096),
    )

    t0 = time.time()
    final_outputs = llm_engine.generate([combine_prompt], sampling_params_final)
    phase2_time = time.time() - t0

    decoded = clean_output(final_outputs[0].outputs[0].text)
    result = limit_words(decoded, actual_target)

    log(f"Summary: {len(result.split())} words in {phase1_time + phase2_time:.1f}s total "
        f"(phase1={phase1_time:.1f}s, phase2={phase2_time:.1f}s)")
    return result


# =====================================================
# RunPod handler
# =====================================================
def handler(event):
    log("Handler started")

    input_data = event["input"]
    pages = input_data["pages"]
    max_words = int(input_data.get("n_words", 500))
    system_prompt = input_data.get("system_prompt", DEFAULT_SUMMARY_PROMPT)
    target_language = input_data.get("target_language", "English")

    log(f"Processing {len(pages)} pages, target: {max_words} words, translate to: {target_language}")

    load_model()

    # 1) Translate all pages in parallel
    log(f"Starting batch translation to {target_language}...")
    start = time.time()
    page_texts = [p["text"] for p in pages]
    translated_texts = translate_text_batch(page_texts, target_language)
    for i, p in enumerate(pages):
        p["text"] = translated_texts[i]
    log(f"Translation done in {time.time()-start:.2f}s")

    # 2) Summarize
    log(f"Creating summary ({max_words} words)")
    start = time.time()
    summary = summarize_all_pages(pages, max_words, system_prompt)
    log(f"Summary done in {time.time()-start:.2f}s")

    if not summary:
        log("WARNING: Summary is empty!")

    log("Handler finished")
    return {"summary": summary, "pages": pages}

runpod.serverless.start({"handler": handler})