import time
import re
import torch
import runpod
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

MODEL_PATH = "/models/hf/nllb"
model = None
tokenizer = None

# =====================================================
# Language name / ISO 639-1 → FLORES-200 code mapping
# NLLB uses FLORES-200 codes: 3-letter ISO 639-3 + 4-letter script
# =====================================================
LANG_CODE_MAP = {
    # English
    "english": "eng_Latn", "en": "eng_Latn",
    # Spanish
    "spanish": "spa_Latn", "es": "spa_Latn",
    "español": "spa_Latn", "espanol": "spa_Latn",
    # French
    "french": "fra_Latn", "fr": "fra_Latn",
    "français": "fra_Latn", "francais": "fra_Latn",
    # German
    "german": "deu_Latn", "de": "deu_Latn", "deutsch": "deu_Latn",
    # Arabic (Modern Standard)
    "arabic": "arb_Arab", "ar": "arb_Arab",
    "العربية": "arb_Arab", "عربي": "arb_Arab",
    # Russian
    "russian": "rus_Cyrl", "ru": "rus_Cyrl", "русский": "rus_Cyrl",
    # Chinese (Simplified)
    "chinese": "zho_Hans", "zh": "zho_Hans", "中文": "zho_Hans",
    # Portuguese
    "portuguese": "por_Latn", "pt": "por_Latn", "português": "por_Latn",
    # Italian
    "italian": "ita_Latn", "it": "ita_Latn", "italiano": "ita_Latn",
    # Turkish
    "turkish": "tur_Latn", "tr": "tur_Latn", "türkçe": "tur_Latn",
    # Japanese
    "japanese": "jpn_Jpan", "ja": "jpn_Jpan", "日本語": "jpn_Jpan",
    # Korean
    "korean": "kor_Hang", "ko": "kor_Hang", "한국어": "kor_Hang",
    # Dutch
    "dutch": "nld_Latn", "nl": "nld_Latn", "nederlands": "nld_Latn",
    # Polish
    "polish": "pol_Latn", "pl": "pol_Latn", "polski": "pol_Latn",
    # Hindi
    "hindi": "hin_Deva", "hi": "hin_Deva", "हिन्दी": "hin_Deva",
    # Greek
    "greek": "ell_Grek", "el": "ell_Grek", "ελληνικά": "ell_Grek",
    # Czech
    "czech": "ces_Latn", "cs": "ces_Latn", "čeština": "ces_Latn",
    # Romanian
    "romanian": "ron_Latn", "ro": "ron_Latn", "română": "ron_Latn",
    # Hungarian
    "hungarian": "hun_Latn", "hu": "hun_Latn", "magyar": "hun_Latn",
    # Swedish
    "swedish": "swe_Latn", "sv": "swe_Latn", "svenska": "swe_Latn",
    # Bulgarian
    "bulgarian": "bul_Cyrl", "bg": "bul_Cyrl", "български": "bul_Cyrl",
    # Ukrainian
    "ukrainian": "ukr_Cyrl", "uk": "ukr_Cyrl", "українська": "ukr_Cyrl",
    # Hebrew
    "hebrew": "heb_Hebr", "he": "heb_Hebr", "עברית": "heb_Hebr",
    # Thai
    "thai": "tha_Thai", "th": "tha_Thai", "ไทย": "tha_Thai",
    # Vietnamese
    "vietnamese": "vie_Latn", "vi": "vie_Latn", "tiếng việt": "vie_Latn",
    # Indonesian
    "indonesian": "ind_Latn", "id": "ind_Latn", "bahasa indonesia": "ind_Latn",
    # Malay
    "malay": "zsm_Latn", "ms": "zsm_Latn", "bahasa melayu": "zsm_Latn",
    # Persian / Farsi
    "persian": "pes_Arab", "farsi": "pes_Arab", "fa": "pes_Arab",
    "فارسی": "pes_Arab",
}

def get_lang_code(language_name: str) -> str:
    """Convert a language name or ISO 639-1 code to its FLORES-200 code."""
    normalized = language_name.strip().lower()
    if normalized in LANG_CODE_MAP:
        return LANG_CODE_MAP[normalized]
    # If already a FLORES-200 code (e.g., "eng_Latn"), return as-is
    if re.match(r"^[a-z]{3}_[A-Z][a-z]{3}$", language_name.strip()):
        return language_name.strip()
    log(f"WARNING: Unknown language '{language_name}', passing as-is")
    return normalized


# =====================================================
# Load model with HuggingFace Transformers
# =====================================================
BATCH_SIZE = 16       # Sentences per batch for GPU inference
MAX_LENGTH = 512      # Max tokens per segment (NLLB distilled context)

def load_model():
    global model, tokenizer
    if model is not None:
        return

    log("Loading NLLB-200-distilled-1.3B...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")

    log(f"NLLB model ready in {time.time()-t0:.1f}s on {device}")


# =====================================================
# Text chunking — split long text into translatable segments
# =====================================================
def _split_into_segments(text: str, max_chars: int = 1000) -> list:
    """Split text into paragraph-based segments that fit within token limits.

    NLLB works best with shorter texts. We split by paragraphs first,
    then by sentences if a paragraph is still too long.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    segments = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= max_chars:
            segments.append(para)
        else:
            # Split long paragraphs by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""
            for sent in sentences:
                if current and len(current) + len(sent) + 1 > max_chars:
                    segments.append(current.strip())
                    current = sent
                else:
                    current = f"{current} {sent}".strip() if current else sent
            if current.strip():
                segments.append(current.strip())

    return segments if segments else [text]


# =====================================================
# TRANSLATION — NLLB-200 batch translation
# =====================================================
def translate_text_batch(texts, target_language="English", source_language="English"):
    target_code = get_lang_code(target_language)
    source_code = get_lang_code(source_language)
    log(f"Translation: {source_language} ({source_code}) → {target_language} ({target_code})")

    # Validate target language token
    try:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_code)
        if forced_bos_token_id == tokenizer.unk_token_id:
            log(f"ERROR: Unknown target language code '{target_code}'")
            return texts
    except Exception as e:
        log(f"ERROR: Failed to get token ID for '{target_code}': {e}")
        return texts

    # Set source language on tokenizer
    tokenizer.src_lang = source_code

    device = next(model.parameters()).device
    results = [""] * len(texts)

    # Collect all segments across all pages
    all_segments = []       # (page_index, segment_text)
    passthrough_pages = set()

    for idx, text in enumerate(texts):
        stripped = (text or "").strip()
        if not stripped or len(re.findall(r"[^\W\d_]", stripped, re.UNICODE)) < 5:
            results[idx] = text or ""
            passthrough_pages.add(idx)
            continue

        segments = _split_into_segments(stripped)
        for seg in segments:
            all_segments.append((idx, seg))

    if not all_segments:
        return results

    log(f"Translating {len(all_segments)} segments from {len(texts)} pages "
        f"(batch_size={BATCH_SIZE})...")

    translated_segments = [""] * len(all_segments)
    total_tokens = 0
    t0 = time.time()

    for batch_start in range(0, len(all_segments), BATCH_SIZE):
        batch_items = all_segments[batch_start:batch_start + BATCH_SIZE]
        batch_texts = [item[1] for item in batch_items]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        # Generate translations
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=MAX_LENGTH,
            )

        total_tokens += output_tokens.numel()

        # Decode
        decoded = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        for i, text in enumerate(decoded):
            translated_segments[batch_start + i] = text.strip()

    gen_time = time.time() - t0
    log(f"Translation: {total_tokens} tokens in {gen_time:.1f}s "
        f"({total_tokens / max(gen_time, 0.001):.0f} tok/s)")

    # Reassemble pages from translated segments
    page_segments = {}
    for seg_idx, (page_idx, _) in enumerate(all_segments):
        if page_idx not in page_segments:
            page_segments[page_idx] = []
        page_segments[page_idx].append(translated_segments[seg_idx])

    for page_idx, segs in page_segments.items():
        results[page_idx] = "\n\n".join(segs)

    return results


# =====================================================
# RunPod handler
# =====================================================
def handler(event):
    log("Handler started")

    input_data = event["input"]
    pages = input_data["pages"]
    target_language = input_data.get("target_language", "English")
    source_language = input_data.get("source_language", "English")

    log(f"Processing {len(pages)} pages, "
        f"translate: {source_language} → {target_language}")

    load_model()

    # Translate all pages
    log(f"Starting batch translation to {target_language}...")
    start = time.time()
    page_texts = [p["text"] for p in pages]
    translated_texts = translate_text_batch(
        page_texts, target_language, source_language
    )
    for i, p in enumerate(pages):
        p["text"] = translated_texts[i]
    log(f"Translation done in {time.time()-start:.2f}s")

    log("Handler finished")
    return {"pages": pages}

runpod.serverless.start({"handler": handler})