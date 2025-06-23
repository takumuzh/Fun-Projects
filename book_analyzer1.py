import os
import re
import json
from typing import List, Tuple

import torch
from PyPDF2 import PdfReader
from transformers import pipeline

# ---- 1. PDF Text Extraction ----
def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text_pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_pages.append(page_text)
    return "\n".join(text_pages)

# ---- 2. Chapter Splitting ----
def split_chapters(text: str) -> List[Tuple[str,str]]:
    pattern = re.compile(r'(?m)^(Chapter\s+\d+[\.:]?.*)$')
    splits = pattern.split(text)
    chapters = []
    if len(splits) > 1:
        for i in range(1, len(splits), 2):
            title = splits[i].strip()
            body  = splits[i+1].strip()
            chapters.append((title, body))
    else:
        chapters = [("Book", text.strip())]
    return chapters

# ---- 3. Summarizer ----
class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        device = 0 if torch.cuda.is_available() else -1
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            device=device
        )

    def summarize(self, text: str, max_length=150, min_length=40) -> str:
        chunks = [ text[i:i+1000] for i in range(0, len(text), 1000) ]
        out = []
        for chunk in chunks:
            res = self.summarizer(chunk,
                                  max_length=max_length,
                                  min_length=min_length,
                                  do_sample=False)
            out.append(res[0]['summary_text'])
        return " ".join(out)

# ---- 4. Trait Classifier ----
class TraitClassifier:
    TRAITS = [
      "Conscientiousness",
      "Low Neuroticism",
      "Openness",
      "Extraversion",
      "Agreeableness"
    ]
    def __init__(self, model_name="facebook/bart-large-mnli"):
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            tokenizer=model_name,
            device=device
        )

    def classify(self, text: str) -> dict:
        out = self.classifier(text,
                              candidate_labels=self.TRAITS,
                              multi_label=True)
        return dict(zip(out["labels"], out["scores"]))

# ---- 5. Main Processing ----
def analyze_book(pdf_path: str, output_json="results.json"):
    print(f"\n📘 Processing: {pdf_path}\n")
    full_text = extract_text(pdf_path)
    chapters  = split_chapters(full_text)

    summarizer = Summarizer()
    classifier = TraitClassifier()

    # Summarize full book
    print("🔍 Summarizing full book…")
    book_summary = summarizer.summarize(full_text)
    print(f"\n📖 Book Summary:\n{book_summary}\n")

    results = {
        "book_summary": book_summary,
        "chapters": []
    }

    # Process chapters
    for idx, (title, body) in enumerate(chapters, 1):
        print(f"📝 {title}")
        chap_sum = summarizer.summarize(body)
        print(f"  ▶️ {chap_sum[:200]}...\n")

        traits = classifier.classify(chap_sum)
        for t, s in traits.items():
            print(f"    - {t}: {s:.2f}")
        print()

        results["chapters"].append({
            "title":   title,
            "summary": chap_sum,
            "traits":  traits
        })

    # Write JSON output
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved to {output_json}")

# ---- Entry Point ----
def select_pdf() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select a PDF Book",
        filetypes=[("PDF Files", "*.pdf")]
    )
    return file_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Summarize a PDF book and classify chapters by Big Five traits")
    parser.add_argument("--pdf_path", help="Path to the PDF file")
    parser.add_argument("--out", default="results.json", help="Output JSON filename")
    args = parser.parse_args()

    if args.pdf_path:
        path = args.pdf_path
    else:
        path = select_pdf()

    if not path or not os.path.isfile(path):
        print("❌ No valid file selected.")
    else:
        analyze_book(path, args.out)
        
