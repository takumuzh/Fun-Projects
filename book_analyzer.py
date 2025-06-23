import sys
import json
import re
from PyPDF2 import PdfReader
import openai
import os

# ---- CONFIGURATION ----
openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure to export this in your terminal

# ---- STEP 1: Read PDF ----
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# ---- STEP 2: Split into Chapters ----
def split_into_chapters(text):
    chapters = re.split(r'(Chapter\s+\d+[:\s]?.*)', text, flags=re.IGNORECASE)
    structured = []
    for i in range(1, len(chapters), 2):
        title = chapters[i].strip()
        content = chapters[i+1].strip()
        structured.append({'title': title, 'content': content})
    return structured

# ---- STEP 3: Summarize with GPT ----
def summarize_text(text, instruction="Summarize this text:"):
    prompt = f"{instruction}\n\n{text[:4000]}"  # Truncate to fit token limits
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message['content'].strip()

# ---- STEP 4: Classify into Big Five ----
def classify_traits(text):
    prompt = f"""
Given the following chapter content, classify the psychological tone across the Big Five traits:
1. Conscientiousness
2. Neuroticism (inverse: low neuroticism)
3. Openness
4. Extraversion
5. Agreeableness

Respond with a JSON object like:
{{
  "Conscientiousness": "High",
  "Neuroticism": "Low",
  "Openness": "Moderate",
  "Extraversion": "Low",
  "Agreeableness": "High"
}}

Content:
{text[:4000]}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    try:
        result = json.loads(response.choices[0].message['content'])
    except json.JSONDecodeError:
        result = {"Error": "Could not parse traits"}
    return result

# ---- MAIN EXECUTION ----
def analyze_book(pdf_path, output_path):
    book_text = extract_text_from_pdf(pdf_path)

    full_summary = summarize_text(book_text, "Summarize this entire book in under 300 words:")

    chapters = split_into_chapters(book_text)

    chapter_analyses = []
    for chap in chapters:
        print(f"Analyzing: {chap['title']}")
        chap_summary = summarize_text(chap['content'], f"Summarize this chapter titled '{chap['title']}':")
        traits = classify_traits(chap['content'])
        chapter_analyses.append({
            "title": chap['title'],
            "summary": chap_summary,
            "traits": traits
        })

    output = {
        "book_summary": full_summary,
        "chapters": chapter_analyses
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Analysis complete. Output saved to: {output_path}")

# ---- ENTRY POINT ----
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python book_analyzer.py <pdf_path> <output_json_path>")
        sys.exit(1)
    analyze_book(sys.argv[1], sys.argv[2])
