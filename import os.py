import os
import re
import json
import pandas as pd
import pdfplumber

# Folder where the Python script and PDF are located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PDF is in the same folder as the Python script
PDF_PATH = os.path.join(BASE_DIR, "kla_hpe_sbm_406.pdf")

# Save the output files in the same folder
CSV_OUTPUT = os.path.join(BASE_DIR, "collection.csv")
JSON_OUTPUT = os.path.join(BASE_DIR, "collection.jsonl")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def clean_text(text):
        """Removes headers, footers, copyright lines, and excess whitespace."""
        patterns = [
            r"HEALTH-RELATED FITNESS\s*•\s*UPPER PRIMARY",
            r"HEALTH AND PHYSICAL EDUCATION\s*•\s*•\s*SOURCEBOOK MODULE",
            r"© The State of Queensland.*",
            r"\n\s*\d+\s*\n",  # Standalone page numbers
            r"\n\s*R\d+\s*\n", # Resource labels (R1, R2, etc.)
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE)

        # Clean up whitespace: replace multiple spaces/tabs with one,
        # and collapse multiple newlines into double newlines.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

def get_section(text):
        """Identifies the document section based on keywords."""
        sections = [
            "Understanding", "Planning", "Acting", "Reflecting",
            "Background information", "Activities", "Assessment strategy",
            "Core learning outcomes", "Core content", "Support materials and references"
        ]
        return next((s for s in sections if s.lower() in text.lower()), "General")

def chunk_text(text):
        """Splits text into overlapping chunks based on word count."""
        words = text.split()
        # Calculate step size to ensure overlap
        step = CHUNK_SIZE - CHUNK_OVERLAP
        return [" ".join(words[i : i + CHUNK_SIZE]) for i in range(0, len(words), step)]

def main():
        # Check if PDF exists before starting
        if not os.path.exists(PDF_PATH):
            print(f"Error: PDF not found at {PDF_PATH}")
            return

        collection = []
        passage_id = 1

        print("Extracting and processing PDF...")
        with pdfplumber.open(PDF_PATH) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                # Extract text, handle empty pages, and clean
                raw_text = page.extract_text() or ""
                cleaned_text = clean_text(raw_text)

                if not cleaned_text:
                    continue

                section = get_section(cleaned_text)
                chunks = chunk_text(cleaned_text)

                for chunk in chunks:
                    collection.append({
                        "id": f"P{passage_id:04d}",
                        "contents": chunk,
                        "page": i,
                        "section": section,
                        "source": "Health-related Fitness"
                    })
                    passage_id += 1

        # Convert to DataFrame for easy exporting
        df = pd.DataFrame(collection)

        # Save as CSV
        df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")

        # Save as JSONL (JSON Lines) - required for Pyserini/Search indices
        df.to_json(JSON_OUTPUT, orient='records', lines=True, force_ascii=False)

        print("-" * 30)
        print(f"Success! Processed {len(collection)} chunks.")
        print(f"CSV saved to: {CSV_OUTPUT}")
        print(f"JSONL saved to: {JSON_OUTPUT}")

if __name__ == "__main__":
        main()