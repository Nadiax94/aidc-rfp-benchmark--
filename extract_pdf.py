import fitz
import os

folder = "documents"

for file in os.listdir(folder):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(folder, file)

        doc = fitz.open(pdf_path)

        text = ""

        for page in doc:
            text += page.get_text()

        txt_path = pdf_path.replace(".pdf", ".txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        print("Done:", file)
