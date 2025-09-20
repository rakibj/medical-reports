import base64
from openai import OpenAI
from PIL import Image
import pypdfium2 as pdfium
from pathlib import Path

class OCRProcessor:
    def __init__(self):
        self.client = OpenAI()

    def pdf_to_images_pypdfium2(self, pdf_path: str, dpi: int = 200):
        pdf = pdfium.PdfDocument(str(pdf_path))
        scale = dpi / 72.0  # PDF base is 72 dpi
        images = []
        for i in range(len(pdf)):
            page = pdf[i]
            pil = page.render(scale=scale).to_pil()   # PIL.Image
            images.append(pil.convert("RGB"))
        pdf.close()
        return images

    def encode_image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL image to base64 string."""
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def ocr_page(self, image: Image.Image) -> str:
        """Send one image to OpenAI for OCR."""
        b64 = self.encode_image_to_base64(image)
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",  # cheap + handles handwriting
            messages=[
                {"role": "system", "content": "You are an OCR engine. Output the text exactly as written."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract the text from this document page. Only return the extracted text and nothing else"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]},
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()

    def ocr_file(self, file_path: str) -> str:
        """Auto-detect type and OCR accordingly."""
        path = Path(file_path)
        ext = path.suffix.lower()

        texts = []

        if ext == ".pdf":
            #pages = convert_from_path(file_path, dpi=200)
            pages = self.pdf_to_images_pypdfium2(file_path, dpi=200)
            for i, page in enumerate(pages, start=1):
                print(f"OCR page {i}/{len(pages)}...")
                texts.append(self.ocr_page(page))
        else:
            img = Image.open(file_path).convert("RGB")
            texts.append(self.ocr_page(img))

        return "\n\n--- PAGE BREAK ---\n\n".join(texts)