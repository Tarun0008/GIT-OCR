from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import cv2
import tempfile
import re
import pandas as pd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import io
from pdf2image import convert_from_bytes
import base64
from io import BytesIO

# ----------- PATCH for Multipart File Upload Size (15MB) -----------

from starlette.requests import Request as StarletteRequest
from starlette.formparsers import MultiPartParser
from typing import Tuple

class CustomUploadRequest(StarletteRequest):
    async def form(self) -> Tuple[dict, dict]:
        parser = MultiPartParser(max_file_size=15 * 1024 * 1024)  # 15MB limit
        return await parser.parse(self)

# --- Configure the FastAPI app ---
app = FastAPI(title="OCR to Excel API")
app.request_class = CustomUploadRequest  # Apply the upload limit patch

MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB

# Allow CORS from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to poppler for Windows
POPPLER_PATH = r"C:\Users\tarun\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin" if os.name == 'nt' else None

# Load the OCR model once
model = ocr_predictor(pretrained=True)


@app.post("/convert-pdf/")
async def convert_pdf(file: UploadFile = File(...)):
    try:
        # Read file and validate size
        pdf_bytes = await file.read()
        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="❌ Uploaded file exceeds the maximum size of 15MB."
            )

        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes, dpi=300, poppler_path=POPPLER_PATH)

        # Convert each image to base64
        encoded_images = []
        for img in images:
            print("Image size:", img.size)  # Log image size
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
            encoded_images.append(base64_img)

        return JSONResponse(content={"pages": encoded_images})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


def fix_misaligned_reg_no(reg_no: str) -> str:
    if len(reg_no) == 16 and reg_no.startswith('1') and reg_no[1] == '2':
        return reg_no[1:] + reg_no[0]
    return reg_no


@app.post("/ocr-to-excel/")
async def ocr_to_excel(files: list[UploadFile] = File(...)):
    ocr_lines = []

    for file in files:
        try:
            suffix = "." + file.filename.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            image = cv2.imread(tmp_path)
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_path = tmp_path.replace(suffix, f"_preprocessed{suffix}")
            cv2.imwrite(preprocessed_path, thresh)

            doc = DocumentFile.from_images(preprocessed_path)
            result = model(doc)

            page = result.pages[0]
            for block in page.blocks:
                for line in block.lines:
                    text = ' '.join(word.value for word in line.words)
                    ocr_lines.append(text)

        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Error processing file: {file.filename}. {str(e)}"})

    subject_code_pattern = re.compile(r'\b\d{2}[A-Z]{3,4}\d{2}\b')
    subject_codes = []
    for line in ocr_lines:
        matches = subject_code_pattern.findall(line)
        if len(matches) >= 3:
            subject_codes = matches
            break

    if not subject_codes:
        return JSONResponse(status_code=400, content={"error": "❌ No valid subject codes found."})

    columns = ['Register No.'] + subject_codes
    rows = []


    i = 0
    while i < len(ocr_lines):
        line = ocr_lines[i]

        # 🔧 NEW registration number logic:
        matches = re.findall(r'(?:\d\s*){16}', line)
        if matches:
            raw_reg = matches[-1]
            digits_only = re.sub(r'\s+', '', raw_reg)

            # If reg no starts incorrectly (e.g. 49...), try rotating to fix
            def rotate_to_valid_reg(digits: str) -> str:
                # Try all rotations — look for one starting with '2403'
                for i in range(len(digits)):
                    rotated = digits[i:] + digits[:i]
                    if rotated.startswith("2403") and len(rotated) == 16:
                        return rotated
                return digits  # fallback

            reg_no = rotate_to_valid_reg(digits_only)
            print(line)
            print(f"Line {i}: Raw OCR: {digits_only} → Fixed Reg: {reg_no}")

            # Validate reg no format
            if not (len(reg_no) == 16 and reg_no.isdigit()):
                i += 1
                continue

            after_reg = line.split(raw_reg)[-1].strip()
            
            grades = []
            if after_reg:
                candidates = after_reg.split()
                for c in candidates:
                    if re.fullmatch(r'[A-Za-z+\-]+', c):  # Accept lowercase too
                        grades.append(c.upper())          # Normalize to uppercase

            j = i + 1
            while len(grades) < len(subject_codes) and j < len(ocr_lines):
                next_line = ocr_lines[j].strip()
                grade_tokens = next_line.split()
                for c in grade_tokens:
                    if re.fullmatch(r'[A-Z+\-]+', c):
                        grades.append(c)
                j += 1

            if len(grades) == len(subject_codes):
                rows.append([reg_no] + grades)

                i = j
            else:
                i += 1
        else:
            i += 1

    if not rows:
        return JSONResponse(status_code=400, content={"error": "❌ No valid rows found in OCR output."})

    df = pd.DataFrame(rows, columns=columns)
    csv_stream = io.StringIO()
    df.to_csv(csv_stream, index=False)
    csv_stream.seek(0)

    return StreamingResponse(
        csv_stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ocr_table.csv"}
    )
