import base64
import csv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
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
from io import BytesIO

from starlette.requests import Request as StarletteRequest
from starlette.formparsers import MultiPartParser
from typing import Tuple

# PATCH for multipart upload limit
class CustomUploadRequest(StarletteRequest):
    async def form(self) -> Tuple[dict, dict]:
        parser = MultiPartParser(max_file_size=15 * 1024 * 1024)
        return await parser.parse(self)

app = FastAPI(title="OCR to Excel API")
app.request_class = CustomUploadRequest
MAX_FILE_SIZE = 15 * 1024 * 1024

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POPPLER_PATH = r"C:\Users\tarun\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin" if os.name == 'nt' else None
model = ocr_predictor(pretrained=True)

@app.post("/grades-summary/")
async def grades_summary(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()
        extension = filename.split('.')[-1]

        if extension not in ['csv', 'xlsx', 'xls']:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .csv or .xlsx file.")

        file_bytes = await file.read()

        # Read CSV or Excel accordingly
        if extension == "csv":
            try:
                content = file_bytes.decode("utf-8")
                df = pd.read_csv(io.StringIO(content))
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="CSV file is not UTF-8 encoded.")
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))

        # Basic validation
        if df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="Invalid file format. Expected at least 2 columns.")

        headers = df.columns.tolist()[1:]  # Skip "Register No." or first column
        grade_types = ['O', 'A+', 'A', 'B+', 'B', 'C', 'U']
        summary = {subject: {grade: 0 for grade in grade_types} for subject in headers}

        for _, row in df.iterrows():
            for i, subject in enumerate(headers):
                grade = str(row[subject]).strip().upper()
                if grade in summary[subject]:
                    summary[subject][grade] += 1

        # Prepare chart data
        chart_data = []
        for subject in headers:
            entry = {"subject": subject}
            entry.update(summary[subject])
            chart_data.append(entry)

        return JSONResponse(content={
            "grade_counts": summary,
            "chart_data": chart_data,
            "subjects": headers,
            "grade_types": grade_types
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))@app.post("/convert-pdf/")
async def convert_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()

        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="❌ Uploaded file exceeds the 15MB limit."
            )

        images = convert_from_bytes(pdf_bytes, dpi=300, poppler_path=POPPLER_PATH)
        base64_pages = []

        for img in images:
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_pages.append(encoded)

        return JSONResponse(content={"pages": base64_pages})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

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

            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        text = ' '.join(word.value for word in line.words)
                        ocr_lines.append(text)

        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Error processing file: {file.filename}. {str(e)}"})

    # === METADATA EXTRACTION ===
    semester = ""
    program = ""
    date_of_pub = ""
    exam_title = ""

    for line in ocr_lines:
        clean = line.strip()
        if "Provisional Results" in clean or "End Semester" in clean:
            exam_title = clean
        if match := re.search(r"Semester\s*:\s*(\d+)", clean, re.IGNORECASE):
            semester = match.group(1)
        if match := re.search(r"Programme\s*:\s*(.*)", clean, re.IGNORECASE):
            program = match.group(1).strip()
        if match := re.search(r"Date of Publication\s*:\s*(.*)", clean, re.IGNORECASE):
            date_of_pub = match.group(1).strip()

    # === SUBJECT CODES ===
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
        match = re.search(r'\b(?:\d+\s+)?((?:\d{2,4}\s*){2,5})(.*)', line)
        if match:
            raw_reg = match.group(1)
            reg_no = re.sub(r'\s+', '', raw_reg)
            after_reg = match.group(2).strip()

            if not (10 <= len(reg_no) <= 16) or not reg_no.isdigit():
                i += 1
                continue

            grades = []
            if after_reg:
                for c in after_reg.split():
                    if re.fullmatch(r'[A-Za-z+\-]+', c):
                        grades.append(c.upper())

            j = i + 1
            while len(grades) < len(subject_codes) and j < len(ocr_lines):
                for c in ocr_lines[j].strip().split():
                    if re.fullmatch(r'[A-Z+\-]+', c):
                        grades.append(c)
                j += 1

            if len(grades) >= len(subject_codes) - 1:
                filled_grades = grades + [''] * (len(subject_codes) - len(grades))
                rows.append([reg_no] + filled_grades)
                i = j - 1
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

    return JSONResponse(
        content={
            "csv": csv_stream.getvalue(),
            "metadata": {
                "exam_title": exam_title,
                "semester": semester,
                "program": program,
                "date_of_publication": date_of_pub
            }
        }
    )
