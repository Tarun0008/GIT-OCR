from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import tempfile
import re
import pandas as pd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import io

app = FastAPI(title="OCR to Excel API")
# Allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the OCR model once
model = ocr_predictor(pretrained=True)

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

    # Extract subject codes
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
        reg_match = re.search(r'((?:\d\s*){16})', line)
        if reg_match:
            raw_reg = reg_match.group(1)
            reg_no = raw_reg.replace(" ", "")

            after_reg = line.split(raw_reg)[-1].strip()
            grades = []
            if after_reg:
                candidates = after_reg.split()
                for c in candidates:
                    if re.fullmatch(r'[A-Z+\-]+', c):
                        grades.append(c)

            j = i + 1
            while len(grades) < len(subject_codes) and j < len(ocr_lines):
                next_line = ocr_lines[j].strip()
                if re.fullmatch(r'[A-Z+\-]+', next_line):
                    grades.append(next_line)
                j += 1

            if len(grades) == len(subject_codes):
                rows.append([fix_misaligned_reg_no(reg_no)] + grades)
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
