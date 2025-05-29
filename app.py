import streamlit as st
import cv2
import tempfile
import re
import pandas as pd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Initialize model once
model = ocr_predictor(pretrained=True)

st.set_page_config(page_title="OCR App", layout="wide")
st.title("OCR ")

# Session state for OCR
if "ocr_lines" not in st.session_state:
    st.session_state.ocr_lines = []
if "ocr_done" not in st.session_state:
    st.session_state.ocr_done = False

# Fix misaligned register numbers like '1240371767262200' → '2403717672622001'
def fix_misaligned_reg_no(reg_no):
    if len(reg_no) == 16 and reg_no.startswith('1') and reg_no[1] == '2':
        return reg_no[1:] + reg_no[0]
    return reg_no

uploaded_files = st.file_uploader(
    "Upload image files (JPG, PNG, TIFF)...",
    type=["jpg", "jpeg", "png", "tiff"],
    accept_multiple_files=True
)

if uploaded_files and st.button(" Convert Step 1"):
    st.session_state.ocr_lines = []
    ocr_output = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        image = cv2.imread(tmp_path)
        if image is None:
            st.warning(f"⚠️ Could not read {uploaded_file.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_path = tmp_path.replace('.jpg', '_preprocessed.jpg')
        cv2.imwrite(preprocessed_path, thresh)

        doc = DocumentFile.from_images(preprocessed_path)
        result = model(doc)

        lines = []
        page = result.pages[0]
        for block in page.blocks:
            for line in block.lines:
                text = ' '.join([word.value for word in line.words])
                lines.append(text)

        ocr_text = "\n".join(lines)
        ocr_output.append((uploaded_file.name, ocr_text))
        st.session_state.ocr_lines.extend(lines)

        st.subheader(f" OCR Result for: {uploaded_file.name}")
        st.text_area("Extracted Text:", ocr_text, height=200)

    # Combine and allow download of clean OCR output
    combined_text = "\n\n".join([f"=== {name} ===\n{text}" for name, text in ocr_output])
    st.download_button(
        label="📥 Download OCR Text File",
        data=combined_text,
        file_name="ocr_output.txt",
        mime="text/plain"
    )

    st.session_state.ocr_done = True
    st.success("✅ OCR complete. Please download the TXT file before converting to Excel.")

# Step 2: Convert to Excel if OCR is done
if st.session_state.ocr_done:
    if st.button("📊 Step 2: Convert OCR to Excel Table"):
        ocr_lines = st.session_state.ocr_lines

        # Subject code detection
        # Enhanced subject code detection: handles '20MSS25', '20MSSE25', etc.
        subject_code_pattern = re.compile(r'\b\d{2}[A-Z]{3,4}\d{2}\b')
        subject_codes = []
        for line in ocr_lines:
            matches = subject_code_pattern.findall(line)
            if len(matches) >= 3:
                subject_codes = matches
                break
        if not subject_codes:
                st.error("❌ No valid subject codes found in OCR text. Cannot proceed to table extraction.")
                st.stop()
        columns = ['Register No.'] + subject_codes
        rows = []
        i = 0
        while i < len(ocr_lines):
            line = ocr_lines[i]
            reg_match = re.search(r'((?:\d\s*){16})', line)
            if reg_match:
                raw_reg = reg_match.group(1)
                reg_no = raw_reg.replace(" ", "")  # Just remove spaces, keep digit order exactly as OCR gave it

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
                    rows.append([reg_no] + grades)
                    i = j
                else:
                    i += 1
            else:
                i += 1

        if not rows:
            st.error("❌ No valid table rows found.")
        else:
            df = pd.DataFrame(rows, columns=columns)
            st.success(f"✅ Extracted {len(rows)} rows")
            st.dataframe(df)

            csv_buffer = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Excel (CSV)",
                data=csv_buffer,
                file_name="extracted_table.csv",
                mime="text/csv"
            )
else:
    st.info("➡️ First click 'Convert (Step 1)' and download the OCR text file to proceed.")
