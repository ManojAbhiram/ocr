import cv2
import numpy as np
import os
import re
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from PIL import Image
import google.generativeai as genai
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import shutil

# Configure Gemini model with API key
genai.configure(api_key="AIzaSyArnQrjfaflMruGUbVtPujnputCxZMUZSk")
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# Initialize PaddleOCR with English language and custom model paths
# ======= OCR INIT (GPU ENABLED) =======
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

def enhance_image(image_path, output_name=None):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    filtered = cv2.bilateralFilter(image, 10, 75, 75)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_enhanced = cv2.merge((cl, a, b))
    contrast_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    final = cv2.filter2D(contrast_enhanced, -1, kernel_sharp)

    # ✅ Save to local temp folder
    temp_dir = os.path.join(os.getcwd(), "temp_images")
    os.makedirs(temp_dir, exist_ok=True)

    enhanced_path = output_name or os.path.join(temp_dir, "enhanced_temp.png")
    cv2.imwrite(enhanced_path, final)
    return enhanced_path

# ======= TABLE EXTRACTION =======
def extract_table(result_lines, row_gap=20):
    rows = {}
    for item in result_lines:
        box = item[0]
        if not isinstance(box, (list, tuple)) or not box or not isinstance(box[0], (list, tuple)):
            continue
        y_min = min(pt[1] for pt in box if isinstance(pt, (list, tuple)) and len(pt) > 1)
        x_min = min(pt[0] for pt in box if isinstance(pt, (list, tuple)) and len(pt) > 0)
        text = item[1][0] if len(item[1]) > 0 else ''
        row_key = int(y_min // row_gap)
        rows.setdefault(row_key, []).append((x_min, text))
    table = []
    for y in sorted(rows.keys()):
        row = [text for x, text in sorted(rows[y])]
        table.append(row)
    return table

# ======= OCR IMAGE =======
def ocr_image(img_path):
    enhanced_img = enhance_image(img_path)
    result = ocr.ocr(enhanced_img)
    os.remove(enhanced_img)
    lines = result[0]
    text_out = [line[1][0] for line in lines]
    table_data = extract_table(lines)
    table_str = "\n".join(["\t".join(row) for row in table_data])
    return "\n".join(text_out), table_str

# ======= OCR PDF =======
def ocr_pdf(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)

    # ✅ Local temp folder
    temp_dir = os.path.join(os.getcwd(), "temp_images")
    os.makedirs(temp_dir, exist_ok=True)

    all_results = []
    for i, page in enumerate(pages, 1):
        temp_img = os.path.join(temp_dir, f"page_{i}.jpg")
        page.save(temp_img, "JPEG")
        text, table = ocr_image(temp_img)
        os.remove(temp_img)
        all_results.append((i, text, table))
    return all_results

# ---------- REGEX EXTRACTION ----------
def preprocess_with_regex(text):
    fields = {
        "name": [
            r"(?:Name\s*of\s*the\s*Candidate|Candidate\s*Name|Student\s*Name|Name)\s*[:\-]?\s*([A-Z][a-zA-Z\s\.]+)",
            r"(?:Certified\s+(?:that|by)\s*(?:Mr\.|Mrs\.|Miss|Ms\.)?\s*)([A-Z][a-zA-Z\s\.]+)"
        ],
        "roll_number": [r"(Reg[\. ]?No|Roll\s*No|Enrollment\s*No)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)"],
        "dob": [r"(?:Date\s*of\s*Birth|DOB)\s*[:\-]?\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})"],
        "class": [r"(Class|Standard|Course|Exam\s*Name)\s*[:\-]?\s*([A-Za-z0-9 \-]+)"],
        "total_marks": [r"(Grand\s*Total|Total\s*Marks)\s*[:\-]?\s*(\d+)"],
        "percentage": [r"Percentage\s*[:\-]?\s*(\d{1,3}\.\d{0,2})\s*%"],
        "gpa": [r"(?:SGPA|CGPA)\s*[:\-]?\s*(\d{1,2}\.\d{1,2})"],
        "grade": [r"Grade\s*[:\-]?\s*([A-F][\+\-]?)"],
        "result": [r"Result\s*[:\-]?\s*([A-Z ]+)"],
        "pass_year": [r"(Year\s*of\s*Passing|Exam\s*Date|Year)\s*[:\-]?\s*(\d{4})"]
    }
    extracted_info = {}
    for key, patterns in fields.items():
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    extracted_info[key] = [m[-1] for m in matches]
                else:
                    extracted_info[key] = matches if len(matches) > 1 else matches[0]
                break
    return extracted_info

# ---------- RUN OCR ----------
def run_ocr(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        text, table = ocr_image(file_path)
        return [(1, text, table)]  # single-page list
    elif ext == ".pdf":
        return ocr_pdf(file_path)
    else:
        return []

def group_pages_by_document(pages):
    groups = []
    current_group = []
    prev_key = None

    for page_num, text, table in pages:
        fields = preprocess_with_regex(text)
        name = fields.get("name", "")
        roll = fields.get("roll_number", "")
        current_key = f"{name}|{roll}".strip()

        # If key matches previous page, group together
        if prev_key and current_key and current_key == prev_key:
            current_group.append((page_num, text, table))
        else:
            if current_group:
                groups.append(current_group)
            current_group = [(page_num, text, table)]

        if current_key:
            prev_key = current_key

    if current_group:
        groups.append(current_group)

    return groups  # list of grouped [(page_num, text, table)]s


# ---------- SUMMARIZE USING GEMINI ----------
def summarize_document(ocr_text, table_str):
    structured_info = preprocess_with_regex(ocr_text)

    prompt = f"""
You are an expert in reading OCR outputs from scanned academic and identity documents.
Your task is to detect the type of document and return details in STRICT JSON format for one of the following cases.
Do NOT add extra commentary or explanations — output JSON only.
If a field is missing, use "Not Available".
If Marks / Percentage / GPA contains a comma, replace it with a dot.
Do NOT hallucinate — only use values from OCR TEXT or TABLE.

Possible JSON formats:

---
CASE 1: 10th, 12th, ITI (2 years), or Diploma (3 years) Mark Sheet
{{
  "Document Type": "10th Mark Sheet / 12th Mark Sheet / ITI Mark Sheet / Diploma Mark Sheet",
  "Name": "...",
  "Parent Name": "...",
  "Roll No / Reg No": "...",
  "Date of Birth": "...",
  "Class/Grade, School Name and Board": "...",
  "Marks / Percentage / GPA": "...",
  "Pass Year": "...",
  "Caste": "..."
}}

---
CASE 2: Degree Mark Sheet
{{
  "Document Type": "Degree Mark Sheet",
  "Name": "...",
  "Parent Name": "...",
  "Roll No / Reg No": "...",
  "ABC ID": "...",
  "Date of Birth": "...",
  "Degree": "...",
  "Semester": "...",
  "Board / University": "...",
  "Marks / Percentage / GPA": "...",
  "Pass Year": "...",
  "Caste": "..."
}}

---
CASE 3: ABC ID
{{
  "Document Type": "ABC ID",
  "Name": "...",
  "Parent Name": "...",
  "ID Number": "..."
}}

---
CASE 4: Aadhar Card or Supporting Identity Document
{{
  "Document Type": "Aadhar Card / PAN Card / Driving License / Passport",
  "Name": "...",
  "Father's/Mother's Name": "...",
  "Date of Birth": "...",
  "Gender": "...",
  "Document Number": "...",
  "Issue Date / Validity": "...",
  "Address": "..."
}}

---
CASE 5: Degree or Convocation Certificate
{{
  "Document Type": "Degree Certificate / Convocation Certificate",
  "Name": "...",
  "Parent Name": "...",
  "Roll No / Reg No": "...",
  "Date of Birth": "...",
  "Degree": "...",
  "University": "...",
  "Pass Year": "...",
  "Class/Grade": "..."
}}

OCR TEXT:
\"\"\"{ocr_text}\"\"\"

TABLE:
{table_str}

Return ONLY valid JSON for the detected case.
"""
    response = model.generate_content(prompt).text.strip()

    # Remove Markdown JSON code fences if present
    if response.startswith("```"):
        response = re.sub(r"^```(?:json)?", "", response, flags=re.IGNORECASE).strip()
        response = re.sub(r"```$", "", response).strip()

    try:
        # Parse JSON to ensure it's valid
        parsed_json = json.loads(response)
        return parsed_json  # ✅ Return dict, not string
    except json.JSONDecodeError:
        return {
            "error": "Invalid JSON output",
            "raw_output": response
        }

def summarize_grouped_documents(groups):
    summaries = []
    for group in groups:
        pages = [str(pn) for pn, _, _ in group]
        full_text = "\n".join(text for _, text, _ in group)
        full_table = "\n".join(table for _, _, table in group)

        summary_data = summarize_document(full_text, full_table)  # dict now
        summaries.append({
            "pages": pages,
            "summary": summary_data
        })
    return summaries


app = FastAPI(
    title="OCR & Document Summarization API",
    description="Upload an image or PDF, extract OCR text, and get structured JSON output.",
    version="1.0.0"
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/upload", summary="Upload document and get JSON summary")
async def upload_document(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Allowed: png, jpg, jpeg, pdf")

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        page_results = run_ocr(file_path)
        grouped_docs = group_pages_by_document(page_results)
        summaries = summarize_grouped_documents(grouped_docs)

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "summaries": summaries
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)