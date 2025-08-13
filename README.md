# OCR Document Processor

## Description
This project processes academic and identity documents in image or PDF format using OCR and AI summarization. It extracts structured data such as names, roll numbers, marks, and other academic details, then uses Google Gemini to generate clean summaries.

Supports multi-page PDFs, image enhancement, and regex-based data extraction for robust results.

## Badges
No badges configured yet.

## Installation

### Requirements
- [ ] Python 3.7+

- [ ] GPU with CUDA support recommended for PaddleOCR GPU acceleration

- [ ] Poppler (for PDF to image conversion)

### Install dependencies on Linux
```
pip install paddlepaddle-gpu==2.6.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install paddleocr==2.7.2
pip install opencv-python pdf2image Pillow google-generativeai
sudo apt-get update && sudo apt-get install -y poppler-utils
pip install scipy==1.11.4 opencv-python-headless==4.7.0.72 scikit-image==0.21.0
```

### Install dependencies on Windows
- [ ] Download [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) for Windows and add its ```bin``` directory to your PATH environment variable.

- [ ] Install Python packages:
```
pip install paddlepaddle-gpu==2.6.0 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
pip install paddleocr==2.7.2
pip install opencv-python pdf2image Pillow google-generativeai
pip install scipy==1.11.4 opencv-python-headless==4.7.0.72 scikit-image==0.21.0
```
## Configuration
Replace the API key in ```app.py```:
```
genai.configure(api_key="YOUR_API_KEY_HERE")
```
Obtain your Google Generative AI key from your Google Cloud Console or Google AI platform.

## Usage
Run the main OCR script:
```
python ocr.py
```
Enter the full path to an image or PDF file when prompted. Type ```exit``` to quit.

## Support
Raise issues via GitLab Issues or contact the maintainers directly.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.


## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
