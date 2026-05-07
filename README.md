# VisioCraft AI
AI-powered object extraction, context-aware background fill, and image composition.
---

## Setup
```bash
pip install -r requirements.txt
python download_models.py   # one-time model download
python main.py
```
Open `http://localhost:5000`
---

## Project Structure
```
VisioCraft/
├── main.py
├── config.py
├── requirements.txt
├── .gitignore
├── README.md
├── static/bg.png
├── models/
│   ├── mobile_sam.pt
│   └── yolov5s.pt
├── Client_Side/
│   ├── SessionManager.py
│   └── Front_End/
│       └── templates/
│           ├── login.html
│           ├── signup.html
│           ├── forget_password.html
│           ├── canvas.html
│           └── generate.html
└── Server_Side/
    ├── Server_Network.py
    ├── page_routes.py
    ├── auth_routes.py
    ├── image_routes.py
    ├── Image_Processing/
    │   ├── Image_Processing_Manager.py
    │   ├── image_processor.py
    │   ├── Image_Generation.py
    │   └── Content_Aware_Fill/
    │       ├── sd_inpaint.py
    │       ├── lama_inpaint.py
    │       └── inpaint_utils.py
    └── Object_Extraction/
        ├── detection_models.py
        ├── object_detector.py
        ├── detection_helpers.py
        ├── masking_models.py
        ├── segmentation.py
        └── object_masking.py
```

---
## Usage
1. Upload image → Drag around object → Auto-segment
2. Extract object (PNG with transparency)
3. Fill background using context-aware inpainting
4. Compose final image → Reposition, scale, rotate → Download
---

## Inpainting Methods
| Method | Quality | Speed | Notes |
|--------|---------|-------|-------|
| Stable Diffusion | ⭐ Best | ~30-60s | Context-aware, Replicate API |
| LaMa AI | Good | ~5-10s | Local, no API needed |
| OpenCV | Basic | Instant | Emergency fallback |
---

## Configuration (`config.py`)
```python
SERVER_PORT = 5000
DEFAULT_INPAINT_METHOD = 'sd'
MAX_IMAGE_DIMENSION = 2048
REPLICATE_API_TOKEN = 'your_token_here'
```

---
## Requirements
- Python 3.8+
- 4GB RAM minimum
- No GPU required — Stable Diffusion runs via Replicate cloud API
- Modern browser (Chrome recommended)
---

## Security
`serviceAccountKey.json` and model files are excluded from version control via `.gitignore`.