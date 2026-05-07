# VisioCraft AI

**VisioCraft AI** is an advanced, AI-powered application for object extraction, context-aware background fill, and seamless image composition. It leverages state-of-the-art models for segmentation, detection, and generative inpainting.

---

## 🚀 Features
- **Object Extraction:** Auto-segment objects easily using Mobile SAM and GrabCut.
- **Context-Aware Inpainting:** Fill backgrounds effortlessly with methods like Stable Diffusion and OpenCV fast inpainting.
- **Image Generation:** Enhance and generate elements dynamically.
- **Composition Canvas:** Reposition, scale, rotate, and layer extracted objects to build custom visuals.
- **GPU Acceleration Support:** Optimized object processing using hardware acceleration.

---

## 🛠️ Project Architecture

VisioCraft has been refactored into a clean, modular architecture to ensure scalability and maintainability.

```text
VisioCraft/
├── app/
│   ├── domain/           # Core business logic and data interfaces (Models, Interfaces)
│   ├── infrastructure/   # External integrations (Detection, Generation, Inpainting, Segmentation)
│   ├── routes/           # API Endpoints (Auth, Composition, Health, Uploads, etc.)
│   └── services/         # Application workflow and logic managers
├── models/               # Local AI models (e.g., mobile_sam.pt, yolov8n.pt)
├── static/               # Static assets (CSS, JS, background images)
├── templates/            # Frontend HTML templates
├── config.py             # Centralized configuration settings
├── main.py               # Application entry point
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

*(Note: Older monolithic components have been transitioned to the new `app/` module system)*

---

## ⚙️ Setup Instructions

### 1. Install Dependencies
Make sure you have Python 3.8+ installed.

```bash
# It is recommended to use a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download Models
Download the necessary AI models for local inference (like Mobile SAM and YOLO).
```bash
python download_models.py
```

### 3. Configure Environment Variables
You can set up environment variables for API keys to use advanced cloud-based AI generation.
```bash
export OPENAI_API_KEY="your_openai_api_key"
export HUGGINGFACE_TOKEN="your_huggingface_token"
export STABILITY_API_KEY="your_stability_api_key"
```
*(Note: For Windows PowerShell, use `$env:KEY="value"` instead of `export`)*

### 4. Run the Application
Start the VisioCraft server:
```bash
python main.py
```
Open your modern web browser (Chrome recommended) and navigate to `http://localhost:5000`.

---

## 🖼️ Usage Workflow
1. **Upload:** Start by uploading an image.
2. **Segment & Extract:** Use the AI tools to click/drag around the subject to auto-segment it, extracting it with a transparent background.
3. **Inpaint:** Remove the object from the background using context-aware fill methods.
4. **Compose:** Use the canvas to re-arrange layers, scale, and rotate objects.
5. **Download:** Save the final composition instantly.

---

## 🧠 Inpainting & AI Methods

| Method | Description | Characteristics |
|--------|-------------|-----------------|
| **Stable Diffusion** | Cloud/API-based generation | ⭐ Best Quality, Highly context-aware |
| **OpenCV** | Fast, local edge blending | Instant speed, Emergency fallback |
| **Mobile SAM** | Local AI segmentation model | Accurate object masking |
| **YOLOv8** | Local Object Detection | Fast and robust bounding boxes |

---

## 🛡️ Security Best Practices
- Model files, caches, and uploaded media are ignored via `.gitignore`.
- Avoid hardcoding secrets in `config.py`. Always rely on environment variables (`.env` or system variables) for sensitive tokens (e.g., `HUGGINGFACE_TOKEN`, `STABILITY_API_KEY`).