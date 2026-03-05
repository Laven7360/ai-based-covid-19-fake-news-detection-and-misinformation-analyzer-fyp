SECRET_KEY = 'your-secret-key'
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

FIREBASE_WEB_API_KEY = 'AIzaSyC6nY5f36SudcbcOHc-oNoSyuaLHfv-tKA'

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  

MODEL_DIR    = str(PROJECT_ROOT / "Model" / "BERT" / "outputs_robertabert" / "best_model")
METRICS_JSON = str(PROJECT_ROOT / "Model" / "BERT" / "outputs_robertabert" / "robertabert_metrics.json")

MAX_LEN      = 320     
INFER_DEVICE = "cpu"   
THRESHOLD    = None    