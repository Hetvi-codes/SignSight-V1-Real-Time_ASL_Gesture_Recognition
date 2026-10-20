from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import ROOT_DIR, ensure_models
from backend.routes import router
import backend.state as state

app = FastAPI(title="SignSight API Real-Time API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def serve_dashboard():
    index_path = ROOT_DIR / "frontend" / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h2>[ERROR] frontend/index.html not found!</h2>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

frontend_dir = ROOT_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=5000, reload=True)
