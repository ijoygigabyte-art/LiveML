from fastapi import FastAPI

app = FastAPI()

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/upload")
def upload_get():
    return {"message": "Upload endpoint is working. Use POST to upload."}
