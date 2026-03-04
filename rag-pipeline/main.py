import uvicorn
from api.server import app
from utils.config import HOST, PORT

if __name__ == "__main__":
    print(f"Starting Hackathon RAG FastAPI server on {HOST}:{PORT}...")
    uvicorn.run("api.server:app", host=HOST, port=PORT, reload=True)
