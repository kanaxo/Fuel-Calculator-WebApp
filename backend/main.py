from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# CORS setup so React can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with frontend URL in production # use "*" for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-csv")
async def process_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # Example processing
    df['processed'] = True

    output_file = "output.csv"
    df.to_csv(output_file, index=False)

    return FileResponse(output_file, filename="output.csv")
