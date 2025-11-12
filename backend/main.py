from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from calculate import calculate_fuel_for_flights

saveopt2 = "All"  # Can be "All", "FIR", "TMA", "New"
holdmode = "BADA"  # Can be "BADA", "BADA_Cruise", "NATS"

app = FastAPI()

# CORS setup so React can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with frontend URL in production # use "*" for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_form_bool(value: str = Form("false")) -> bool:
    """
    Converts a FormData string to a boolean.
    Accepts: "true", "false", "1", "0", "yes", "no", "on", "off" (case-insensitive)
    """
    print("The value is", value)
    if isinstance(value, bool):
        return value
    val = value.lower()
    if val in ("true", "1", "yes", "on"):
        return True
    elif val in ("false", "0", "no", "off"):
        return False
    else:
        raise HTTPException(status_code=400, detail=f"Invalid boolean value: {value}")


@app.post("/process-csv")
async def process_csv(
    file: UploadFile = File(...),
    holdmode: str = Form(...),
    saveopt2: str = Form(...),
    change_speed: str = Form(...),
    boundary_file: UploadFile | None = File(None),
    ):
    try:
        
        df = pd.read_csv(file.file)

        #### Handle ChangeSpeed ####
        if change_speed.lower() in ("true", "1", "yes", "on"):
            change_speed = True
        else:
            change_speed = False

        #### Handle custom boundary ####
        boundary_df = None
        if saveopt2 == "Custom":
            if boundary_file:
                boundary_df = pd.read_csv(boundary_file.file)

                # Validation: columns exist
                required_cols = ["latitude", "longitude"]
                if not all(col in boundary_df.columns for col in required_cols):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Boundary CSV must have columns: {required_cols}"
                    )

                # Validation: values are numeric
                try:
                    boundary_df["latitude"] = boundary_df["latitude"].astype(float)
                    boundary_df["longitude"] = boundary_df["longitude"].astype(float)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail="Latitude and Longitude values must be numeric."
                    )

            else:
                # use default file if no upload
                print("No custom boundary file uploaded, using default.")
                boundary_df = pd.read_csv("Boundaries/custom_boundary.csv")

        # Example processing
        # df['processed'] = True

        #### MAIN PROCESSING FUNCTION CALL ####
        final_df = calculate_fuel_for_flights(df, holdmode, saveopt2, boundary_df, change_speed)

        output_file = "output.csv"
        final_df.to_csv(output_file, index=False)

        return FileResponse(output_file, filename="output.csv")
    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

