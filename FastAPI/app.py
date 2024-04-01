from fastapi import FastAPI, HTTPException
from PredictionInputs import PredictionInput
import joblib
import numpy as np
from fastapi import  Request
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Create an instance of FastAPI
app = FastAPI()

origins = ["*"]  

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["X-Custom-Header"],
)

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Define the request body schema using Pydantic BaseModel


# Define a middleware function to log incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log the request method and URL
    print(f"Received {request.method} request to {request.url}")
    
    # Call the next middleware or route handler
    response = await call_next(request)
    
    return response

@app.options("/predict/")
async def options_handler():
    return {"message": "CORS preflight request handled successfully"}

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

# Define a route to make predictions with the model
@app.post("/predict/")
async def predict(data: PredictionInput):

    print("test")
    # Convert the input data to a numpy array
    input_data = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang, data.oldpeak,
        data.slope, data.ca, data.thal
    ]])

    print(data)

    # Make predictions using the loaded model
    prediction = model.predict(input_data)

    # Return the predicted target value
    return {"prediction": int(prediction[0])}
