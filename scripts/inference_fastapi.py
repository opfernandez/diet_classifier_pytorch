import os
import torch
import argparse
from fastapi import FastAPI
import uvicorn
from diet_classifier.inference.server import DIETServer


# Instantiates the FastAPI app.
app = FastAPI(title="DIET Inference API", version="1.0")

@app.get("/")
async def root():
    """
    Root Endpoint: Provides basic information about the API.
    """
    return {"message": "DIET inference API is running",
            "endpoints": ["/health", "/predict"],
            "model": "DIETClassifier"}


@app.get("/health")
async def health():
    """
    Health Check Endpoint: Verifies that the API is running and the model is loaded.
    """
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
async def predict(request: dict):
    """
    Prediction Endpoint: Accepts input data and returns model predictions.
    """
    global server
    if not server:
        return {"error": "Model server is not initialized", "status": "error"}
    if "text" not in request:
        return {"error": "No text provided", "status": "error"}
    text_input = request["text"]
    try:
        results = server.predict([text_input])
        return {"status": "success", "result": results[0]}
    except Exception as e:
        return {"error": str(e), "status": "error"}

def main():
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Start DIET FastAPI Inference Server")
    parser.add_argument("-p", "--port", type=int, default=8000, help="Port to run the FastAPI server on")
    args = parser.parse_args()
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Config files
    entities_file = os.path.join(script_dir, "../data/entity_labels.json")
    intents_file =  os.path.join(script_dir, "../data/intent_labels.json")
    model_path = os.path.join(script_dir, "../models/diet_model.pt")
    word_dict_path = os.path.join(script_dir, "../data/word_dict.json")
    ngram_dict_path = os.path.join(script_dir, "../data/ngram_dict.json")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Initialize DIETServer
    global server
    server = DIETServer(
        device=device,
        model_path=model_path,
        word_dict_path=word_dict_path,
        ngram_dict_path=ngram_dict_path,
        entity_labels_path=entities_file,
        intent_labels_path=intents_file
    )
    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()