#!/usr/bin/env python3
"""
FastAPI server for model inference.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI CI/CD Fixer API", version="1.0.0")

class FixRequest(BaseModel):
    error_message: str
    context_code: str
    file_path: str

class FixResponse(BaseModel):
    generated_diff: str
    confidence: float

# Load model and tokenizer
@app.on_event("startup")
async def load_model():
    global model, tokenizer, device
    
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Using device: {'cuda' if device == 0 else 'cpu'}")
    
    try:
        model_path = "./codebert-fix-model-final"
        logger.info(f"Loading model from {model_path}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        if device == 0:
            model = model.cuda()
            
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@app.post("/predict", response_model=FixResponse)
async def predict_fix(request: FixRequest):
    """
    Generate a fix for a compilation error.
    """
    try:
        # Prepare input text
        input_text = f"Fix this Java error: {request.error_message} \n Code: {request.context_code}"
        
        # Tokenize input
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding="max_length"
        )
        
        if device == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=256,
                num_beams=5,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Calculate confidence (average probability of generated sequence)
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        confidence = float(torch.exp(transition_scores.mean()).cpu().numpy())
        
        return FixResponse(
            generated_diff=generated_text,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)