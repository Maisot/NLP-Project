from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import json
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Add `enumerate` to the Jinja2 environment
templates.env.globals.update(enumerate=enumerate)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the pre-trained model and tokenizer
model_dir = "/Users/rahaf/NLP_Group_25/model"  # Update this path to your model directory
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)
model.eval()

label_list = ['B-O', 'B-AC', 'I-AC', 'B-LF', 'I-LF']

log_file = "logs.json"

def log_trial(input_text, result):
    try:
        with open(log_file, "r") as file:
            logs = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs.append({"input": input_text, "result": result, "timestamp": timestamp})
    
    with open(log_file, "w") as file:
        json.dump(logs, file, indent=4)

def get_logs():
    try:
        with open(log_file, "r") as file:
            logs = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Error reading logs: %s", e)
        logs = []
    return logs

def test_model(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
    reconstructed_words = []
    current_word = ""
    label_counts = {}
    for token, label_idx in zip(tokens, predictions):
        label = label_list[label_idx]
        if token.startswith("Ġ"):
            if current_word:
                most_common_label = max(label_counts, key=label_counts.get)
                reconstructed_words.append((current_word, most_common_label))
                label_counts = {}
            current_word = token[1:]
        else:
            current_word += token.lstrip("Ġ")
        label_counts[label] = label_counts.get(label, 0) + 1
    if current_word:
        most_common_label = max(label_counts, key=label_counts.get)
        reconstructed_words.append((current_word, most_common_label))
    return reconstructed_words

@app.get("/", response_class=HTMLResponse)
async def read_main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/model", response_class=HTMLResponse)
async def read_model(request: Request):
    return templates.TemplateResponse("model.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, user_input: str = Form(...)):
    result = test_model(user_input)
    log_trial(user_input, result)
    return templates.TemplateResponse("model.html", {"request": request, "result": result, "user_input": user_input})

@app.get("/logs", response_class=HTMLResponse)
async def read_logs(request: Request):
    logs = get_logs()
    logger.debug("Logs: %s", logs)
    return templates.TemplateResponse("logs.html", {"request": request, "logs": logs})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join(os.path.dirname(__file__), "favicon.ico"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
