from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import io
import wave

app = Flask(__name__)
CORS(app)

DEVICE = "cuda"
TORCH_DTYPE = torch.float16
MODEL_ID = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, torch_dtype=TORCH_DTYPE, use_safetensors=True)
model.to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_ID)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=1,
    return_timestamps=True,
    torch_dtype=TORCH_DTYPE,
    device=DEVICE,
)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['file']
    audio_data = audio_file.read()

    result = pipe(audio_data)
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
