import pyaudio
import numpy as np
import wave
import io
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import concurrent.futures

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 
SILENCE_THRESHOLD = 300  # Threshold for determining silence in audio chunks
SILENCE_DURATION = 1.0  # Silence duration for determining the end of an audio chunk

DEVICE = "cuda"
TORCH_DTYPE = torch.float16
MODEL_ID = "openai/whisper-large-v3"

executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

def is_silence(data, threshold):
    audio_data = np.frombuffer(data, np.int16)
    sound_level = np.abs(audio_data).mean()
    return sound_level < threshold


def process_audio_data(audio_chunk):
    # check that the audio chunk is at least 0.5 seconds + 1 second of silence
    if len(audio_chunk) < RATE * 3:
        return

    bytes_memory = io.BytesIO()
    wf = wave.open(bytes_memory, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(audio_chunk)
    wf.close()

    bytes_memory.seek(0)
    audio_data = bytes_memory.read()

    result = pipe(audio_data)
    print (result)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, torch_dtype=TORCH_DTYPE, use_safetensors=True
)
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


p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    if device_info['maxInputChannels'] > 0:
        print(f"Device {i}: {device_info['name']}")

device_input = int(input("Select input device: "))
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=1)

buffer = []
total_chunks = 0
silent_chunks = 0

while True:
    data = stream.read(CHUNK)
    buffer.append(data)
    
    if is_silence(data, SILENCE_THRESHOLD):
        silent_chunks += 1
    else:
        silent_chunks = 0
    
    total_chunks += 1
    
    if silent_chunks > (RATE / CHUNK * SILENCE_DURATION):
        audio_chunk = b''.join(buffer)
        future = executor.submit(process_audio_data, audio_chunk)

        buffer = []
        silent_chunks = 0
        total_chunks = 0

stream.stop_stream()
stream.close()
p.terminate()