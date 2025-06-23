import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer

# Load your downloaded model
model = Model("vosk-model-small-en-us-0.15")  # path to your unzipped model

# Configuration
samplerate = 16000
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

# Recognizer
rec = KaldiRecognizer(model, samplerate)

print("🎤 Listening... Speak into the mic.\nPress Ctrl+C to stop.\n")

try:
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                print("🗣️ Recognized:", result.get("text", ""))
            else:
                # Optional: show partial result
                partial = json.loads(rec.PartialResult())
                print("... " + partial.get("partial", ""), end='\r')

except KeyboardInterrupt:
    print("\n🛑 Stopped.")
