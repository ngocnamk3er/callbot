import os
import json
import base64
import asyncio
import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv
load_dotenv()
from pydub import AudioSegment
import io

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPGRAM_API_KEY   = os.getenv("DEEPGRAM_API_KEY")

app = FastAPI()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your_openrouter_key")
DEEPGRAM_API_KEY   = os.getenv("DEEPGRAM_API_KEY", "your_deepgram_key")

OPENROUTER_MODEL   = "openai/gpt-4o-mini"  # free model on OpenRouter
SYSTEM_PROMPT = """You are a friendly English conversation partner on a phone call.
Keep responses SHORT (1-3 sentences max) since this is a voice call.
Be natural, warm, and engaging. Ask follow-up questions to keep conversation flowing.
Never use markdown, bullet points, or special characters - speak in plain natural English."""

# ─── CONVERSATION STORE ───────────────────────────────────────────────────────
conversations: dict[str, list] = {}

# ─── TWILIO WEBHOOK ───────────────────────────────────────────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    """Twilio calls this when someone dials in."""
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    host = request.headers.get("host")

    conversations[call_sid] = []

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Hello! I'm your AI conversation partner. Let's have a chat. What would you like to talk about today?</Say>
    <Connect>
        <Stream url="wss://{host}/media-stream/{call_sid}" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


# ─── WEBSOCKET — MEDIA STREAM ─────────────────────────────────────────────────
@app.websocket("/media-stream/{call_sid}")
async def media_stream(websocket: WebSocket, call_sid: str):
    """Handles real-time audio stream from Twilio."""
    await websocket.accept()
    print(f"[{call_sid}] WebSocket connected")

    stream_sid = None
    audio_buffer = bytearray()

    try:
        async for raw in websocket.iter_text():
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                print(f"[{call_sid}] Stream started: {stream_sid}")

            elif event == "media":
                # Accumulate audio chunks (mulaw 8kHz from Twilio)
                chunk = base64.b64decode(msg["media"]["payload"])
                audio_buffer.extend(chunk)

                # Process every ~1.5 seconds of audio (~12000 bytes at 8kHz mulaw)
                if len(audio_buffer) >= 36000:
                    print("Processing audio...")
                    audio_data = bytes(audio_buffer)
                    audio_buffer.clear()

                    # STT → LLM → TTS pipeline
                    transcript = await transcribe_audio(audio_data)
                    if transcript and len(transcript.strip()) > 2:
                        print(f"[{call_sid}] User said: {transcript}")
                        reply = await get_llm_response(call_sid, transcript)
                        print(f"[{call_sid}] Bot reply: {reply}")
                        tts_audio = await text_to_speech(reply)
                        if tts_audio and stream_sid:
                            await send_audio_to_twilio(websocket, stream_sid, tts_audio)

            elif event == "stop":
                print(f"[{call_sid}] Stream stopped")
                break

    except WebSocketDisconnect:
        print(f"[{call_sid}] WebSocket disconnected")
    except Exception as e:
        print(f"[{call_sid}] Error: {e}")
    finally:
        conversations.pop(call_sid, None)


# ─── STT — DEEPGRAM ───────────────────────────────────────────────────────────
async def transcribe_audio(audio_data: bytes) -> str:
    """Send mulaw audio to Deepgram for transcription."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&language=en",
                headers={
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": "audio/mulaw",
                },
                content=audio_data,
            )
            data = resp.json()
            transcript = (
                data.get("results", {})
                    .get("channels", [{}])[0]
                    .get("alternatives", [{}])[0]
                    .get("transcript", "")
            )
            return transcript.strip()
    except Exception as e:
        print(f"STT error: {e}")
        return ""


# ─── LLM — OPENROUTER ─────────────────────────────────────────────────────────
async def get_llm_response(call_sid: str, user_text: str) -> str:
    """Send transcript to OpenRouter LLM and get reply."""
    history = conversations.get(call_sid, [])
    history.append({"role": "user", "content": user_text})

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
                    "max_tokens": 150,
                },
            )
            data = resp.json()
            reply = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"LLM error: {e}")
        reply = "Sorry, I had a little trouble there. Could you say that again?"

    history.append({"role": "assistant", "content": reply})
    conversations[call_sid] = history[-20:]  # keep last 20 turns
    return reply


# ─── TTS — GOOGLE TRANSLATE TTS (free) ───────────────────────────────────────
async def text_to_speech(text: str) -> bytes | None:
    """Use Google Translate TTS (free, no key needed) to generate audio."""
    try:
        params = {
            "ie": "UTF-8",
            "q": text,
            "tl": "en",
            "client": "tw-ob",
        }
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://translate.google.com/translate_tts",
                params=params,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            print(f"TTS status: {resp.status_code}")  # thêm
            print(f"TTS size: {len(resp.content)} bytes")  # thêm
            if resp.status_code == 200:
                return resp.content
    except Exception as e:
        print(f"TTS error: {e}")
    return None


# ─── SEND AUDIO BACK TO TWILIO ────────────────────────────────────────────────
async def send_audio_to_twilio(websocket: WebSocket, stream_sid: str, audio_bytes: bytes):
    try:
        # Convert MP3 → mulaw 8kHz
        mp3_buf = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_mp3(mp3_buf)
        audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(1)
        
        mulaw_buf = io.BytesIO()
        audio.export(mulaw_buf, format="mulaw")
        mulaw_bytes = mulaw_buf.getvalue()
        print(f"Converted to mulaw: {len(mulaw_bytes)} bytes")
    except Exception as e:
        print(f"Convert error: {e}")
        return

    chunk_size = 3200
    for i in range(0, len(mulaw_bytes), chunk_size):
        chunk = mulaw_bytes[i:i+chunk_size]
        payload = base64.b64encode(chunk).decode("utf-8")
        await websocket.send_text(json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload},
        }))
        await asyncio.sleep(0.05)


# ─── HEALTH CHECK ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "Voice bot is running 🎙️"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
