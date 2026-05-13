import os
import json
import base64
import asyncio
import audioop
import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
import uvicorn
from dotenv import load_dotenv
load_dotenv()
from pydub import AudioSegment
import io

app = FastAPI()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY    = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY        = os.getenv("GOOGLE_API_KEY")

OPENROUTER_MODEL   = "openai/gpt-4o"
SYSTEM_PROMPT = """ທ່ານແມ່ນຜູ້ຊ່ວຍ AI ທີ່ເປັນມິດ ກຳລັງລົມກັນທາງໂທລະສັບ.
ຕອບສັ້ນໆ (1-3 ປະໂຫຍກ) ເພາະນີ້ແມ່ນການໂທ.
ເປັນທຳມະຊາດ ອົບອຸ່ນ ແລະ ຕັ້ງຄຳຖາມຕໍ່ເນື່ອງ.
ຫ້າມໃຊ້ markdown ຫຼື ສັນຍາລັກພິເສດ - ເວົ້າພາສາລາວທຳມະຊາດ."""

conversations: dict[str, list] = {}

@app.post("/incoming-call")
async def incoming_call(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    host = request.headers.get("host")
    conversations[call_sid] = []
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/media-stream/{call_sid}" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/media-stream/{call_sid}")
async def media_stream(websocket: WebSocket, call_sid: str):
    await websocket.accept()
    print(f"[{call_sid}] WebSocket connected")
    stream_sid = None
    audio_buffer = bytearray()
    greeted = False

    try:
        async for raw in websocket.iter_text():
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                print(f"[{call_sid}] Stream started: {stream_sid}")

            elif event == "media":
                chunk = base64.b64decode(msg["media"]["payload"])
                audio_buffer.extend(chunk)

                # Gửi lời chào ngay khi nhận đủ ~0.5s audio đầu tiên
                if not greeted and stream_sid and len(audio_buffer) >= 4000:
                    greeted = True
                    print(f"[{call_sid}] Sending greeting...")
                    greeting = "ສະບາຍດີ! ຂ້ອຍແມ່ນຜູ້ຊ່ວຍ AI. ທ່ານຕ້ອງການຊ່ວຍຫຍັງ?"
                    tts_audio = await text_to_speech(greeting)
                    if tts_audio and stream_sid:
                        await send_audio_to_twilio(websocket, stream_sid, tts_audio)

                # Xử lý ~4.5 giây audio (36000 bytes mulaw = 4.5s @ 8kHz)
                if len(audio_buffer) >= 36000:
                    print("Processing audio...")
                    audio_data = bytes(audio_buffer)
                    audio_buffer.clear()
                    transcript = await transcribe_audio(audio_data)
                    if transcript and len(transcript.strip()) > 1:
                        print(f"[{call_sid}] User said: {transcript}")
                        reply = await get_llm_response(call_sid, transcript)
                        print(f"[{call_sid}] Bot reply: {reply}")
                        tts_audio = await text_to_speech(reply)
                        if tts_audio and stream_sid:
                            await send_audio_to_twilio(websocket, stream_sid, tts_audio)
                    else:
                        print(f"[{call_sid}] No speech detected or empty transcript")

            elif event == "stop":
                print(f"[{call_sid}] Stream stopped")
                break

    except WebSocketDisconnect:
        print(f"[{call_sid}] WebSocket disconnected")
    except Exception as e:
        print(f"[{call_sid}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conversations.pop(call_sid, None)


async def transcribe_audio(audio_data: bytes) -> str:
    """Google Cloud Speech-to-Text — gửi đúng định dạng LINEAR16."""
    try:
        # ✅ Bước 1: Decode mulaw → linear16 PCM (dùng audioop built-in, không cần ffmpeg)
        pcm_data = audioop.ulaw2lin(audio_data, 2)  # 2 = 16-bit

        # ✅ Bước 2: Tạo WAV file header đúng chuẩn
        wav_buf = io.BytesIO()
        audio_seg = AudioSegment(
            data=pcm_data,
            sample_width=2,
            frame_rate=8000,
            channels=1
        )
        audio_seg.export(wav_buf, format="wav")
        wav_bytes = wav_buf.getvalue()
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": 8000,
                "languageCode": "lo-LA",
                "alternativeLanguageCodes": ["th-TH", "en-US"],
                "enableAutomaticPunctuation": False,
                "model": "default",
            },
            "audio": {
                "content": audio_b64
            },
        }

        url = f"https://speech.googleapis.com/v1/speech:recognize?key={GOOGLE_API_KEY}"
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            data = resp.json()
            print(f"Google STT response: {data}")
            results = data.get("results", [])
            if results:
                transcript = results[0]["alternatives"][0]["transcript"]
                return transcript.strip()
            return ""
    except Exception as e:
        print(f"STT error: {e}")
        import traceback
        traceback.print_exc()
        return ""


async def get_llm_response(call_sid: str, user_text: str) -> str:
    """GPT-4o via OpenRouter."""
    history = conversations.get(call_sid, [])
    history.append({"role": "user", "content": user_text})
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
                    "max_tokens": 150
                },
            )
            data = resp.json()
            reply = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"LLM error: {e}")
        reply = "ຂໍໂທດ, ມີບັນຫາເລັກນ້ອຍ. ກະລຸນາເວົ້າອີກຄັ້ງ."
    history.append({"role": "assistant", "content": reply})
    conversations[call_sid] = history[-20:]
    return reply


async def text_to_speech(text: str) -> bytes | None:
    """OpenAI TTS-1."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "tts-1",
                    "input": text,
                    "voice": "nova",
                    "response_format": "mp3"
                },
            )
            print(f"TTS status: {resp.status_code}, size: {len(resp.content)} bytes")
            if resp.status_code == 200:
                return resp.content
            else:
                print(f"TTS error response: {resp.text}")
    except Exception as e:
        print(f"TTS error: {e}")
    return None


async def send_audio_to_twilio(websocket: WebSocket, stream_sid: str, audio_bytes: bytes):
    """Chuyển MP3 → mulaw 8kHz → gửi về Twilio."""
    try:
        # ✅ Decode MP3 → PCM qua pydub
        mp3_buf = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_mp3(mp3_buf)
        audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)

        # ✅ Lấy raw PCM bytes rồi encode sang mulaw bằng audioop
        pcm_bytes = audio.raw_data
        mulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)  # 2 = 16-bit input
        print(f"Converted to mulaw: {len(mulaw_bytes)} bytes")
    except Exception as e:
        print(f"Convert error: {e}")
        import traceback
        traceback.print_exc()
        return

    chunk_size = 3200
    for i in range(0, len(mulaw_bytes), chunk_size):
        chunk = mulaw_bytes[i:i + chunk_size]
        payload = base64.b64encode(chunk).decode("utf-8")
        await websocket.send_text(json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload}
        }))
        await asyncio.sleep(0.05)

    await websocket.send_text(json.dumps({
        "event": "mark",
        "streamSid": stream_sid,
        "mark": {"name": "done"}
    }))
    print("Audio sent to Twilio ✅")


@app.get("/")
async def root():
    return {"status": "Lao Voice Bot is running 🎙️"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)