# 🎙️ English Voice Bot

Bot gọi điện tiếng Anh tự động. Người dùng gọi vào → nói chuyện tự do bằng tiếng Anh với AI.

## Tech Stack

| Thành phần | Service | Chi phí |
|---|---|---|
| Telephony | Twilio | ~$1/tháng (số điện thoại) + $0.0085/phút |
| STT | Deepgram | Free tier: 200 giờ/tháng |
| LLM | OpenRouter (Mistral-7B free) | Miễn phí |
| TTS | Google Translate TTS | Miễn phí |
| Backend | FastAPI | Miễn phí |

---

## Setup từng bước

### 1. Cài dependencies

```bash
pip install -r requirements.txt
```

### 2. Lấy API keys

**OpenRouter:**
- Vào https://openrouter.ai → Sign up → API Keys → Create key
- Copy key vào `.env`

**Deepgram:**
- Vào https://console.deepgram.com → Sign up (free)
- Create API Key → Copy vào `.env`

### 3. Config .env

```bash
cp .env.example .env
# Mở .env và điền API keys vào
```

### 4. Chạy server

```bash
# Load env và chạy
export $(cat .env | xargs)
python main.py
```

Server chạy tại http://localhost:8000

### 5. Expose ra internet với ngrok

```bash
# Cài ngrok: https://ngrok.com/download
ngrok http 8000
```

Copy URL dạng `https://abc123.ngrok.io`

### 6. Setup Twilio

1. Vào https://twilio.com → Sign up → lấy $15 credit free
2. **Phone Numbers** → **Buy a number** → chọn số Mỹ (+1) có Voice
3. Click vào số vừa mua → **Voice Configuration**:
   - "A call comes in": **Webhook**
   - URL: `https://abc123.ngrok.io/incoming-call`
   - Method: **HTTP POST**
4. Save

### 7. Test

Gọi vào số Twilio của bạn → bot sẽ chào và bắt đầu chat!

---

## Cấu trúc code

```
main.py
├── /incoming-call      ← Twilio gọi vào đây khi có cuộc gọi
├── /media-stream/{id}  ← WebSocket stream audio real-time
├── transcribe_audio()  ← STT via Deepgram
├── get_llm_response()  ← LLM via OpenRouter
└── text_to_speech()    ← TTS via Google Translate
```

## Luồng hoạt động

```
Người dùng gọi vào số Twilio
    ↓
Twilio gọi POST /incoming-call
    ↓
Server trả TwiML → mở WebSocket stream
    ↓
Audio stream real-time qua WebSocket
    ↓
Deepgram STT → transcript text
    ↓
OpenRouter LLM → reply text
    ↓
Google TTS → audio
    ↓
Gửi audio ngược lại qua WebSocket → Twilio phát cho user
```

---

## Nâng cấp sau này

- **TTS tốt hơn**: Dùng ElevenLabs hoặc Azure TTS (giọng tự nhiên hơn)
- **Latency thấp hơn**: Stream LLM response (chunk by chunk)
- **Barge-in**: Cho phép user ngắt lời bot
- **Audio conversion**: Dùng ffmpeg convert MP3 → mulaw đúng format Twilio
- **Deploy**: Railway hoặc VPS Singapore

---

## Lưu ý quan trọng

Google Translate TTS hiện trả về **MP3**, Twilio cần **mulaw 8kHz**.  
Để production hoàn chỉnh, cần convert audio:

```bash
pip install pydub
# hoặc dùng ffmpeg
```

Mình đã comment trong code chỗ cần convert. Hiện tại dùng Twilio's built-in `<Say>` voice để chào đầu tiên vẫn hoạt động tốt.
# callbot
