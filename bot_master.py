import os
import sys
import json
import asyncio
import requests
import edge_tts
import whisper
from PIL import Image
from google import genai
from deep_translator import GoogleTranslator
from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeAudioClip, 
    concatenate_videoclips, afx
)
from pydub import AudioSegment
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, 
    filters, ContextTypes
)

# --- TELEGRAM BOT TOKEN ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# --- MULTIPLE API KEYS CONFIGURATION (3x Gemini & 3x Pixabay) ---
GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]  # Filter empty values

PIXABAY_KEYS = [
    os.getenv("PIXABAY_API_KEY_1") or os.getenv("PIXABAY_API_KEY"),
    os.getenv("PIXABAY_API_KEY_2"),
    os.getenv("PIXABAY_API_KEY_3")
]
PIXABAY_KEYS = [k for k in PIXABAY_KEYS if k]  # Filter empty values

# Global Whisper AI Model for Video Dubbing
print("Loading OpenAI Whisper Model for Dubbing...")
whisper_model = whisper.load_model("base")

VOICE_MAP = {
    'or': 'or-IN-SubhasiniNeural', # Odia
    'hi': 'hi-IN-SwaraNeural',     # Hindi
    'en': 'en-US-AvaNeural',        # English
    'es': 'es-ES-ElviraNeural',    # Spanish
    'fr': 'fr-FR-DeniseNeural',    # French
}

# --- AUTOMATIC API ROTATION ENGINE ---

def call_gemini_with_fallback(prompt, image_path=None):
    """
    Tries Gemini API Keys sequentially (Key 1 -> Key 2 -> Key 3).
    Auto switches on rate limit, error, or quota depletion.
    Supports both text prompts and photo analysis.
    """
    for idx, key in enumerate(GEMINI_KEYS):
        try:
            print(f" Attempting Gemini Generation with Key #{idx + 1}...")
            client = genai.Client(api_key=key)
            
            if image_path and os.path.exists(image_path):
                img = Image.open(image_path)
                res = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[img, prompt]
                )
            else:
                res = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
            return res.text
        except Exception as e:
            print(f"⚠️ Gemini Key #{idx + 1} failed (Error: {e}). Switching to next key...")
    
    raise Exception("❌ All Gemini API Keys exhausted or failed!")

def fetch_pixabay_video_with_fallback(keyword, scene_index):
    """
    Tries Pixabay API Keys sequentially (Key 1 -> Key 2 -> Key 3).
    Auto switches on rate limit (429/403) or download failure.
    Downloads HD stock video clips for video creation.
    """
    for idx, key in enumerate(PIXABAY_KEYS):
        clean_keyword = requests.utils.quote(keyword)
        url = f"https://pixabay.com/api/videos/?key={key}&q={clean_keyword}&per_page=3&video_type=film"
        
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if data.get('hits') and len(data['hits']) > 0:
                    video_info = data['hits'][0]['videos']
                    v_link = (
                        video_info.get('large', {}).get('url') or 
                        video_info.get('medium', {}).get('url') or
                        video_info.get('small', {}).get('url')
                    )
                    
                    if v_link:
                        v_data = requests.get(v_link, timeout=20).content
                        fn = f"stock_{scene_index}.mp4"
                        with open(fn, "wb") as f:
                            f.write(v_data)
                        print(f" Stock video downloaded using Pixabay Key #{idx + 1}")
                        return fn
            elif res.status_code in [429, 403]:
                print(f"⚠️ Pixabay Key #{idx + 1} hit rate limit ({res.status_code}). Switching key...")
                continue
        except Exception as e:
            print(f"⚠️ Pixabay Key #{idx + 1} error: {e}. Trying next key...")
            
    print(f"❌ All Pixabay API Keys failed for '{keyword}'. Skipping scene visual.")
    return None

# --- FILE HOSTING ENGINE ---

def upload_to_catbox(file_path):
    """Uploads final video to Catbox for 100% Direct Download Link"""
    try:
        url = "https://catbox.moe/user/api.php"
        data = {"reqtype": "fileupload"}
        with open(file_path, "rb") as f:
            files = {"fileToUpload": f}
            res = requests.post(url, data=data, files=files, timeout=60)
        if res.status_code == 200:
            return res.text.strip()
    except Exception as e:
        print(f"Catbox Upload Failed: {e}")
    return None

async def generate_edge_tts(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

# --- PIPELINE 1: AI VIDEO CREATION (TEXT & PHOTO) ---

def generate_storyboard(prompt_text, image_path=None):
    prompt = f"""
    You are an expert AI video producer. Create a detailed video storyboard from this input: '{prompt_text}'
    Return ONLY a raw JSON array of scenes without markdown formatting or code fences:
    [
      {{
        "scene_num": 1,
        "narration": "Full narration script in Odia or requested language",
        "search_keyword": "English keyword for stock video clip"
      }}
    ]
    """
    raw_response = call_gemini_with_fallback(prompt, image_path=image_path)
    clean_json = raw_response.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_json)

async def create_ai_video(user_prompt, image_path=None):
    storyboard = generate_storyboard(user_prompt, image_path=image_path)
    compiled_clips = []
    
    for idx, scene in enumerate(storyboard):
        audio_file = f"speech_{idx}.mp3"
        await generate_edge_tts(scene['narration'], VOICE_MAP['or'], audio_file)
        
        stock_file = fetch_pixabay_video_with_fallback(scene['search_keyword'], idx)
        audio_clip = AudioFileClip(audio_file)
        
        if stock_file and os.path.exists(stock_file):
            v_clip = VideoFileClip(stock_file).loop(duration=audio_clip.duration)
            v_clip = v_clip.set_audio(audio_clip)
            compiled_clips.append(v_clip)
            
    if not compiled_clips:
        return None, None

    # Merge Video & Add BGM
    final_visuals = concatenate_videoclips(compiled_clips, method="compose")
    
    bgm_file = "bgm.mp3"
    if not os.path.exists(bgm_file):
        bgm_res = requests.get("https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3")
        with open(bgm_file, "wb") as f:
            f.write(bgm_res.content)
        
    bgm_clip = AudioFileClip(bgm_file).volumex(0.10)
    bgm_clip = bgm_clip.fx(afx.audio_loop, duration=final_visuals.duration)
    
    mixed_audio = CompositeAudioClip([final_visuals.audio, bgm_clip])
    final_video = final_visuals.set_audio(mixed_audio)
    
    out_file = "ai_created_video.mp4"
    final_video.write_videofile(out_file, fps=24, codec="libx264", audio_codec="aac", logger=None)
    
    direct_link = upload_to_catbox(out_file)
    
    # Cleanup temp clips
    for idx in range(len(storyboard)):
        if os.path.exists(f"speech_{idx}.mp3"):
            os.remove(f"speech_{idx}.mp3")
        if os.path.exists(f"stock_{idx}.mp4"):
            os.remove(f"stock_{idx}.mp4")

    return out_file, direct_link

# --- PIPELINE 2: AUTO VIDEO DUBBING ---

def dub_video(input_path, target_lang='or'):
    video = VideoFileClip(input_path)
    audio_path = "temp_dub_audio.wav"
    video.audio.write_audiofile(audio_path, logger=None)
    
    transcription = whisper_model.transcribe(audio_path)
    segments = transcription.get("segments", [])
    
    translator = GoogleTranslator(source='auto', target=target_lang)
    voice = VOICE_MAP.get(target_lang, VOICE_MAP['or'])
    
    combined_audio = AudioSegment.silent(duration=int(video.duration * 1000))
    
    for idx, seg in enumerate(segments):
        start_ms = int(seg['start'] * 1000)
        translated_text = translator.translate(seg['text'])
        
        seg_audio = f"seg_{idx}.mp3"
        asyncio.run(generate_edge_tts(translated_text, voice, seg_audio))
        
        speech = AudioSegment.from_file(seg_audio)
        combined_audio = combined_audio.overlay(speech, position=start_ms)
        os.remove(seg_audio)
        
    dubbed_audio_path = "final_dubbed_audio.mp3"
    combined_audio.export(dubbed_audio_path, format="mp3")
    
    dubbed_clip = AudioFileClip(dubbed_audio_path)
    final_dubbed_video = video.set_audio(dubbed_clip)
    
    out_file = "dubbed_output.mp4"
    final_dubbed_video.write_videofile(out_file, codec="libx264", audio_codec="aac", logger=None)
    
    direct_link = upload_to_catbox(out_file)
    
    # Clean temporary files
    video.close()
    dubbed_clip.close()
    os.remove(audio_path)
    os.remove(dubbed_audio_path)
    
    return out_file, direct_link

# --- TELEGRAM HANDLERS ---

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 **Super AI Brain Video Agent Ready! (Pixabay & Multi-Key Active)**\n\n"
        "1️⃣ **Create AI Video**: Send text prompt OR send a Photo with instructions.\n"
        "2️⃣ **Dub Video**: Reply to any video message with `/dub or` (Odia), `/dub hi` (Hindi), `/dub en` (English)."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.caption or update.message.text or "Create a professional concept video"
    photo = update.message.photo
    
    image_path = None
    if photo:
        status = await update.message.reply_text("📸 **Analyzing Photo with Gemini AI Brain...**")
        photo_file = await photo[-1].get_file()
        image_path = "temp_input_photo.jpg"
        await photo_file.download_to_drive(image_path)
    else:
        status = await update.message.reply_text(f"🧠 **AI Brain Processing Prompt...**\n'{user_text}'")

    try:
        video_path, direct_link = await create_ai_video(user_text, image_path=image_path)
        if video_path:
            await status.edit_text("📤 Uploading Video & Generating Direct Link...")
            caption = f"✨ **Video Ready!**\n\n🔗 **Direct Download Link:**\n{direct_link}"
            with open(video_path, 'rb') as vf:
                await update.message.reply_video(video=vf, caption=caption, parse_mode="Markdown")
        else:
            await status.edit_text("❌ Error creating video.")
    except Exception as e:
        await status.edit_text(f"❌ Execution Error: {str(e)}")
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

async def dub_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_lang = context.args[0] if context.args else "or"
    
    if not update.message.reply_to_message or not update.message.reply_to_message.video:
        await update.message.reply_text("⚠️ Reply to a video with `/dub or` or `/dub hi`.")
        return
        
    status = await update.message.reply_text("📥 Downloading video for AI Dubbing...")
    video_file = await update.message.reply_to_message.video.get_file()
    input_path = "input_to_dub.mp4"
    await video_file.download_to_drive(input_path)
    
    await status.edit_text("🎙️ **AI Speech Recognition & Language Dubbing in progress...**")
    try:
        dubbed_file, direct_link = dub_video(input_path, target_lang=target_lang)
        caption = f"✅ **Dubbed Video ({target_lang.upper()}) Complete!**\n\n🔗 **Direct Download Link:**\n{direct_link}"
        with open(dubbed_file, 'rb') as vf:
            await update.message.reply_video(video=vf, caption=caption, parse_mode="Markdown")
    except Exception as e:
        await status.edit_text(f"❌ Dubbing Error: {str(e)}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("dub", dub_command_handler))
    app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, message_handler))
    
    print("🚀 AI Agent Bot Running with Pixabay + Multi-Key Auto-Rotation...")
    app.run_polling()
