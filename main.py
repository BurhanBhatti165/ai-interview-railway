from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from ai_interview_agent import InterviewSession, smart_tts, speech_to_text_whisper
import os
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from models import User
from schemas import UserCreate, UserLogin, UserOut
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import subprocess
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        # Don't raise the exception, just log it
        # This allows the app to start even if database creation fails

# Add this CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Only allow React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep your original endpoints
@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hello"}

@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint for Railway"""
    return {"status": "healthy", "message": "AI Interview API is running"}

@app.get("/about")
def about() -> dict[str, str]:
    return {"message": "This is the about page."}

@app.get("/voice-demo", response_class=HTMLResponse)
async def voice_demo():
    """Serve the voice demo HTML page"""
    try:
        with open("voice_demo.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Voice demo page not found</h1>", status_code=404)

# --- INTERVIEW SESSION ENDPOINTS ---
sessions = {}  # In-memory session store (for demo)

@app.post("/logout/")
async def logout():
    """Clear all sessions and force re-authentication"""
    global sessions
    sessions.clear()
    return {"message": "All sessions cleared successfully"}

@app.post("/start-interview/")
async def start_interview(
    name: str = Form(...),
    agent_choice: str = Form(...),
    resume: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    if resume is not None:
        # Save the uploaded resume as a temp file
        temp_path = f"temp_{name}_{agent_choice}.pdf"
        with open(temp_path, "wb") as f:
            f.write(await resume.read())
        resume_path = temp_path
    else:
        # Look up the user's resume_path in the database
        db_user = db.query(User).filter(User.username == name).first()
        if not db_user or not db_user.resume_path or not os.path.exists(db_user.resume_path):
            raise HTTPException(status_code=404, detail="Resume not found for user")
        resume_path = db_user.resume_path

    try:
        session = InterviewSession(name, agent_choice, resume_path)
        question = session.start()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "An unexpected error occurred: " + str(e)})
    sessions[name] = session
    return {"question": question}

@app.post("/answer/")
async def answer(
    name: str = Form(...),
    answer: str = Form(...)
):
    session = sessions.get(name)
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    if not answer or not answer.strip():
        return {"score": 0, "feedback": "No answer provided. Please answer the question."}
    score, feedback = session.submit_answer(answer)
    if session.is_complete():
        return {"done": True, "summary": session.summary()}
    else:
        next_q = session.next_question()
        return {"done": False, "question": next_q, "score": score, "feedback": feedback}

@app.post("/tts/")
async def tts(text: str = Form(...), voice_id: str = Form("21m00Tcm4TlvDq8ikWAM"), use_fallback: bool = Form(True)):
    """Generate speech using smart TTS with fallback"""
    audio_b64 = smart_tts(text, voice_id, fallback_to_gtts=use_fallback)
    if audio_b64:
        # Return a data URL for direct use in <audio src="">
        audio_url = f"data:audio/mpeg;base64,{audio_b64}"
        return {"audio_url": audio_url, "success": True}
    else:
        return {"error": "Failed to generate speech", "success": False}

@app.post("/stt/")
async def stt(audio: UploadFile = File(...)):
    """Convert speech to text using Whisper"""
    try:
        # Read the uploaded audio file
        audio_data = await audio.read()
        
        # Check if it's a webm file and convert if needed
        if audio.filename and audio.filename.endswith('.webm'):
            # Convert webm to wav using ffmpeg
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_in:
                tmp_in.write(audio_data)
                tmp_in.flush()
                tmp_in_path = tmp_in.name

            # Convert to wav
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            tmp_out_path = tmp_out.name
            tmp_out.close()
            
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", tmp_out_path
                ], check=True, capture_output=True)
                
                # Read the converted wav file
                with open(tmp_out_path, 'rb') as f:
                    audio_data = f.read()
                
                # Clean up temp files
                os.remove(tmp_in_path)
                os.remove(tmp_out_path)
                
            except subprocess.CalledProcessError as e:
                logging.error(f"FFmpeg conversion failed: {e}")
                return {"text": "", "error": "Audio conversion failed"}
        
        # Use Whisper for transcription
        text = speech_to_text_whisper(audio_data)
        
        if text:
            return {"text": text, "success": True}
        else:
            return {"text": "", "error": "Could not transcribe audio", "success": False}
            
    except Exception as e:
        logging.error(f"STT error: {e}")
        return {"text": "", "error": str(e), "success": False}

@app.post("/signup/", response_model=UserOut)
async def signup(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    resume: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    db_user = db.query(User).filter((User.username == username) | (User.email == email)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    # Ensure resumes directory exists
    os.makedirs("resumes", exist_ok=True)
    resume_filename = f"resumes/{username}_{resume.filename}"
    with open(resume_filename, "wb") as f:
        f.write(await resume.read())
    new_user = User(username=username, email=email, password=password, resume_path=resume_filename)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/login/")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or db_user.password != user.password:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    # Optionally, return a simple token or just a success message
    return {"message": "Login successful", "user": db_user.username}

@app.post("/update_resume/")
async def update_resume(
    username: str = Form(...),
    resume: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    db_user = db.query(User).filter(User.username == username).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    os.makedirs("resumes", exist_ok=True)
    resume_filename = f"resumes/{username}_{resume.filename}"
    # Delete old resume if it exists and is different
    if db_user.resume_path and os.path.exists(db_user.resume_path) and db_user.resume_path != resume_filename:
        os.remove(db_user.resume_path)
    with open(resume_filename, "wb") as f:
        f.write(await resume.read())
    db_user.resume_path = resume_filename
    db.commit()
    db.refresh(db_user)
    return {"message": "Resume updated", "resume_path": resume_filename}
