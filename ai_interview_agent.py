from dotenv import load_dotenv
load_dotenv()

import os
import re
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import Field
import logging
import tempfile
import base64
import requests
import numpy as np
import whisper
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")

# Suppress INFO logs for cleaner output
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("pikepdf").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

os.environ["USER_AGENT"] = "AI-Interview-App/1.0 (contact: burhanbhatti166@gamil.com)"

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader

# --- CONFIG ---
MAX_QUESTIONS = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
AGENT_CHUNK_SIZE = 1000
AGENT_CHUNK_OVERLAP = 200
RETRIEVER_TOP_K = 6

# --- AGENTS ---
AGENTS = {
    "HR": {
        "url": "https://www.geeksforgeeks.org/hr-interview-questions/",
        "prompt": "You are a professional HR interviewer. Ask relevant soft skill or personality questions based on the resume. We have provided you the resume for context. As an HR interviewer, focus on questions related to soft skills, personality traits, and general behavioral aspects of the candidate. Use the resume to tailor your questions. Do not ask repetitive questions or questions already answered in previous questions. Also don't include the name of the user in the question."
    },
    "Web Development": {
        "url": "https://www.simplilearn.com/web-development-interview-questions-article",
        "prompt": "You are a Web Development interviewer. Ask questions related to HTML, CSS, JavaScript, React, or other web-related technologies based on the resume. We have provided you the resume for context. Focus on web development technologies, frameworks, and best practices. Do not ask questions unrelated to web development, such as AI/ML or Data Science. Do not ask repetitive questions or questions already answered in previous questions. Also don't include the name of the user in the question."
    },
    "AI/ML": {
        "url": "https://github.com/andrewekhalel/MLQuestions",  # Changed to InterviewBit (no bot block)
        "prompt": "You are an AI/ML interviewer. Ask questions related to machine learning algorithms, data processing, model evaluation, and AI concepts based on the resume and the provided context. We have provided you the resume for context. Focus on AI/ML topics and avoid questions unrelated to AI/ML, such as Web Development or Data Science. Do not ask repetitive questions or questions already answered in previous questions. Do not focus only on the most prominent project; try to cover different skills, experiences, or projects from the resume. If the candidate has multiple AI/ML experiences, vary your questions to cover different aspects. Also, don't include the name of the user in the question."
    },
    "Data Science": {
        "url": "https://www.geeksforgeeks.org/data-science/data-science-interview-questions-and-answers/",
        "prompt": "You are a Data Science interviewer. Ask questions related to data analysis, statistical methods, and data visualization based on the resume. We have provided you the resume for context. Focus on Data Science topics and avoid questions unrelated to Data Science, such as Web Development or AI/ML. Do not ask repetitive questions or questions already answered in previous questions. Also don't include the name of the user in the question."
    },
    "Project Management": {
        "url": "https://www.simplilearn.com/project-management-interview-questions-and-answers-article",
        "prompt": "You are a Project Management interviewer. Ask questions related to project planning, execution, risk management, and team leadership based on the resume. We have provided you the resume for context. Focus on Project Management topics and avoid questions unrelated to Project Management, such as Web Development or AI/ML. Do not ask repetitive questions or questions already answered in previous questions. Also don't include the name of the user in the question."
    }
}

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# Lazy loading of embeddings to reduce startup time
_embedding = None

def get_embedding():
    global _embedding
    if _embedding is None:
        _embedding = get_embeddings()
    return _embedding

# --- GOOGLE API KEY LOGGING ---
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    logging.warning("[AI-INTERVIEW] GOOGLE_API_KEY not found in environment! LLM calls may fail or use a cached key.")
else:
    logging.info(f"[AI-INTERVIEW] GOOGLE_API_KEY loaded: {google_api_key[:6]}...{'*' * (len(google_api_key)-6) if google_api_key else ''}")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=google_api_key
)

# --- ELEVENLABS TTS ---
def elevenlabs_tts(text, voice_id="21m00Tcm4TlvDq8ikWAM"):
    """Generate speech using ElevenLabs API"""
    api_key = os.getenv("ELEVEN_API_KEY")
    if not api_key:
        logging.error("ELEVEN_API_KEY not found in environment variables.")
        return None
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.7}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            logging.error(f"TTS failed: {response.text}")
            return None
        
        audio_bytes = response.content
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return audio_b64
    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        return None

# --- GTTS FALLBACK TTS ---
def gtts_tts(text, lang='en'):
    """Generate speech using gTTS as fallback"""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{time.time()}.mp3')
        tts.save(temp_file.name)
        temp_file.close()
        
        with open(temp_file.name, 'rb') as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        # Clean up temp file
        try:
            os.unlink(temp_file.name)
        except Exception as e:
            logging.warning(f"Could not delete gTTS temp file: {e}")
        
        return audio_b64
    except Exception as e:
        logging.error(f"gTTS error: {e}")
        return None

# --- SMART TTS FUNCTION ---
def smart_tts(text, voice_id="21m00Tcm4TlvDq8ikWAM", fallback_to_gtts=True):
    """Smart TTS that tries ElevenLabs first, then falls back to gTTS"""
    # Try ElevenLabs first
    audio_b64 = elevenlabs_tts(text, voice_id)
    
    if audio_b64:
        return audio_b64
    
    # If ElevenLabs fails and fallback is enabled, try gTTS
    if fallback_to_gtts:
        logging.info("ElevenLabs failed, trying gTTS fallback...")
        return gtts_tts(text)
    
    return None

# --- WHISPER STT ---
# Lazy loading of Whisper model to reduce startup time
_whisper_model = None

def load_whisper_model():
    """Load Whisper model for speech recognition"""
    global _whisper_model
    if _whisper_model is None:
        try:
            _whisper_model = whisper.load_model("small")
        except Exception as e:
            logging.error(f"Error loading Whisper model: {e}")
            return None
    return _whisper_model

def process_audio_for_whisper(audio_data, sample_rate=16000):
    """Process audio data for better Whisper transcription"""
    try:
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Check if audio has meaningful content
        if len(audio_array) == 0:
            logging.error("No audio data detected")
            return None, None
        
        # Check audio quality
        max_amplitude = np.max(np.abs(audio_array))
        if max_amplitude < 100:  # Very low amplitude
            logging.warning("Audio seems too quiet. Please speak louder.")
        
        # Normalize audio properly
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        # Remove silence from beginning and end with better detection
        # Use RMS-based silence detection
        frame_length = int(sample_rate * 0.025)  # 25ms frames
        hop_length = int(sample_rate * 0.010)    # 10ms hop
        
        rms_values = []
        for i in range(0, len(audio_array) - frame_length, hop_length):
            frame = audio_array[i:i + frame_length]
            rms = np.sqrt(np.mean(frame ** 2))
            rms_values.append(rms)
        
        if rms_values:
            rms_threshold = np.mean(rms_values) * 0.1  # 10% of mean RMS
            non_silent_frames = np.where(np.array(rms_values) > rms_threshold)[0]
            
            if len(non_silent_frames) > 0:
                start_frame = max(0, non_silent_frames[0] - 5)  # Keep 50ms before
                end_frame = min(len(rms_values), non_silent_frames[-1] + 5)  # Keep 50ms after
                
                start_sample = start_frame * hop_length
                end_sample = min(len(audio_array), end_frame * hop_length + frame_length)
                
                audio_array = audio_array[start_sample:end_sample]
        
        # Ensure minimum length
        if len(audio_array) < sample_rate * 0.5:  # At least 0.5 seconds
            logging.warning("Audio recording too short. Please record for at least 0.5 seconds.")
            return None, None
        
        # Ensure maximum length (Whisper works better with shorter segments)
        if len(audio_array) > sample_rate * 30:  # Max 30 seconds
            logging.warning("Audio too long. Using first 30 seconds.")
            audio_array = audio_array[:sample_rate * 30]
        
        return audio_array, sample_rate
    except Exception as e:
        logging.error(f"Audio processing error: {e}")
        return None, None

def speech_to_text_whisper(audio_data):
    """Convert speech to text using Whisper"""
    temp_file_path = None
    try:
        # Process audio for better transcription
        processed_audio, sample_rate = process_audio_for_whisper(audio_data)
        
        if processed_audio is None:
            return None
        
        # Create temporary WAV file with proper format
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file_path = temp_file.name
        temp_file.close()
        
        import wave
        with wave.open(temp_file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((processed_audio * 32768).astype(np.int16).tobytes())
        
        # Use Whisper with improved parameters for better accuracy
        model = load_whisper_model()
        if model is None:
            return None
            
        result = model.transcribe(
            temp_file_path,
            language="en",
            task="transcribe",
            fp16=False,  # Use float32 for better accuracy
            verbose=False,
            condition_on_previous_text=False,  # Don't condition on previous text
            temperature=0.0,  # Use deterministic decoding
            compression_ratio_threshold=2.4,  # Lower threshold for better detection
            logprob_threshold=-1.0,  # Lower threshold for better detection
            no_speech_threshold=0.6  # Higher threshold to avoid false positives
        )
        
        text = result["text"].strip()
        
        # Check if transcription seems reasonable
        def is_reasonable_transcription(text):
            """Check if transcription seems reasonable"""
            if not text:
                return False
            
            # Check for common interview-related words
            interview_words = ['name', 'experience', 'work', 'project', 'team', 'skill', 'job', 'company', 'interview', 'question', 'answer', 'think', 'believe', 'would', 'could', 'should', 'have', 'been', 'working', 'developed', 'created', 'managed', 'led', 'helped', 'improved']
            
            text_lower = text.lower()
            word_count = len(text.split())
            
            # If text is too short, it might be incomplete
            if word_count < 2:
                return False
            
            # Check if it contains reasonable words
            reasonable_words = sum(1 for word in interview_words if word in text_lower)
            if reasonable_words > 0:
                return True
            
            # If no interview words, check for basic English words
            basic_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might']
            basic_word_count = sum(1 for word in basic_words if word in text_lower)
            
            return basic_word_count >= 2
        
        if text and is_reasonable_transcription(text):
            return text
        else:
            # Try with base model if available
            try:
                base_model = whisper.load_model("base")
                result = base_model.transcribe(
                    temp_file_path,
                    language="en",
                    task="transcribe",
                    fp16=False,
                    verbose=False
                )
                text = result["text"].strip()
                
                if text and is_reasonable_transcription(text):
                    return text
                else:
                    return None
            except Exception as e:
                logging.error(f"Base model transcription failed: {e}")
                return None
                
    except Exception as e:
        logging.error(f"Whisper transcription error: {e}")
        return None
    finally:
        # Clean up temp file with retry logic
        if temp_file_path and os.path.exists(temp_file_path):
            for attempt in range(3):
                try:
                    os.unlink(temp_file_path)
                    break
                except PermissionError:
                    if attempt < 2:
                        import time
                        time.sleep(0.1)  # Wait a bit before retry
                    else:
                        logging.warning(f"Could not delete temp file after 3 attempts: {temp_file_path}")
                        break
                except Exception as e:
                    logging.warning(f"Error cleaning up temp file: {e}")
                    break

def load_agent_retriever(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=AGENT_CHUNK_SIZE, chunk_overlap=AGENT_CHUNK_OVERLAP).split_documents(docs)
        vectordb = FAISS.from_documents(chunks, get_embedding())
        return vectordb.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Failed to load web content from {url}: {e}. Agent retriever is required.")

def load_resume(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

class HybridRetriever(BaseRetriever):
    resume_retriever: BaseRetriever = Field(...)
    agent_retriever: BaseRetriever = Field(default=None)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        resume_docs = self.resume_retriever.invoke(query)
        agent_docs = self.agent_retriever.invoke(query) if self.agent_retriever else []
        combined_docs = resume_docs + agent_docs if resume_docs or agent_docs else []
        return combined_docs

def get_agent_chain(retriever, prompt):
    full_prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=prompt + "\n\nResume:\n{context}\nQuestion: {query}\nGenerate one interview question."
    )
    def debug_chain(inputs):
        # Print context and prompt for debugging
        context = inputs["context"] if isinstance(inputs, dict) else inputs
        query = inputs["query"] if isinstance(inputs, dict) else ""
        logging.info("[AI-INTERVIEW] --- LLM CALL ---")
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Query: {query}")
        # Try to get a preview of the context
        if isinstance(context, list):
            preview = str(context[:2]) + ("..." if len(context) > 2 else "")
        else:
            preview = str(context)
        logging.info(f"Context preview: {preview}")
        return inputs
    chain = (
        RunnableMap({"context": retriever, "query": RunnablePassthrough()})
        | debug_chain
        | full_prompt
        | llm
        | StrOutputParser()
    )
    return chain

def get_evaluator_chain(domain):
    eval_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template=f"""You are a senior {domain} interviewer evaluating a candidate's answer.

Question: {{question}}
Answer: {{answer}}

Provide:
Score: <1-10>
Feedback: <short feedback>
"""
    )
    return eval_prompt | llm | StrOutputParser()

def parse_eval_output(result):
    score_match = re.search(r"[Ss]core[:\-–]?\s*\*?(\d{1,2})", result)
    feedback_match = re.search(r"[Ff]eedback[:\-–]?\s*(.*)", result, re.DOTALL)
    score = int(score_match.group(1)) if score_match else 0
    feedback = feedback_match.group(1).strip() if feedback_match else "No feedback available."
    return score, feedback

def save_to_file(agent, name, results):
    filename = f"results_{agent.lower().replace(' ', '_')}.txt"
    with open(filename, "a") as f:
        f.write(f"=== Interview Summary ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===\n")
        f.write(f"Candidate: {name} | Domain: {agent}\n\n")
        for i, entry in enumerate(results, 1):
            f.write(f"Q{i}: {entry['question']}\n")
            f.write(f"A{i}: {entry['answer']}\n")
            f.write(f"Score: {entry['score']}/10\nFeedback: {entry['feedback']}\n\n")
        f.write("=============================================\n\n")

# --- LEGACY TTS (keeping for backward compatibility) ---
def text_to_speech(text, lang='en'):
    """Legacy gTTS function - now uses ElevenLabs"""
    return smart_tts(text)

# --- INTERVIEW SESSION LOGIC (API-friendly) ---
class InterviewSession:
    def __init__(self, name, agent_choice, resume_path):
        self.name = name
        self.agent_choice = agent_choice
        self.resume_path = resume_path
        self.current_q = 1
        self.questions = []
        self.answers = []
        self.results = []
        self.resume_docs = load_resume(resume_path)
        self.resume_chunks = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(self.resume_docs)
        self.resume_vectordb = FAISS.from_documents(self.resume_chunks, get_embedding())
        self.resume_retriever = self.resume_vectordb.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
        agent_config = AGENTS[agent_choice]
        self.agent_retriever = load_agent_retriever(agent_config["url"])
        self.hybrid_retriever = HybridRetriever(
            resume_retriever=self.resume_retriever,
            agent_retriever=self.agent_retriever
        )
        self.qa_chain = get_agent_chain(self.hybrid_retriever, agent_config["prompt"])
        self.evaluator = get_evaluator_chain(agent_choice)
        self.current_question = None

    def start(self):
        print("Resume path:", self.resume_path)
        print("Agent choice:", self.agent_choice)
        # Prepare to capture context and prompt
        # We'll manually run the first part of the chain to get the context
        query = "Start interview"
        # Get context from hybrid retriever
        context = self.hybrid_retriever.invoke(query)
        # Get the agent prompt
        agent_config = AGENTS[self.agent_choice]
        prompt_template = agent_config["prompt"] + "\n\nResume:\n{context}\nQuestion: {query}\nGenerate one interview question."
        # Fill the prompt
        filled_prompt = prompt_template.format(context='\n'.join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in context]), query=query)
        print("--- DEBUG: LLM INPUT ---")
        print(f"Prompt Template: {prompt_template}")
        print(f"Context (first 2 docs): {[doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in context[:2]]}")
        print(f"Full Prompt Sent to LLM:\n{filled_prompt}")
        # Now actually generate the question
        question = self.qa_chain.invoke(query)
        print("Generated question:", question)
        self.current_question = question
        return question

    def next_question(self):
        self.current_question = self.qa_chain.invoke("Next question")
        return self.current_question

    def submit_answer(self, answer):
        eval_result = self.evaluator.invoke({
            "question": self.current_question,
            "answer": answer
        })
        score, feedback = parse_eval_output(eval_result)
        self.questions.append(self.current_question)
        self.answers.append(answer)
        self.results.append({
            "question": self.current_question,
            "answer": answer,
            "score": score,
            "feedback": feedback
        })
        self.current_q += 1
        return score, feedback

    def is_complete(self):
        return self.current_q > MAX_QUESTIONS

    def summary(self):
        total_score = sum(item["score"] for item in self.results)
        return {
            "total_score": total_score,
            "max_score": MAX_QUESTIONS * 10,
            "results": self.results
        }