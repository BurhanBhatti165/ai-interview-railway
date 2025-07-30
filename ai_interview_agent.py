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
# Removed whisper import - using simpler speech recognition
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress INFO logs for cleaner output
logging.getLogger("pikepdf").setLevel(logging.WARNING)

os.environ["USER_AGENT"] = "AI-Interview-App/1.0 (contact: burhanbhatti166@gamil.com)"

from langchain_google_genai import ChatGoogleGenerativeAI
# Removed HuggingFaceEmbeddings and FAISS imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser

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

# Simplified embeddings function - using basic text similarity instead of FAISS
def get_simple_embeddings():
    """Simple text similarity using basic NLP techniques"""
    return None  # We'll use simple text matching instead

# Lazy loading of embeddings to reduce startup time
_embedding = None

def get_embedding():
    global _embedding
    if _embedding is None:
        _embedding = get_simple_embeddings()
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
    try:
        api_key = os.getenv("ELEVEN_API_KEY")
        if not api_key:
            raise ValueError("ELEVEN_API_KEY not found")
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"ElevenLabs API error: {response.status_code}")
    except Exception as e:
        logging.error(f"ElevenLabs TTS error: {e}")
        raise

def gtts_tts(text, lang='en'):
    """Generate speech using gTTS (Google Text-to-Speech)"""
    try:
        from gtts import gTTS
        import io
        
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        logging.error(f"gTTS error: {e}")
        raise

def smart_tts(text, voice_id="21m00Tcm4TlvDq8ikWAM", fallback_to_gtts=True):
    """Smart TTS with fallback to gTTS if ElevenLabs fails"""
    try:
        return elevenlabs_tts(text, voice_id)
    except Exception as e:
        if fallback_to_gtts:
            logging.warning(f"ElevenLabs failed, falling back to gTTS: {e}")
            return gtts_tts(text)
        else:
            raise

# Simplified speech recognition without whisper
def speech_to_text_simple(audio_data):
    """Simple speech recognition using basic audio processing"""
    try:
        # For now, return a placeholder - you can implement basic speech recognition
        # or use a cloud service like Google Speech-to-Text API
        return "Speech recognition not available in simplified version"
    except Exception as e:
        logging.error(f"Speech recognition error: {e}")
        return ""

# Keep the original function name for compatibility
def speech_to_text_whisper(audio_data):
    """Alias for speech_to_text_simple for compatibility"""
    return speech_to_text_simple(audio_data)

# Simplified retriever without FAISS
def load_agent_retriever(url):
    """Load agent retriever using simple text matching instead of FAISS"""
    try:
        # Simple web scraping without WebBaseLoader
        import requests
        from bs4 import BeautifulSoup
        
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text content
        text_content = soup.get_text()
        
        # Create a simple document
        doc = Document(page_content=text_content, metadata={"source": url})
        
        # Simple text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=AGENT_CHUNK_SIZE,
            chunk_overlap=AGENT_CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents([doc])
        
        # Return a simple retriever that uses text matching
        return SimpleTextRetriever(chunks)
    except Exception as e:
        logging.error(f"Error loading agent retriever: {e}")
        return None

def load_resume(file_path):
    """Load resume using simple PDF processing"""
    try:
        import pypdf
        
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text_content = ""
            
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            doc = Document(page_content=text_content, metadata={"source": file_path})
            return [doc]
    except Exception as e:
        logging.error(f"Error loading resume: {e}")
        return []

# Simple text retriever to replace FAISS
class SimpleTextRetriever(BaseRetriever):
    def __init__(self, documents):
        self.documents = documents
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """Simple text matching instead of vector similarity"""
        # Basic keyword matching
        query_lower = query.lower()
        relevant_docs = []
        
        for doc in self.documents:
            if query_lower in doc.page_content.lower():
                relevant_docs.append(doc)
        
        return relevant_docs[:RETRIEVER_TOP_K]

class HybridRetriever(BaseRetriever):
    resume_retriever: BaseRetriever = Field(...)
    agent_retriever: BaseRetriever = Field(default=None)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """Combine resume and agent retrievers"""
        docs = []
        if self.resume_retriever:
            docs.extend(self.resume_retriever._get_relevant_documents(query, run_manager=run_manager))
        if self.agent_retriever:
            docs.extend(self.agent_retriever._get_relevant_documents(query, run_manager=run_manager))
        return docs[:RETRIEVER_TOP_K]

def get_agent_chain(retriever, prompt):
    """Create agent chain with simple retriever"""
    template = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt + "\n\nContext: {context}\n\nQuestion: {question}"
    )
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | template
        | llm
        | StrOutputParser()
    )
    
    return chain

def get_evaluator_chain(domain):
    """Create evaluator chain"""
    eval_prompt = f"""You are an expert {domain} interviewer evaluating a candidate's answer. 
    Rate the answer on a scale of 1-10 and provide constructive feedback.
    
    Answer: {{answer}}
    
    Provide your evaluation in this format:
    Score: [1-10]
    Feedback: [your detailed feedback]
    """
    
    template = PromptTemplate(
        input_variables=["answer"],
        template=eval_prompt
    )
    
    chain = template | llm | StrOutputParser()
    return chain

def parse_eval_output(result):
    """Parse evaluation output"""
    try:
        lines = result.strip().split('\n')
        score = None
        feedback = ""
        
        for line in lines:
            if line.startswith("Score:"):
                score = int(line.split(":")[1].strip())
            elif line.startswith("Feedback:"):
                feedback = line.split(":", 1)[1].strip()
        
        return {"score": score or 5, "feedback": feedback or result}
    except:
        return {"score": 5, "feedback": result}

def save_to_file(agent, name, results):
    """Save interview results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interview_results_{name}_{agent}_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Interview Results for {name}\n")
        f.write(f"Agent: {agent}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"Question {i}:\n")
            f.write(f"Question: {result['question']}\n")
            f.write(f"Answer: {result['answer']}\n")
            f.write(f"Score: {result['score']}/10\n")
            f.write(f"Feedback: {result['feedback']}\n")
            f.write("-" * 30 + "\n\n")
    
    return filename

def text_to_speech(text, lang='en'):
    """Text to speech function"""
    return smart_tts(text)

class InterviewSession:
    def __init__(self, name, agent_choice, resume_path):
        self.name = name
        self.agent_choice = agent_choice
        self.resume_path = resume_path
        self.current_question = 0
        self.questions = []
        self.answers = []
        self.scores = []
        self.feedbacks = []
        self.retriever = None
        self.chain = None
        self.evaluator = None
        
        # Load resume and create retriever
        if resume_path and os.path.exists(resume_path):
            resume_docs = load_resume(resume_path)
            if resume_docs:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                resume_chunks = text_splitter.split_documents(resume_docs)
                resume_retriever = SimpleTextRetriever(resume_chunks)
                
                # Load agent retriever
                agent_retriever = load_agent_retriever(AGENTS[agent_choice]["url"])
                
                self.retriever = HybridRetriever(
                    resume_retriever=resume_retriever,
                    agent_retriever=agent_retriever
                )
            else:
                # Fallback to agent-only retriever
                self.retriever = load_agent_retriever(AGENTS[agent_choice]["url"])
        else:
            # No resume, use agent-only retriever
            self.retriever = load_agent_retriever(AGENTS[agent_choice]["url"])
        
        if self.retriever:
            self.chain = get_agent_chain(self.retriever, AGENTS[agent_choice]["prompt"])
            self.evaluator = get_evaluator_chain(agent_choice)

    def start(self):
        """Start the interview session"""
        if not self.chain:
            return "Error: Could not initialize interview chain"
        
        try:
            # Generate first question
            question = self.chain.invoke("Generate an interview question")
            self.questions.append(question)
            return question
        except Exception as e:
            logging.error(f"Error starting interview: {e}")
            return "Error: Could not generate interview question"

    def next_question(self):
        """Generate next question"""
        if self.current_question >= MAX_QUESTIONS:
            return None
        
        try:
            question = self.chain.invoke("Generate the next interview question")
            self.questions.append(question)
            return question
        except Exception as e:
            logging.error(f"Error generating next question: {e}")
            return "Error: Could not generate next question"

    def submit_answer(self, answer):
        """Submit answer and get evaluation"""
        if self.current_question >= len(self.questions):
            return "Error: No current question"
        
        self.answers.append(answer)
        
        # Evaluate the answer
        if self.evaluator:
            try:
                eval_result = self.evaluator.invoke({"answer": answer})
                parsed_eval = parse_eval_output(eval_result)
                self.scores.append(parsed_eval["score"])
                self.feedbacks.append(parsed_eval["feedback"])
            except Exception as e:
                logging.error(f"Error evaluating answer: {e}")
                self.scores.append(5)
                self.feedbacks.append("Evaluation failed")
        else:
            self.scores.append(5)
            self.feedbacks.append("No evaluator available")
        
        self.current_question += 1
        
        # Generate next question if not complete
        if not self.is_complete():
            return self.next_question()
        else:
            return "Interview complete"

    def is_complete(self):
        """Check if interview is complete"""
        return self.current_question >= MAX_QUESTIONS

    def summary(self):
        """Get interview summary"""
        if not self.answers:
            return "No answers submitted"
        
        avg_score = sum(self.scores) / len(self.scores)
        return {
            "name": self.name,
            "agent": self.agent_choice,
            "total_questions": len(self.questions),
            "total_answers": len(self.answers),
            "average_score": round(avg_score, 2),
            "scores": self.scores,
            "feedbacks": self.feedbacks
        }