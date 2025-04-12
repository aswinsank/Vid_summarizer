from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from typing import List, Optional, Literal, Dict, Any, Union
from enum import Enum
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import yt_dlp
import os
import time
import random
import google.generativeai as genai
from dotenv import load_dotenv
import json
import uuid
import datetime
import re
import nltk
from nltk.tokenize import sent_tokenize
load_dotenv()
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

app = FastAPI(
    title="VidSummarizer",
    description="API for transcribing, summarizing, and creating quizzes from YouTube videos using Transformers and Google Gemini"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelManager:
    def __init__(self):
        self.whisper_model = None
        self.tokenizer = None
        self.summarization_model = None
        self.gemini_model = None
        self._initialize_models()

    def _initialize_models(self):
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base", device=device)
        print("Loading summarization model...")
        model_name = "sshleifer/distilbart-cnn-12-6"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarization_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            genai.configure(api_key=google_api_key)
            self.gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')
        else:
            print("WARNING: Google Gemini API key is missing. Quiz generation will not work.")
            self.gemini_model = None
    
    def get_whisper_model(self):
        return self.whisper_model
    def get_summarization_model(self):
        return self.summarization_model, self.tokenizer
    def get_gemini_model(self):
        return self.gemini_model
    
model_manager = ModelManager()

class QuizType(str, Enum):
    FILL_IN_THE_BLANK = "fill_in_the_blank"
    MULTIPLE_CHOICE = "multiple_choice"
    BOTH = "both"
class PDFType(str, Enum):
    SUMMARY_ONLY = "summary_only"
    SUMMARY_AND_QUIZ = "summary_and_quiz"
class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
class SummarizationRequest(BaseModel):
    youtube_url: str = Field(..., description="YouTube video URL to process")
    max_length: int = Field(150, description="Maximum token length for summary")
    min_length: int = Field(100, description="Minimum token length for summary")
class SummarizationResponse(BaseModel):
    original_text: str
    summary: str
    video_title: str
class FillInTheBlankQuestion(BaseModel):
    question_text: str
    answer: str
class MultipleChoiceOption(BaseModel):
    option: str
    is_correct: bool
class MultipleChoiceQuestion(BaseModel):
    question_text: str
    options: List[MultipleChoiceOption]
class QuizRequest(BaseModel):
    youtube_url: str
    quiz_type: QuizType = QuizType.BOTH
    num_questions: int = Field(5, ge=1, le=20, description="Number of questions to generate (1-20)")
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
class QuizResponse(BaseModel):
    video_title: str
    fill_in_the_blank_questions: Optional[List[FillInTheBlankQuestion]] = None
    multiple_choice_questions: Optional[List[MultipleChoiceQuestion]] = None
class PDFRequest(BaseModel):
    youtube_url: str
    pdf_type: PDFType = PDFType.SUMMARY_ONLY
    quiz_type: Optional[QuizType] = QuizType.BOTH
    num_questions: Optional[int] = Field(5, ge=1, le=20)
    difficulty: Optional[DifficultyLevel] = DifficultyLevel.MEDIUM
    max_summary_length: int = Field(150, ge=50, le=500)
    min_summary_length: int = Field(50, ge=30, le=300)

class YouTubeService: 
    @staticmethod
    def download_youtube_audio(url: str, max_retries=3) -> tuple:
        os.makedirs('downloads', exist_ok=True)
        for attempt in range(max_retries):
            try:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': 'downloads/%(title)s.%(ext)s',
                    'noplaylist': True,
                    'quiet': True,
                    'no_warnings': True,
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    # Handle playlists (though we try to avoid them with noplaylist option)
                    if 'entries' in info:
                        info = info['entries'][0]
                    filename = ydl.prepare_filename(info)
                    # Replace extension with mp3 (due to the audio extraction)
                    filename = os.path.splitext(filename)[0] + '.mp3'
                    if not os.path.exists(filename):
                        raise HTTPException(status_code=500, detail="File not found after download")
                    if os.path.getsize(filename) == 0:
                        raise HTTPException(status_code=500, detail="Downloaded file is empty")  
                    return filename, info.get('title', 'Unknown Title')
                    
            except HTTPException:
                raise    
            except Exception as e:
                if attempt < max_retries - 1:
                    # Wait before retrying with exponential backoff
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                    continue
                else:
                    raise HTTPException(status_code=400, detail=f"Error downloading audio after {max_retries} attempts: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected error in download function")
class TranscriptionService:
    @staticmethod
    def transcribe_audio(audio_path: str) -> str:
        try:
            if not os.path.exists(audio_path):
                raise HTTPException(status_code=500, detail="Audio file not found for transcription")
            if os.path.getsize(audio_path) == 0:
                raise HTTPException(status_code=500, detail="Audio file is empty")
            whisper_model = model_manager.get_whisper_model()
            result = whisper_model.transcribe(audio_path)
            return result['text']
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
class SummarizationService:
    @staticmethod
    def chunk_and_summarize(text, max_length=150, min_length=50, max_chunk_size=1000):
        text = text.replace("\n", " ").strip()
        text = re.sub(r'\s+', ' ', text)
        summarization_model, tokenizer = model_manager.get_summarization_model()
        sentences = sent_tokenize(text)
        
        # Create chunks based on sentences rather than arbitrary token counts
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Tokenize each sentence and combine them into chunks that respect max_chunk_size
        for sentence in sentences:
            # Tokenize to get accurate token count
            sentence_tokens = tokenizer(sentence, add_special_tokens=False)["input_ids"]
            sentence_length = len(sentence_tokens)
            
            # If adding this sentence would exceed max_chunk_size, start a new chunk
            if current_length + sentence_length > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # If there's only one chunk, summarize it directly
        if len(chunks) == 1:
            return SummarizationService.generate_summary(chunks[0], max_length, min_length)
        
        # Otherwise, summarize each chunk and then combine them
        chunk_summaries = []
        for chunk in chunks:
            # Adjust length constraints based on number of chunks
            chunk_max_length = max(100, int(max_length / len(chunks) * 1.5))
            chunk_min_length = max(30, int(min_length / len(chunks) * 0.8))
            
            chunk_summary = SummarizationService.generate_summary(chunk, chunk_max_length, chunk_min_length)
            chunk_summaries.append(chunk_summary)
        
        # Combine chunk summaries and create final summary
        combined_chunks = " ".join(chunk_summaries)
        final_summary = SummarizationService.generate_summary(combined_chunks, max_length, min_length)
        return SummarizationService.post_process_summary(final_summary)
    
    @staticmethod
    def generate_summary(text, max_length=150, min_length=50):
        # Preprocess text
        text = text.replace("\n", " ").strip()
        text = re.sub(r'\s+', ' ', text)
        # Ensure max and min lengths are reasonable
        max_length = max(min_length + 10, max_length)
        # Get model and tokenizer
        summarization_model, tokenizer = model_manager.get_summarization_model()
        # Tokenize with proper handling
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Generate summary with improved parameters
        with torch.no_grad():
            summary_ids = summarization_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=5,  # Increased from 4 for better quality
                length_penalty=1.5,  # Adjusted from 2.0
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=False,  # For deterministic outputs
            )
        
        # Decode and return
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()
    
    @staticmethod
    def post_process_summary(summary):
        # Fix any uncapitalized sentences
        summary = re.sub(r'([.!?])\s*([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), summary)
        
        # Remove duplicate sentences
        sentences = sent_tokenize(summary)
        unique_sentences = []
        
        for sentence in sentences:
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            is_duplicate = False
            for existing in unique_sentences:
                existing_normalized = re.sub(r'\s+', ' ', existing.lower().strip())
                if normalized == existing_normalized:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_sentences.append(sentence)
        processed_summary = " ".join(unique_sentences)
        if processed_summary and not processed_summary.endswith(('.', '!', '?')):
            processed_summary += '.'
            
        return processed_summary
class QuizService:
    @staticmethod
    async def verify_api_key():
        if not os.getenv("GOOGLE_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="Google Gemini API key not configured. Set the GOOGLE_API_KEY environment variable."
            )
    
    @staticmethod
    def generate_quiz(transcribed_text: str, quiz_type: QuizType, num_questions: int, difficulty: str) -> dict:
        try:
            max_chars = 30000
            transcribed_text = transcribed_text[:max_chars]
            model = model_manager.get_gemini_model()
            if not model:
                raise HTTPException(status_code=500, detail="Gemini model not initialized - API key missing")
            if quiz_type == QuizType.FILL_IN_THE_BLANK:
                prompt = f"""
                Create {num_questions} fill-in-the-blank questions at {difficulty} difficulty level based on this content:
                
                {transcribed_text}
                
                Format each question as a JSON object with 'question_text' (include a blank as '___') and 'answer' fields.
                Return only a JSON array of these questions, no other text.
                Example format:
                [
                    {{
                        "question_text": "The capital of France is ___.",
                        "answer": "Paris"
                    }},
                    ...
                ]
                """
                
                response = model.generate_content(prompt)
                fill_in_blank_questions = QuizService.parse_json_response(response.text)
                return {"fill_in_the_blank_questions": fill_in_blank_questions}
            elif quiz_type == QuizType.MULTIPLE_CHOICE:
                prompt = f"""
                Create {num_questions} multiple-choice questions at {difficulty} difficulty level based on this content:
                
                {transcribed_text}
                
                Format each question as a JSON object with 'question_text' and 'options' fields. The 'options' field should be an array of objects, each with 'option' (the text) and 'is_correct' (boolean) fields. Each question should have exactly one correct answer and three incorrect answers.
                Return only a JSON array of these questions, no other text.
                Example format:
                [
                    {{
                        "question_text": "What is the capital of France?",
                        "options": [
                            {{ "option": "Paris", "is_correct": true }},
                            {{ "option": "London", "is_correct": false }},
                            {{ "option": "Berlin", "is_correct": false }},
                            {{ "option": "Madrid", "is_correct": false }}
                        ]
                    }},
                    ...
                ]
                """
                
                response = model.generate_content(prompt)
                mc_questions = QuizService.parse_json_response(response.text)
                return {"multiple_choice_questions": mc_questions}
                
            else:  # Both types
                # Split questions between the two types
                fill_count = num_questions // 2
                mc_count = num_questions - fill_count
                
                fill_prompt = f"""
                Create {fill_count} fill-in-the-blank questions at {difficulty} difficulty level based on this content:
                
                {transcribed_text}
                
                Format each question as a JSON object with 'question_text' (include a blank as '___') and 'answer' fields.
                Return only a JSON array of these questions, no other text.
                Example format:
                [
                    {{
                        "question_text": "The capital of France is ___.",
                        "answer": "Paris"
                    }},
                    ...
                ]
                """
                
                mc_prompt = f"""
                Create {mc_count} multiple-choice questions at {difficulty} difficulty level based on this content:
                
                {transcribed_text}
                
                Format each question as a JSON object with 'question_text' and 'options' fields. The 'options' field should be an array of objects, each with 'option' (the text) and 'is_correct' (boolean) fields. Each question should have exactly one correct answer and three incorrect answers.
                Return only a JSON array of these questions, no other text.
                Example format:
                [
                    {{
                        "question_text": "What is the capital of France?",
                        "options": [
                            {{ "option": "Paris", "is_correct": true }},
                            {{ "option": "London", "is_correct": false }},
                            {{ "option": "Berlin", "is_correct": false }},
                            {{ "option": "Madrid", "is_correct": false }}
                        ]
                    }},
                    ...
                ]
                """
                
                fill_response = model.generate_content(fill_prompt)
                mc_response = model.generate_content(mc_prompt)
                
                fill_in_blank_questions = QuizService.parse_json_response(fill_response.text)
                mc_questions = QuizService.parse_json_response(mc_response.text)
                
                return {
                    "fill_in_the_blank_questions": fill_in_blank_questions,
                    "multiple_choice_questions": mc_questions
                }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating quiz questions: {str(e)}")
    
    @staticmethod
    def parse_json_response(response_text: str) -> list:
        # Clean the response text to ensure it's valid JSON
        cleaned_text = response_text.strip()
        # Find the first '[' and last ']' to extract just the JSON array
        start_idx = cleaned_text.find('[')
        end_idx = cleaned_text.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            cleaned_text = cleaned_text[start_idx:end_idx+1]
        
        # Remove any markdown code block formatting
        cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            # Try to fix common JSON errors
            try:
                # Replace single quotes with double quotes
                cleaned_text = cleaned_text.replace("'", '"')
                # Try parsing again
                return json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to parse Gemini response as JSON: {str(e)}"
                )
class PDFService:
    @staticmethod
    def create_pdf(filename, data):
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=16,
            spaceAfter=12
        )
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15
        )
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8
        )
        
        elements = []
        
        elements.append(Paragraph(f"Video Summary: {data['video_title']}", title_style))
        
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Created on: {current_date}", normal_style))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("Summary", heading_style))
        elements.append(Paragraph(data['summary'], normal_style))
        elements.append(Spacer(1, 12))
        
        # Add quiz section if requested
        if 'quiz' in data:
            elements.append(Paragraph("Quiz Questions", heading_style))
            
            # Add fill-in-the-blank questions if available
            if data['quiz'].get('fill_in_the_blank_questions'):
                elements.append(Paragraph("Fill in the Blank Questions:", ParagraphStyle(
                    'Subheading',
                    parent=styles['Heading3'],
                    fontSize=12,
                    spaceAfter=8
                )))
                
                for i, q in enumerate(data['quiz']['fill_in_the_blank_questions']):
                    elements.append(Paragraph(f"{i+1}. {q['question_text']}", normal_style))
                    elements.append(Paragraph(f"Answer: {q['answer']}", ParagraphStyle(
                        'Answer',
                        parent=normal_style,
                        leftIndent=20,
                        textColor=colors.blue
                    )))
                    elements.append(Spacer(1, 8))
            
            if data['quiz'].get('multiple_choice_questions'):
                elements.append(Paragraph("Multiple Choice Questions:", ParagraphStyle(
                    'Subheading',
                    parent=styles['Heading3'],
                    fontSize=12,
                    spaceAfter=8
                )))
                
                for i, q in enumerate(data['quiz']['multiple_choice_questions']):
                    elements.append(Paragraph(f"{i+1}. {q['question_text']}", normal_style))
                    
                    # Format options as a table
                    option_data = []
                    for j, opt in enumerate(q['options']):
                        option_letter = chr(65 + j)  # A, B, C, D
                        is_correct = "✓" if opt['is_correct'] else ""
                        option_data.append([f"{option_letter}. {opt['option']}", is_correct])
                    
                    option_table = Table(option_data, colWidths=[400, 30])
                    option_table.setStyle(TableStyle([
                        ('TEXTCOLOR', (1, 0), (1, -1), colors.green),
                        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
                        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                        ('LEFTPADDING', (0, 0), (0, -1), 20),
                    ]))
                    
                    elements.append(option_table)
                    elements.append(Spacer(1, 12))
        
        # Add original transcript section
        elements.append(Paragraph("Full Transcript", heading_style))
        
        # Break the transcript into paragraphs
        paragraphs = data['transcript'].split('\n\n')
        for para in paragraphs:
            if para.strip():
                elements.append(Paragraph(para, normal_style))
                elements.append(Spacer(1, 6))
        
        # Build the PDF
        doc.build(elements)
        return filename
@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_youtube_video(request: SummarizationRequest):
    audio_path = None
    try:
        if request.max_length <= request.min_length:
            raise HTTPException(status_code=400, detail="max_length must be greater than min_length")
        audio_path, video_title = YouTubeService.download_youtube_audio(request.youtube_url)
        
        # Transcribe audio
        transcribed_text = TranscriptionService.transcribe_audio(audio_path)

        # Validate transcription result
        if not transcribed_text or not transcribed_text.strip():
            raise HTTPException(status_code=400, detail="Transcription is empty or invalid")

        # Generate summary using improved summarization algorithm
        try:
            summary = SummarizationService.chunk_and_summarize(
                transcribed_text,
                max_length=request.max_length,
                min_length=request.min_length
            )
            
            # Validate summary content
            if not summary or not summary.strip():
                raise HTTPException(status_code=500, detail="Summarization produced empty result")
                
        except Exception as summarization_error:
            raise HTTPException(status_code=500, detail=f"Summarization failed: {str(summarization_error)}")
        return SummarizationResponse(
            original_text=transcribed_text,
            summary=summary,
            video_title=video_title
        )
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
        
    except Exception as e:
        # Convert any exception to an HTTPException
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up the downloaded audio
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")

@app.post("/generate-quiz", response_model=QuizResponse, dependencies=[Depends(QuizService.verify_api_key)])
async def generate_quiz(request: QuizRequest):
    audio_path = None
    try:
        # Download audio
        audio_path, video_title = YouTubeService.download_youtube_audio(request.youtube_url)
        
        # Transcribe audio
        transcribed_text = TranscriptionService.transcribe_audio(audio_path)
        
        # Generate quiz questions
        quiz_data = QuizService.generate_quiz(
            transcribed_text,
            request.quiz_type,
            request.num_questions,
            request.difficulty
        )
        
        # Create response
        response = QuizResponse(
            video_title=video_title,
            fill_in_the_blank_questions=quiz_data.get("fill_in_the_blank_questions"),
            multiple_choice_questions=quiz_data.get("multiple_choice_questions")
        )
        return response
    
    except Exception as e:
        # Convert any exception to HTTPException for proper API response
        if not isinstance(e, HTTPException):
            e = HTTPException(status_code=500, detail=str(e))
        raise e
    
    finally:
        # Clean up downloaded audio
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                # Just log cleanup failures, don't fail the request
                pass

@app.post("/generate-pdf")
async def generate_pdf(request: PDFRequest):
    audio_path = None
    pdf_path = None
    
    try:
        # Download audio
        audio_path, video_title = YouTubeService.download_youtube_audio(request.youtube_url)
        
        # Transcribe audio with CUDA acceleration
        transcribed_text = TranscriptionService.transcribe_audio(audio_path)
        
        # Use our transformer-based summarizer for better results
        summary = SummarizationService.chunk_and_summarize(
            transcribed_text, 
            max_length=request.max_summary_length, 
            min_length=request.min_summary_length
        )
        
        # Create unique filename in the downloads folder
        os.makedirs('downloads', exist_ok=True)
        file_id = str(uuid.uuid4())[:8]
        sanitized_title = ''.join(c if c.isalnum() else '_' for c in video_title)[:30]
        pdf_path = f"downloads/{sanitized_title}_{file_id}.pdf"
        
        # Prepare data for PDF generation
        pdf_data = {
            'video_title': video_title,
            'summary': summary,
            'transcript': transcribed_text
        }
        
        # Generate quiz if requested
        if request.pdf_type == PDFType.SUMMARY_AND_QUIZ:
            if not request.quiz_type or not request.num_questions or not request.difficulty:
                raise HTTPException(status_code=400, detail="Quiz parameters must be provided for PDF with quiz")
                
            # Generate quiz questions
            quiz_data = QuizService.generate_quiz(
                transcribed_text,
                request.quiz_type,
                request.num_questions,
                request.difficulty
            )
            
            pdf_data['quiz'] = quiz_data
        
        # Create PDF file
        pdf_file = PDFService.create_pdf(pdf_path, pdf_data)
        
        # Return the file
        return FileResponse(
            path=pdf_file,
            filename=f"{sanitized_title}.pdf",
            media_type="application/pdf"
        )
    except Exception as e:
        # Convert any exception to HTTPException for proper API response
        if not isinstance(e, HTTPException):
            e = HTTPException(status_code=500, detail=str(e))
        raise e
    
    finally:
        # Clean up downloaded audio
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                # Just log cleanup failures, don't fail the request
                pass
@app.get("/health")
async def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda
        }
    else:
        gpu_info = {"gpu_available": False}
        
    return {
        "status": "healthy", 
        "gpu_info": gpu_info,
        "gemini_api_configured": bool(os.getenv("GOOGLE_API_KEY"))
    }