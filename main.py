import os
import json
import re
import warnings
from io import BytesIO
from typing import List, Dict, Any, Tuple

import certifi
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from nltk.tokenize import sent_tokenize
# from fastapi import File, UploadFile
# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# ==================== Configuration ====================
class Config:
    # Load from .env
    GOOGLE_API_KEY: str = os.getenv('GOOGLE_API_KEY', '')
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
    # DOCUMENT_URLS may be comma-separated or JSON list
    _raw_urls: str = os.getenv('DOCUMENT_URLS', '')
    DOCUMENT_URLS: List[str] = [
        "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
        "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/CHOTGDP23004V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
        "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/EDLHLGA23009V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
        "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/HDFHLIP23024V072223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
        "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/ICIHLIP22012V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
        "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    ]
    if _raw_urls:
        # try comma-separated first
        DOCUMENT_URLS = [u.strip() for u in _raw_urls.split(',') if u.strip()]
        # if that yields a single element starting with '[', try JSON
        if len(DOCUMENT_URLS) == 1 and DOCUMENT_URLS[0].startswith('['):
            try:
                DOCUMENT_URLS = json.loads(DOCUMENT_URLS[0])
            except json.JSONDecodeError:
                pass
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', '500'))
    CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', '50'))
    TOP_K_RETRIEVAL: int = int(os.getenv('TOP_K_RETRIEVAL', '5'))
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.3'))
# ==================== Authentication ====================
security = HTTPBasic()
# In-memory user store
USERS: Dict[str, Dict[str, str]] = {}
ADMIN_USER = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASS = os.getenv('ADMIN_PASSWORD', 'adminpass')
USERS[ADMIN_USER] = { 'password': ADMIN_PASS, 'role': 'admin' }
EMPLOYEE_USER = os.getenv('EMPLOYEE_USERNAME', 'employee')
EMPLOYEE_PASS = os.getenv('EMPLOYEE_PASSWORD', 'employeepass')
USERS[EMPLOYEE_USER] = { 'password': EMPLOYEE_PASS, 'role': 'employee' }

# Auth dependency
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    user = USERS.get(credentials.username)
    if not user or user['password'] != credentials.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"}
        )
    return {'username': credentials.username, 'role': user['role']}

# ==================== FastAPI App ====================
app = FastAPI(title="Document Processing RAG Agent API")


def _summarize_chunks(chunks: list[str], focus_keyword: str) -> str:
    sentences = [s for chunk in chunks for s in sent_tokenize(chunk)]
    # 1) Prefer sentences that mention both the focus and a time unit
    for s in sentences:
        if focus_keyword in s.lower() and any(u in s.lower() for u in ("month", "year")):
            return s.strip()
    # 2) Otherwise, fallback to first sentence with the focus keyword
    for s in sentences:
        if focus_keyword in s.lower():
            return s.strip()
    # 3) Last resort: first sentence at all
    return sentences[0].strip() if sentences else ""
# ==================== Pydantic Models ====================
class QueryRequest(BaseModel):
    query: str

class UserCreate(BaseModel):
    username: str
    password: str
    role: str  # "admin" or "employee"

# ==================== Core Classes ====================
# DocumentProcessor
import PyPDF2
from docx import Document as DocxDocument

class DocumentProcessor:
    def __init__(self):
        pass

    def download_pdf(self, url: str) -> bytes:
        try:
            resp = requests.get(url, timeout=30, verify=certifi.where())
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
            return None

    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            text = ""
            for p in reader.pages:
                text += p.extract_text() or ''
            return text
        except Exception as e:
            print(f"❌ Error extracting PDF text: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.,;:!?\-()\[\]]', ' ', text)
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i+chunk_size])
            if len(chunk) > 50:
                chunks.append(chunk)
        return chunks

    def process_documents(self, urls: List[str]) -> Dict[str, List[str]]:
        all_chunks = {}
        for i, url in enumerate(urls):
            pdf = self.download_pdf(url)
            if not pdf:
                continue
            txt = self.extract_text_from_pdf(pdf)
            if not txt:
                continue
            clean = self.clean_text(txt)
            chunks = self.chunk_text(clean, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
            all_chunks[f"doc_{i+1}"] = chunks
        return all_chunks

# QueryParser
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

from word2number import w2n

class QueryParser:
    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
            self.spacy = True
        except:
            self.nlp = None
            self.spacy = False

        self.number_map = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
            'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
            'ninety': 90
        }
        self.age_re = re.compile(r'\b(\d+)\s*(?:years?|yrs?|y)?', re.I)
        self.gender_re = re.compile(r'\b(male|female|m|f)\b', re.I)
        self.proc_re = re.compile(
            r'\b([A-Za-z\- ]+?(?:surgery|operation|treatment|procedure|therapy))\b',
            re.I
        )
        self.loc_patterns = [r'in ([A-Z][a-z]+)', r'at ([A-Z][a-z]+)', r'from ([A-Z][a-z]+)']
        self.pol_re = re.compile(r'\b(\d+)\s*(?:months?|years?|days?)\b', re.I)

    def _norm_nums(self, txt: str) -> str:
        for w, n in self.number_map.items():
            txt = re.sub(rf'\b{w}\b', str(n), txt, flags=re.I)
        return txt

    def extract_age(self, q: str) -> Any:
        txt = self._norm_nums(q)
        if self.spacy:
            for ent in self.nlp(txt).ents:
                if ent.label_ == 'DATE' and 'year' in ent.text.lower():
                    m = re.search(r'(\d+)', ent.text)
                    if m:
                        return int(m.group(1))
        m = self.age_re.search(txt)
        return int(m.group(1)) if m else None

    def extract_gender(self, q: str) -> Any:
        m = self.gender_re.search(q)
        if not m:
            return None
        g = m.group(1).lower()
        if g in ('male', 'm'):
            return 'Male'
        if g in ('female', 'f'):
            return 'Female'
        return None

    def extract_procedure(self, query: str) -> str | None:
    
        keywords = ("surgery", "operation", "treatment", "procedure", "therapy",
                    "consultation", "checkup", "diagnosis", "examination","emergency","suffer","injure")

        # 1) spaCy noun‐chunks
        if self.spacy and self.nlp:
            doc = self.nlp(query)
            for chunk in doc.noun_chunks:
                txt = chunk.text.strip()
                if any(kw in txt.lower() for kw in keywords):
                    return txt

        # 2) exact‐match regex fallback
        m = self.proc_re.search(query)
        if m:
            return m.group(1).strip()

        # 3) sentence‐level fallback
        for sent in nltk.sent_tokenize(query):
            if any(kw in sent.lower() for kw in keywords):
                return sent.strip()

        # 4) “<keyword> of <object>” pattern
        m2 = re.search(rf"\b({'|'.join(keywords)}) of ([A-Za-z ]+)", query, re.IGNORECASE)
        if m2:
            return f"{m2.group(1)} of {m2.group(2).strip()}"

        return None

    def extract_location(self, query: str) -> str:
        gender_keywords = {"male","female","m","f","he","she","they"}

        # 1) Try spaCy GPE/LOC
        if self.spacy and self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                if ent.label_ in ("GPE","LOC"):
                    val = ent.text.strip()
                    if val.lower() not in gender_keywords:
                        return val

        # 2) Regex fallback for “in/at/from X”
        for p in (
            r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        ):
            m = re.search(p, query)
            if m:
                return m.group(1)

        # 3) Capitalized‐words fallback (now includes single words)
        seqs = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query)
        # drop any gender terms
        seqs = [s for s in seqs if s.lower() not in gender_keywords]
        if seqs:
            # prefer multi-word, else single‐word
            return max(seqs, key=lambda s: len(s.split()))

        return None


    def extract_policy_duration(self, q: str) -> dict | None:
    
        # 1) “<n> <unit> old policy”
        m = re.search(
            r"(\d+)\s*(months?|years?|days?)\s+old\s+policy",
            q,
            flags=re.IGNORECASE,
        )
        if m:
            return {"value": int(m.group(1)), "unit": m.group(2)}

        # 2) “policy … <n> <unit>”
        m2 = re.search(
            r"policy.*?(\d+)\s*(months?|years?|days?)",
            q,
            flags=re.IGNORECASE,
        )
        if m2:
            return {"value": int(m2.group(1)), "unit": m2.group(2)}

        # 3) fallback: first standalone <n> <unit>
        txt = self._norm_nums(q)
        m3 = self.pol_re.search(txt)
        if m3:
            return {"value": int(m3.group(1)), "unit": m3.group(0).split()[-1]}

        return None


    def parse_query(self, q: str) -> Dict[str, Any]:
        return {
            'original_query':     q,
            'age':                self.extract_age(q),
            'gender':             self.extract_gender(q),
            'procedure':          self.extract_procedure(q),
            'location':           self.extract_location(q),
            'policy_duration':    self.extract_policy_duration(q),
        }

# SemanticSearchEngine with FAISS fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except:
    from sklearn.metrics.pairwise import cosine_similarity
    FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer

class SemanticSearchEngine:
    def __init__(self):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        self.meta = []

    def build_index(self, docs: Dict[str,List[str]]):
        all_chunks, meta = [], []
        for did, chs in docs.items():
            for i, c in enumerate(chs):
                all_chunks.append(c)
                meta.append({'doc_id':did,'chunk_id':i,'text':c})
        if not all_chunks: return
        embs = self.model.encode(all_chunks)
        if FAISS_AVAILABLE:
            embs = np.ascontiguousarray(embs.astype('float32'))
            faiss.normalize_L2(embs)
            self.index = faiss.IndexFlatIP(embs.shape[1])
            self.index.add(embs)
        else:
            self.index = embs
        self.chunks, self.meta = all_chunks, meta

    def search(self, query: str, k: int) -> List[Dict[str,Any]]:
        if self.index is None: return []
        qemb = self.model.encode([query])
        if FAISS_AVAILABLE:
            qemb = np.ascontiguousarray(qemb.astype('float32'))
            faiss.normalize_L2(qemb)
            scores, idxs = self.index.search(qemb, k)
            res = []
            for s, i in zip(scores[0], idxs[0]):
                if s >= Config.SIMILARITY_THRESHOLD:
                    m = self.meta[i]
                    res.append({**m,'similarity_score':float(s)})
            return res
        else:
            sims = cosine_similarity(qemb, self.index)[0]
            idxs = sims.argsort()[::-1][:k]
            return [{'text':self.chunks[i],'doc_id':self.meta[i]['doc_id'],'chunk_id':self.meta[i]['chunk_id'],'similarity_score':float(sims[i])}
                    for i in idxs if sims[i]>=Config.SIMILARITY_THRESHOLD]

# LLMDecisionEngine
import google.generativeai as genai

genai.configure(api_key=Config.GOOGLE_API_KEY)

class LLMDecisionEngine:
    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name=Config.GEMINI_MODEL,
            generation_config={"response_mime_type":"application/json"}
        )


    def create_information_prompt(self, query: str, chunks: List[dict]) -> str:
        """
        Ask *only* for a JSON object with two keys:
        - summary: a one- or two-sentence answer
        - sources: array of "doc_id:chunk_id"
        """
    # build a short context block (all retrieved chunks)
        chunk_data = "\n".join(f"--- {c['doc_id']}:{c['chunk_id']} ---\n{c['text']}"
                            for c in chunks)

        return f"""
            You are a professional insurance policy analyst. Answer succinctly **only** about the **pre-existing diseases (PED)** waiting period, ignoring any other waiting‐period references.

            Clauses:(make sure to emphasise on important time durations such as expiration timelines etc.)
            {chunk_data}

            User Query:
            {query}

            Please respond with JSON exactly:
            {{
            "summary": "<brief answer>",
            "sources": ["<doc_id:chunk_id>", ...]
            }}
            (no extra text)
            """    


    def extract_json(self, txt: str):
        m = re.search(r'\{.*\}', txt, re.DOTALL)
        if not m: return None
        try: return json.loads(m.group())
        except: return None

    def fallback_decision(self, info, chunks):
        proc = info.get('procedure','').lower() if info.get('procedure') else ''
        if any(k in proc for k in ['surgery','treatment']):
            return {'decision':'approved','amount':'50000','confidence':'medium',
                    'justification':'Fallback approval','referenced_clauses':[],
                    'waiting_period_status':'not_applicable','additional_notes':'fallback'}
        return {'decision':'rejected','amount':None,'confidence':'medium',
                'justification':'Fallback rejection','referenced_clauses':[],
                'waiting_period_status':'not_applicable','additional_notes':'fallback'}

    def make_decision(self, info, chunks):
        prompt = self.create_decision_prompt(info, chunks)
        res = self.model.generate_content(prompt)
        data = self.extract_json(res.text)
        return data or self.fallback_decision(info, chunks)

    def make_information(self, query: str, chunks: list[dict]) -> dict:
        prompt = self.create_information_prompt(query, chunks)
        resp = self.model.generate_content(prompt)
        text = resp.text.strip()

        # 1) Try JSON parsing
        parsed = self.extract_json(text)
        if parsed:
            return parsed

        # 2) If JSON failed, **summarize the actual chunks** instead of echoing resp.text
        raw_chunks = [c["text"] for c in chunks]
        summary = _summarize_chunks(raw_chunks)

        return {
            "summary": summary,
            "sources": [f"{c['doc_id']}:{c['chunk_id']}" for c in chunks]
        }
    
    def answer_question_direct(self, question: str, relevant_chunks: list) -> str:
        """
        Ask Gemini for a direct, professional answer to a factual question, given context.
        """
        context = "\n".join([f"- {chunk['text']}" for chunk in relevant_chunks])
        prompt = f"""You are an expert insurance policy FAQ bot.

    Goal: Answer the following customer question about the policy using ONLY the provided excerpts.
    - If the exact answer is stated in the text, quote or paraphrase it concisely.
    - Do NOT add information not present in the context.
    - If the answer is NOT present, reply: "The policy document does not specify this information."
    - Do not speculate, hedge, or reference other products. Do not repeat the question. Answer ONLY for the provided policy.
    - Format the answer as a single clear sentence or paragraph, without disclaimers or apologies.

    QUESTION:
    {question}

    EXCERPTS FROM POLICY DOCUMENT:
    {context}

    INSTRUCTION: Answer as above."""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"❌ Error from LLM: {e}")
            return "The policy document does not specify this information."

# DocumentQuerySystem
class DocumentQuerySystem:
    def __init__(self):
        self.dp = DocumentProcessor()
        self.qp = QueryParser()
        self.ss = SemanticSearchEngine()
        self.de = LLMDecisionEngine()
        self.inited = False

    def initialize(self) -> bool:
        docs = self.dp.process_documents(Config.DOCUMENT_URLS)
        if not docs: return False
        self.ss.build_index(docs)
        self.inited = True
        return True

    def process_query(self, q: str) -> Tuple[Dict[str, Any], int]:
        """
        Process a user query:
        - Parse out age, gender, procedure, location, policy_duration
        - If any required fields are missing, return follow-up prompts (status=1)
        - Otherwise run semantic search, invoke the LLM, and return:
            {
                "query_info": {...},
                "decision": "approved"|"rejected",
                "decision_details": { full JSON from LLM },
                "relevant_chunks": [ {doc_id,chunk_id,text,score}, ... ]
            }, 0
        """
    # 1) Ensure initialization
        if not self.inited:
            return {"error": "Service not initialized"}, 1

        # 2) Parse the query
        info = self.qp.parse_query(q)

        # 3) Identify any missing fields (except original_query)
        missing = [field for field, val in info.items() 
                if field != "original_query" and val is None]
        if missing:
            # Ask for each missing piece explicitly
            prompts = {
                "age": "What is the patient’s age (in years)?",
                "gender": "Please specify the patient’s gender (Male/Female).",
                "procedure": "Which procedure or treatment is requested?",
                "location": "In which city or location is the treatment needed?",
                "policy_duration": "How long has the policy been active (e.g. “2 years old policy”)?"
            }
            return (
                [{"field": f, "prompt": prompts[f]} for f in missing],
                1
            )

        # 4) Run semantic search
        chunks = self.ss.search(q, Config.TOP_K_RETRIEVAL)

        # 5) Make the LLM decision
        full_decision = self.de.make_decision(info, chunks)

        # 6) Extract the one-word verdict
        verdict = full_decision.get("decision")
        if verdict is None:
            # Shouldn't happen if LLM JSON is valid, but guard anyway
            verdict = "undetermined"

        # 7) Return everything
        return ({
            "query_info":       info,
            "decision":         verdict,         # "approved" or "rejected"
            "decision_details": full_decision,   # the full structured JSON from Gemini
            "relevant_chunks":  chunks           # raw chunk objects with doc_id, chunk_id, text, score
        }, 0)

    def process_query_for_information(self, query: str) -> Tuple[Dict[str, Any], int]:
        if not self.inited:
            return {"error": "Service not initialized"}, 503

        # 1) Retrieve all relevant chunks (you said these are perfect already)
        hits = self.ss.search(query, Config.TOP_K_RETRIEVAL)
        if not hits:
            return {"error": "No relevant information found"}, 404

        # 2) Ask the LLM for a clean summary
        info = self.de.make_information(query, hits)

        # 3) Return exactly the summary + the chunks that produced it
        return ({
            "answer": info.get("summary"),
            "relevant_clauses": [c["text"] for c in hits],
            "sources":         [f"{c['doc_id']}:{c['chunk_id']}" for c in hits]
        }, status.HTTP_200_OK)
    

    def answer_questions(self, questions: list) -> dict:   
    
        if not self.inited:
            return {"error": "Service not initialized"}, 503
        
        answers = []
        
        for q in questions:

            # 1) Retrieve all relevant chunks (you said these are perfect already)
            hits = self.ss.search(q, Config.TOP_K_RETRIEVAL)
            if not hits:
                return {"error": "No relevant information found"}, 404

            # 2) Ask the LLM for a clean summary
            info = self.de.make_information(q, hits)
            answers.append(info.get("summary"))

            # 3) Return exactly the summary + the chunks that produced it
            return (
                {"answer": answers}, status.HTTP_200_OK)

# ==================== App Startup ====================
@app.on_event('startup')
def on_startup():
    if not Config.GOOGLE_API_KEY or 'YOUR_KEY' in Config.GOOGLE_API_KEY:
        raise RuntimeError('GOOGLE_API_KEY not set')
    global system
    system = DocumentQuerySystem()
    if not system.initialize():
        print("❌ WARNING: Document ingestion failed at startup; API will still run for debugging.")
# ==================== Endpoints ====================
@app.post('/auth/login')
def login(creds: HTTPBasicCredentials = Depends(security)):
    user = USERS.get(creds.username)
    if not user or user['password']!=creds.password:
        raise HTTPException(401,'Invalid credentials')
    return {'username':creds.username,'role':user['role']}

@app.post('/auth/logout')
def logout():
    return {'message':'Logged out'}


@app.post('/admin/users', status_code=201)
def create_user(u: UserCreate, cur=Depends(get_current_user)):
    if cur['role']!='admin': raise HTTPException(403,'Admin required')
    if u.username in USERS: raise HTTPException(400,'Exists')
    if u.role not in ('admin','employee'): raise HTTPException(400,'Bad role')
    USERS[u.username]={'password':u.password,'role':u.role}
    return {'username':u.username,'role':u.role}

@app.get('/admin/users')
def list_users(cur=Depends(get_current_user)):
    if cur['role']!='admin': raise HTTPException(403,'Admin required')
    return [{'username':k,'role':v['role']} for k,v in USERS.items()]

@app.post('/admin/ingest')
def ingest(cur=Depends(get_current_user)):
    if cur['role']!='admin': raise HTTPException(403,'Admin required')
    ok = system.initialize()
    if not ok: raise HTTPException(500,'Ingest failed')
    return {'message':'Ingestion complete'}


# @app.post('/admin/ingest/upload', dependencies=[Depends(admin_required)], tags=['Admin'])
# async def ingest_upload(files: List[UploadFile] = File(...)):
#     """
#     Upload one or more PDF files and ingest them into the RAG engine, merged with existing documents.
#     """
#     # process existing remote docs
#     dp = system.dp
#     all_chunks = dp.process_documents(Config.DOCUMENT_URLS)

#     # process uploaded files
#     for upload in files:
#         content = await upload.read()
#         text = dp.extract_text_from_pdf(content)
#         if not text:
#             continue
#         clean = dp.clean_text(text)
#         chunks = dp.chunk_text(clean)
#         all_chunks[upload.filename] = chunks

#     # rebuild the index with both remote and uploaded docs
#     system.ss.build_index(all_chunks)
#     system.inited = True
#     return {'message':'Uploaded documents ingested successfully.'}

@app.post('/employee/query')
def employee_query(req: QueryRequest):
    result, status_code = system.process_query(req.query)
    if status_code != 0:
        # missing fields or bad request
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result
        )
    return result


@app.post('/employee/info')
def emp_info(req: QueryRequest, cur=Depends(get_current_user)):
    if cur['role']!='employee': raise HTTPException(403,'Employee required')
    res, st = system.process_query_for_information(req.query)
    if st!=0: raise HTTPException(400,res)
    return res

@app.post('/hackrx/run')
def emp_ans(req: QueryRequest, cur=Depends(get_current_user)):
    if cur['role']!='employee': raise HTTPException(403,'Employee required')
    res, st = system.answer_questions(req.query['questions'])
    if st!=0: raise HTTPException(400,res)
    return res








# ==============================================================================
# main.py: AI Insurance Agent with FastAPI, Gemini, and FAISS
# ==============================================================================

# ========== CORE IMPORTS ==========
# import os
# import json
# import re
# import warnings
# from io import BytesIO
# from typing import List, Dict, Any, Tuple

# # ========== THIRD-PARTY IMPORTS ==========
# import certifi
# import google.generativeai as genai
# import nltk
# import numpy as np
# import pandas as pd
# import PyPDF2
# import requests
# import spacy
# from fastapi import FastAPI, Depends, HTTPException, status
# from fastapi.security import HTTPBasic, HTTPBasicCredentials
# from pydantic import BaseModel, Field
# from pydantic import BaseSettings

# # --- Resilient FAISS / sklearn import ---
# try:
#     import faiss
#     FAISS_AVAILABLE = True
#     print("🔍 Using FAISS for high-performance search")
# except ImportError:
#     from sklearn.metrics.pairwise import cosine_similarity
#     FAISS_AVAILABLE = False
#     print("⚠️ FAISS not found; falling back to scikit-learn (slower)")

# # ========== INITIAL SETUP ==========
# warnings.filterwarnings('ignore')
# nltk.download('punkt', quiet=True)
# os.environ['SSL_CERT_FILE'] = certifi.where()

# # ==============================================================================
# # CONFIGURATION & MODELS
# # ==============================================================================

# class AppConfig(BaseSettings):
#     """Load settings from .env, with validation & defaults."""
#     GOOGLE_API_KEY: str
#     ADMIN_USERNAME: str = "admin"
#     ADMIN_PASSWORD: str = "admin_pass"
#     EMPLOYEE_USERNAME: str = "employee"
#     EMPLOYEE_PASSWORD: str = "employee_pass"
#     EMBEDDING_MODEL: str = "models/text-embedding-004"
#     GENERATIVE_MODEL: str = "gemini-1.5-flash-latest"
#     DOCUMENT_URLS: List[str] = [
#         "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
#         "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/CHOTGDP23004V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
#         "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/EDLHLGA23009V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
#         "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/HDFHLIP23024V072223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
#         "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/ICIHLIP22012V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
#     ]
#     CHUNK_SIZE: int = 400
#     CHUNK_OVERLAP: int = 50
#     TOP_K_RETRIEVAL: int = 5
#     SIMILARITY_THRESHOLD: float = 0.7

#     class Config:
#         env_file = ".env"


# try:
#     config = AppConfig()
# except Exception as e:
#     raise RuntimeError(f"❌ Could not load configuration (.env): {e}")

# if not config.GOOGLE_API_KEY or "your_key" in config.GOOGLE_API_KEY:
#     raise RuntimeError("❌ GOOGLE_API_KEY is missing or invalid in .env")

# genai.configure(api_key=config.GOOGLE_API_KEY)


# class QueryIn(BaseModel):
#     query: str = Field(..., example="A 45 year old male needs knee surgery. Is it covered?")


# class UserCreate(BaseModel):
#     username: str
#     password: str
#     role: str = Field(..., regex="^(admin|employee)$")


# # ==============================================================================
# # CORE LOGIC CLASSES
# # ==============================================================================

# class DocumentProcessor:
#     """Download PDFs, extract text, clean & chunk into passages."""
#     def download_pdf(self, url: str) -> bytes | None:
#         try:
#             r = requests.get(url, timeout=30, verify=certifi.where())
#             r.raise_for_status()
#             return r.content
#         except Exception as e:
#             print(f"❌ Download error ({url}): {e}")
#             return None

#     def extract_text(self, content: bytes) -> str:
#         try:
#             reader = PyPDF2.PdfReader(BytesIO(content))
#             return "\n".join(p.extract_text() or "" for p in reader.pages)
#         except Exception as e:
#             print(f"❌ PDF parse error: {e}")
#             return ""

#     def clean_text(self, text: str) -> str:
#         return re.sub(r"\s+", " ", text).strip()

#     def chunk_text(self, text: str) -> List[str]:
#         sentences = nltk.sent_tokenize(text)
#         chunks, buf = [], ""
#         for s in sentences:
#             if len((buf + " " + s).split()) <= config.CHUNK_SIZE:
#                 buf += " " + s
#             else:
#                 if len(buf.split()) >= 50:
#                     chunks.append(buf.strip())
#                 buf = s
#         if len(buf.split()) >= 50:
#             chunks.append(buf.strip())
#         return chunks

#     def process_documents(self, urls: List[str]) -> Dict[str, List[str]]:
#         all_chunks: Dict[str, List[str]] = {}
#         for i, url in enumerate(urls):
#             content = self.download_pdf(url)
#             if not content:
#                 continue
#             text = self.extract_text(content)
#             cid = f"doc_{i+1}"
#             all_chunks[cid] = self.chunk_text(self.clean_text(text))
#         return all_chunks


# class QueryParser:
#     """Extracts structured fields from a free-text query."""
#     def __init__(self):
#         try:
#             self.nlp = spacy.load("en_core_web_sm")
#         except:
#             self.nlp = None
#             print("⚠️ spaCy model not found. Run `python -m spacy download en_core_web_sm`")
#         self.age_re = re.compile(r"\b(\d+)\s*year", re.IGNORECASE)
#         self.gender_re = re.compile(r"\b(male|female|m|f)\b", re.IGNORECASE)
#         self.proc_re = re.compile(r"\b([A-Za-z\- ]+?(?:surgery|treatment|procedure|therapy))\b", re.IGNORECASE)
#         self.policy_re = re.compile(r"(\d+)\s*(month|year|day)s?", re.IGNORECASE)

#     def parse(self, q: str) -> Dict[str, Any]:
#         age = gender = procedure = location = policy_duration = None

#         m = self.age_re.search(q)
#         if m:
#             age = int(m.group(1))

#         m = self.gender_re.search(q)
#         if m:
#             g = m.group(1).lower()
#             gender = "Male" if g in ("male", "m") else "Female"

#         m = self.proc_re.search(q)
#         if m:
#             procedure = m.group(1).strip()

#         if self.nlp:
#             doc = self.nlp(q)
#             for ent in doc.ents:
#                 if ent.label_ in ("GPE", "LOC"):
#                     location = ent.text
#                     break

#         m = self.policy_re.search(q)
#         if m:
#             policy_duration = {"value": int(m.group(1)), "unit": m.group(2)}

#         return {
#             "original_query": q,
#             "age": age,
#             "gender": gender,
#             "procedure": procedure,
#             "location": location,
#             "policy_duration": policy_duration,
#         }


# class SemanticSearchEngine:
#     """Embeds with Google, indexes with FAISS (or falls back)."""
#     def __init__(self):
#         self.index = None
#         self.metadata: List[Dict[str, Any]] = []

#     def build_index(self, docs: Dict[str, List[str]]):
#         texts = [chunk for chunks in docs.values() for chunk in chunks]
#         self.metadata = [
#             {"doc_id": doc_id, "chunk_id": idx, "text": chunk}
#             for doc_id, chunks in docs.items()
#             for idx, chunk in enumerate(chunks)
#         ]
#         if not texts:
#             return

#         # embed documents
#         resp = genai.embed_content(
#             model=config.EMBEDDING_MODEL,
#             content=texts,
#             task_type="retrieval_document"
#         )
#         embs = np.ascontiguousarray(
#             pd.DataFrame(resp["embedding"]).to_numpy(dtype="float32")
#         )

#         if FAISS_AVAILABLE:
#             faiss.normalize_L2(embs)
#             self.index = faiss.IndexFlatIP(embs.shape[1])
#             self.index.add(embs)
#         else:
#             norms = np.linalg.norm(embs, axis=1, keepdims=True)
#             self.index = embs / norms

#     def search(self, query: str) -> List[Dict[str, Any]]:
#         if self.index is None:
#             return []

#         # embed query
#         resp = genai.embed_content(
#             model=config.EMBEDDING_MODEL,
#             content=[query],
#             task_type="retrieval_query"
#         )
#         qemb = np.ascontiguousarray(
#             pd.DataFrame(resp["embedding"]).to_numpy(dtype="float32")
#         )

#         if FAISS_AVAILABLE:
#             faiss.normalize_L2(qemb)
#             scores, idxs = self.index.search(qemb, config.TOP_K_RETRIEVAL)
#             scores, idxs = scores[0], idxs[0]
#         else:
#             faiss.normalize_L2(qemb)
#             sims = (self.index @ qemb.T).flatten()
#             idxs = np.argsort(sims)[::-1][: config.TOP_K_RETRIEVAL]
#             scores = sims[idxs]

#         results = []
#         for s, i in zip(scores, idxs):
#             if s >= config.SIMILARITY_THRESHOLD:
#                 m = self.metadata[i]
#                 results.append({**m, "score": float(s)})
#         return results


# class LLMDecisionEngine:
#     """Uses Gemini with JSON Mode for reliable structured output."""
#     def __init__(self):
#         self.model = genai.GenerativeModel(
#             model_name=config.GENERATIVE_MODEL,
#             generation_config={"response_mime_type": "application/json"}
#         )

#     def _generate_json(self, prompt: str) -> Dict[str, Any]:
#         resp = self.model.generate_content(prompt)
#         try:
#             return json.loads(resp.text)
#         except Exception as e:
#             return {"error": f"JSON parse failed: {e}"}

#     def make_decision(self, info: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
#         ctx = "\n---\n".join(f"{c['doc_id']}: {c['text']}" for c in chunks)
#         prompt = (
#             f"You are an insurance policy analyst.\n"
#             f"USER INFO: {json.dumps(info)}\n"
#             f"POLICY CLAUSES:\n{ctx or 'None'}\n"
#             f"Respond with JSON schema:\n"
#             f'{{"decision":"approved"|"rejected","justification":str,"referenced_docs":[str]}}'
#         )
#         return self._generate_json(prompt)

#     def get_info(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
#         ctx = "\n---\n".join(f"{c['doc_id']}: {c['text']}" for c in chunks)
#         prompt = (
#             f"You are an insurance assistant.\n"
#             f"USER QUERY: {query}\n"
#             f"POLICY CLAUSES:\n{ctx or 'None'}\n"
#             f"Respond with JSON schema:\n"
#             f'{{"summary":str,"top_3_chunks":[str]}}'
#         )
#         return self._generate_json(prompt)


# class DocumentQuerySystem:
#     """Orchestrates ingestion, indexing, and RAG operations."""
#     def __init__(self):
#         self.processor = DocumentProcessor()
#         self.parser = QueryParser()
#         self.searcher = SemanticSearchEngine()
#         self.decider = LLMDecisionEngine()
#         self.initialized = False

#     def initialize(self) -> bool:
#         docs = self.processor.process_documents(config.DOCUMENT_URLS)
#         if not docs:
#             return False
#         self.searcher.build_index(docs)
#         self.initialized = True
#         return True

#     def process_query(self, query: str) -> Tuple[Dict[str, Any], int]:
#         if not self.initialized:
#             return {"error": "Service not initialized"}, status.HTTP_503_SERVICE_UNAVAILABLE

#         info = self.parser.parse(query)
#         # ensure no key fields missing
#         for k, v in info.items():
#             if k != "original_query" and v is None:
#                 return {"error": f"Missing field: {k}"}, status.HTTP_400_BAD_REQUEST

#         hits = self.searcher.search(query)
#         decision = self.decider.make_decision(info, hits)
#         return {"query_info": info, "decision": decision}, status.HTTP_200_OK

#     def process_query_info(self, query: str) -> Tuple[Dict[str, Any], int]:
#         if not self.initialized:
#             return {"error": "Service not initialized"}, status.HTTP_503_SERVICE_UNAVAILABLE

#         hits = self.searcher.search(query)
#         if not hits:
#             return {"error": "No relevant information"}, status.HTTP_404_NOT_FOUND

#         summary = self.decider.get_info(query, hits)
#         return summary, status.HTTP_200_OK


# # ==============================================================================
# # FASTAPI APP & ENDPOINTS
# # ==============================================================================

# app = FastAPI(title="AI Insurance Agent", version="2.0.0")
# system = DocumentQuerySystem()
# security = HTTPBasic()

# # In-memory user DB
# users_db = {
#     config.ADMIN_USERNAME:    {"password": config.ADMIN_PASSWORD,    "role": "admin"},
#     config.EMPLOYEE_USERNAME: {"password": config.EMPLOYEE_PASSWORD, "role": "employee"},
# }


# def get_current_user(creds: HTTPBasicCredentials = Depends(security)):
#     u = users_db.get(creds.username)
#     if not u or u["password"] != creds.password:
#         raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
#     return {"username": creds.username, "role": u["role"]}


# def admin_required(user: Dict[str, Any] = Depends(get_current_user)):
#     if user["role"] != "admin":
#         raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
#     return user


# def employee_required(user: Dict[str, Any] = Depends(get_current_user)):
#     if user["role"] not in ("admin", "employee"):
#         raise HTTPException(status.HTTP_403_FORBIDDEN, "Employee access required")
#     return user


# @app.on_event("startup")
# def startup():
#     if not system.initialize():
#         print("❌ WARNING: Document ingestion failed at startup")


# @app.post("/auth/login", tags=["Auth"])
# def login(user=Depends(get_current_user)):
#     return {"message": f"{user['username']} logged in", "role": user["role"]}


# @app.post("/auth/logout", tags=["Auth"])
# def logout():
#     return {"message": "Logged out"}


# @app.post("/admin/users", dependencies=[Depends(admin_required)], status_code=201, tags=["Admin"])
# def create_user(u: UserCreate):
#     if u.username in users_db:
#         raise HTTPException(400, "Username exists")
#     users_db[u.username] = {"password": u.password, "role": u.role}
#     return {"username": u.username, "role": u.role}


# @app.get("/admin/users", dependencies=[Depends(admin_required)], tags=["Admin"])
# def list_users():
#     return [{"username": k, "role": v["role"]} for k, v in users_db.items()]


# @app.post("/admin/ingest", dependencies=[Depends(admin_required)], tags=["Admin"])
# def ingest():
#     ok = system.initialize()
#     if not ok:
#         raise HTTPException(500, "Ingestion failed")
#     return {"message": "Re-ingestion complete"}


# @app.post("/employee/query", dependencies=[Depends(employee_required)], tags=["Employee"])
# def employee_query(q: QueryIn):
#     res, code = system.process_query(q.query)
#     if code != status.HTTP_200_OK:
#         raise HTTPException(code, res)
#     return res


# @app.post("/employee/info", dependencies=[Depends(employee_required)], tags=["Employee"])
# def employee_info(q: QueryIn):
#     res, code = system.process_query_info(q.query)
#     if code != status.HTTP_200_OK:
#         raise HTTPException(code, res)
#     return res






















# import os
# import json
# import re
# import warnings
# from io import BytesIO
# from typing import List, Dict, Any, Tuple

# from dotenv import load_dotenv
# import certifi
# import requests
# import pandas as pd
# import nltk
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from docx import Document as DocxDocument
# import PyPDF2
# import google.generativeai as genai
# import spacy
# from word2number import w2n
# from sklearn.metrics.pairwise import cosine_similarity

# from fastapi import FastAPI, Depends, HTTPException
# from fastapi.security import HTTPBasic, HTTPBasicCredentials
# from pydantic import BaseModel, HttpUrl

# # Suppress warnings
# warnings.filterwarnings('ignore')

# # Ensure SSL certificates are found (macOS)
# os.environ['SSL_CERT_FILE'] = certifi.where()

# # Download NLTK data (silent failures permitted)
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# load_dotenv()
# # ========== CONFIGURATION ==========

#     # ========== CONFIGURATION ==========
# class Config:
#     GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
#     # --- UPDATED ---
#     EMBEDDING_MODEL = 'models/text-embedding-004' 
#     GENERATIVE_MODEL = 'gemini-1.5-flash-latest' # Using latest for better features
#     DOCUMENT_URLS = [
#         "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
#         "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/CHOTGDP23004V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
#         "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/EDLHLGA23009V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
#         "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/HDFHLIP23024V072223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
#         "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/ICIHLIP22012V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
#     ]
        
#     CHUNK_SIZE = 500
#     CHUNK_OVERLAP = 50
#     TOP_K_RETRIEVAL = 5
#     SIMILARITY_THRESHOLD = 0.7 # Often a higher threshold works better with stronger embeddings
    

# # ========== DOCUMENT PROCESSOR ==========
# class DocumentProcessor:
#     def download_pdf(self, url: str) -> bytes:
#         try:
#             resp = requests.get(url, timeout=30)
#             resp.raise_for_status()
#             return resp.content
#         except Exception:
#             return None

#     def extract_text_from_pdf(self, content: bytes) -> str:
#         reader = PyPDF2.PdfReader(BytesIO(content))
#         return '\n'.join(page.extract_text() or '' for page in reader.pages)

#     def clean_text(self, txt: str) -> str:
#         txt = re.sub(r'\s+', ' ', txt)
#         txt = re.sub(r'[^\w\s\.,;:!\?\-\(\)\[\]]', ' ', txt)
#         return txt.strip()

#     def chunk_text(self, text: str) -> List[str]:
#         words = text.split()
#         chunks = []
#         for i in range(0, len(words), Config.CHUNK_SIZE - Config.CHUNK_OVERLAP):
#             chunk = ' '.join(words[i:i + Config.CHUNK_SIZE])
#             if len(chunk) > 50:
#                 chunks.append(chunk)
#         return chunks

#     def process_documents(self, urls: List[str]) -> Dict[str, List[str]]:
#         all_chunks = {}
#         for idx, url in enumerate(urls, start=1):
#             content = self.download_pdf(url)
#             if not content:
#                 continue
#             text = self.extract_text_from_pdf(content)
#             cleaned = self.clean_text(text)
#             all_chunks[f'doc_{idx}'] = self.chunk_text(cleaned)
#         return all_chunks

# # ========== QUERY PARSER ==========
# class QueryParser:
#     def __init__(self):
#         try:
#             self.nlp = spacy.load('en_core_web_sm')
#         except:
#             self.nlp = None
#         self.age_re = re.compile(r'\b(\d+)\s*(?:years?)?', re.IGNORECASE)
#         self.gender_re = re.compile(r'\b(male|female|they)\b', re.IGNORECASE)
#         self.proc_re = re.compile(r'\b([A-Za-z\- ]+? (?:surgery|treatment|procedure))\b', re.IGNORECASE)
#         self.dur_re = re.compile(r'\b(\d+)\s*(?:months?|years?)', re.IGNORECASE)

#     def parse(self, query: str) -> Dict[str, Any]:
#         info = {'original_query': query}
#         m = self.age_re.search(query)
#         info['age'] = int(m.group(1)) if m else None
#         m = self.gender_re.search(query)
#         info['gender'] = m.group(1).title() if m else None
#         m = self.proc_re.search(query)
#         info['procedure'] = m.group(1) if m else None
#         m = self.dur_re.search(query)
#         info['policy_duration'] = {'value': int(m.group(1)), 'unit': 'months'} if m else None
#         if self.nlp:
#             doc = self.nlp(query)
#             for ent in doc.ents:
#                 if ent.label_ in ('GPE', 'LOC'):
#                     info['location'] = ent.text
#                     break
#         else:
#             info['location'] = None
#         return info

# # ========== SEMANTIC SEARCH ==========
# # ========== SEMANTIC SEARCH (UPGRADED) ==========
# class SemanticSearchEngine:
#     def __init__(self):
#         # The SentenceTransformer model is no longer needed here.
#         self.index = None
#         self.metadata = []
#         # We don't store embeddings directly anymore unless FAISS fails.

#     def build_index(self, docs: Dict[str, List[str]]):
#         texts_for_embedding = []
#         for doc_id, chunks in docs.items():
#             for i, chunk_text in enumerate(chunks):
#                 texts_for_embedding.append(chunk_text)
#                 self.metadata.append({'doc_id': doc_id, 'chunk_id': i, 'text': chunk_text})

#         if not texts_for_embedding:
#             print("Warning: No text chunks found to build the index.")
#             return

#         print(f"Generating embeddings for {len(texts_for_embedding)} chunks...")
#         # Embed content in batches for reliability
#         response = genai.embed_content(
#             model=Config.EMBEDDING_MODEL,
#             content=texts_for_embedding,
#             task_type="retrieval_document" # Important for RAG
#         )
#         embeddings = response['embedding']
        
#         df = pd.DataFrame(embeddings)
#         embs_array = np.ascontiguousarray(df.to_numpy(dtype='float32'))

#         faiss.normalize_L2(embs_array)
#         dim = embs_array.shape[1]
#         self.index = faiss.IndexFlatIP(dim)
#         self.index.add(embs_array.astype('float32'))
#         print("FAISS index built successfully.")

#     def search(self, query: str) -> List[Dict[str, Any]]:
#         if self.index is None:
#             return []

#         # Embed the query
#         response = genai.embed_content(
#             model=Config.EMBEDDING_MODEL,
#             content=query,
#             task_type="retrieval_query" # Important for RAG
#         )
#         query_embedding = response['embedding']

#         qemb = np.ascontiguousarray(pd.DataFrame([query_embedding]).to_numpy(dtype='float32'))
    
#         faiss.normalize_L2(qemb)
        
#         scores, idxs = self.index.search(qemb.astype('float32'), Config.TOP_K_RETRIEVAL)
        
#         results = []
#         for score, idx in zip(scores[0], idxs[0]):
#             if score > Config.SIMILARITY_THRESHOLD:
#                 results.append({
#                     'score': float(score),
#                     **self.metadata[idx]
#                 })
#         return results
# # ========== LLM ENGINE ==========
# class LLMDecisionEngine:
#     def __init__(self):
#         genai.configure(api_key=Config.GOOGLE_API_KEY)
#         self.model = genai.GenerativeModel(
#             model_name=Config.GENERATIVE_MODEL,
#             # --- ADDED: Tell the model its output MUST be JSON ---
#             generation_config={"response_mime_type": "application/json"}
#         )
#     def make_decision(self, info: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
#         prompt = f"You are an insurance policy analyst.\nQUERY INFO: {info}\n\n"
#         prompt += "RELEVANT CLAUSES:\n"
#         for i, c in enumerate(chunks, 1):
#             prompt += f"--- Clause {i} from {c['doc_id']} ---\n{c['text']}\n"
#         prompt += "\nRespond with JSON {decision, amount, confidence, justification, referenced_clauses, waiting_period_status, additional_notes}."

#         resp = self.model.generate_content(prompt)
#         m = re.search(r'\{.*?\}', resp.text, re.DOTALL)
#         if m:
#             return json.loads(m.group())
#         return {'decision': 'rejected', 'justification': resp.text}

# # ========== SYSTEM ORCHESTRATOR ==========
# class DocumentQuerySystem:
#     def __init__(self):
#         self.processor = DocumentProcessor()
#         self.parser = QueryParser()
#         self.searcher = SemanticSearchEngine()
#         self.decisioner = LLMDecisionEngine()
#         self.initialized = False

#     def initialize(self) -> bool:
#         docs = self.processor.process_documents(Config.DOCUMENT_URLS)
#         if not docs:
#             return False
#         self.searcher.build_index(docs)
#         self.initialized = True
#         return True

#     def process_query(self, query: str) -> Tuple[Any, int]:
#         if not self.initialized:
#             return {'error': 'Uninitialized'}, 1
#         info = self.parser.parse(query)
#         missing = [k for k in ('age','gender','procedure','location','policy_duration') if info.get(k) is None]
#         if missing:
#             return missing, 1
#         chunks = self.searcher.search(query)
#         decision = self.decisioner.make_decision(info, chunks)
#         return {'query_info': info, 'decision': decision, 'relevant_chunks': len(chunks)}, 0

#     def process_query_info(self, query: str) -> Tuple[Any, int]:
#         if not self.initialized:
#             return {'error': 'Uninitialized'}, 1
#         chunks = self.searcher.search(query)
#         if not chunks:
#             return {'error': 'No relevant info'}, 1
#         return {'query': query, 'relevant_info': [c['text'] for c in chunks[:3]]}, 0

# # ========== FASTAPI SETUP ==========
# app = FastAPI()
# security = HTTPBasic()

# users = {
#     os.getenv('ADMIN_USERNAME'): {'password': os.getenv('ADMIN_PASSWORD'), 'role': 'admin'},
#     os.getenv('EMPLOYEE_USERNAME'): {'password': os.getenv('EMPLOYEE_PASSWORD'), 'role': 'employee'}
# }

# class Credentials(BaseModel):
#     username: str
#     password: str
#     role: str

# class QueryIn(BaseModel):
#     query: str

# # Auth dependencies
# def get_current_user(creds: HTTPBasicCredentials = Depends(security)):
#     u = users.get(creds.username)
#     if not u or creds.password != u['password']:
#         raise HTTPException(401, 'Invalid credentials')
#     return {'username': creds.username, 'role': u['role']}

# def admin_required(user=Depends(get_current_user)):
#     if user['role'] != 'admin':
#         raise HTTPException(403, 'Admin only')
#     return user

# def employee_required(user=Depends(get_current_user)):
#     if user['role'] != 'employee':
#         raise HTTPException(403, 'Employee only')
#     return user

# # Initialize system on startup
# system = DocumentQuerySystem()
# @app.on_event('startup')
# def init_system():
#     if not system.initialize():
#         raise RuntimeError('Initialization failed')

# # Authentication
# @app.post('/auth/login')
# def login(user=Depends(get_current_user)):
#     return {'msg': f"{user['role'].capitalize()} login successful"}

# @app.post('/auth/logout')
# def logout():
#     return {'msg': 'Logged out'}

# # User management (Admin only)
# @app.post('/admin/users', dependencies=[Depends(admin_required)])
# def create_user(creds: Credentials):
#     users[creds.username] = {'password': creds.password, 'role': creds.role}
#     return {'msg': 'User created'}

# @app.get('/admin/users', dependencies=[Depends(admin_required)])
# def list_users():
#     return [{'username': u, 'role': v['role']} for u, v in users.items()]

# # Ingestion endpoint
# @app.post('/admin/ingest', dependencies=[Depends(admin_required)])
# def ingest():
#     if not system.initialize():
#         raise HTTPException(500, 'Ingestion failed')
#     return {'msg': 'Ingested', 'indexed': len(Config.DOCUMENT_URLS)}

# # Employee endpoints
# @app.post('/employee/query', dependencies=[Depends(employee_required)])
# def employee_query(q: QueryIn):
#     result, status = system.process_query(q.query)
#     if status == 1:
#         raise HTTPException(status_code=400, detail={'missing_fields': result})
#     return result

# @app.post('/employee/info', dependencies=[Depends(employee_required)])
# def employee_info(q: QueryIn):
#     result, status = system.process_query_info(q.query)
#     if status == 1:
#         raise HTTPException(status_code=404, detail=result.get('error', 'No info'))
#     return result
