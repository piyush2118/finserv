import os
import json
import re
import warnings
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional, Union

import certifi
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, Request, applications
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, BaseSettings
from nltk.tokenize import sent_tokenize
# from fastapi import File, UploadFile
# Suppress warnings
warnings.filterwarnings('ignore')

import google.generativeai as genai

# Load environment variables
load_dotenv()

# ==================== Configuration ====================
class Config:
    # Load from .env
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "")
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


def _summarize_chunks(chunks: List[str], focus_keyword: Optional[str] = None) -> str:
    sentences = [s for chunk in chunks for s in sent_tokenize(chunk)]
    if not sentences:
        return ""

    if focus_keyword:
        fk = focus_keyword.lower()
        # prefer sentence with keyword + time unit
        for s in sentences:
            ls = s.lower()
            if fk in ls and any(u in ls for u in ("day", "days", "month", "months", "year", "years")):
                return s.strip()
        for s in sentences:
            if fk in s.lower():
                return s.strip()

    # sensible generic fallback: take the first policy-like sentence
    for s in sentences:
        if any(w in s.lower() for w in ("shall", "will", "grace period", "waiting period", "covered", "excluded")):
            return s.strip()
    return sentences[0].strip()

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
        self.chunks: List[str] = []
        self.meta: List[Dict[str, Any]] = []

    def _persist(self):
        # optional persistence; only if FAISS and path is provided
        if FAISS_AVAILABLE and Config.FAISS_INDEX_PATH:
            faiss.write_index(self.index, Config.FAISS_INDEX_PATH)
            side = {
                "chunks": self.chunks,
                "meta": self.meta,
            }
            with open(Config.FAISS_INDEX_PATH + ".json", "w", encoding="utf-8") as f:
                json.dump(side, f)

    def build_index(self, docs: Dict[str, List[str]]):
        all_chunks, meta = [], []
        for did, chs in docs.items():
            for i, c in enumerate(chs):
                all_chunks.append(c)
                meta.append({'doc_id': did, 'chunk_id': i, 'text': c})
        if not all_chunks:
            return
        embs = self.model.encode(all_chunks)
        if FAISS_AVAILABLE:
            embs = np.ascontiguousarray(embs.astype('float32'))
            faiss.normalize_L2(embs)
            self.index = faiss.IndexFlatIP(embs.shape[1])
            self.index.add(embs)
        else:
            self.index = embs  # ndarray
        self.chunks, self.meta = all_chunks, meta
        self._persist()

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        
        query_emb = self.model.encode([query])
        
        if FAISS_AVAILABLE:
            query_emb = np.ascontiguousarray(query_emb.astype('float32'))
            faiss.normalize_L2(query_emb)
            scores, indices = self.index.search(query_emb, top_k)
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.meta) and score >= Config.SIMILARITY_THRESHOLD:
                    result = self.meta[idx].copy()
                    result['score'] = float(score)
                    results.append(result)
            return results
        else:
            # Fallback using sklearn
            similarities = cosine_similarity(query_emb, self.index)[0]
            top_indices = similarities.argsort()[-top_k:][::-1]
            results = []
            for idx in top_indices:
                if similarities[idx] >= Config.SIMILARITY_THRESHOLD:
                    result = self.meta[idx].copy()
                    result['score'] = float(similarities[idx])
                    results.append(result)
            return results

    def add_to_index(self, docs: Dict[str, List[str]]) -> int:
        """Append new docs to the existing global index (creating it if missing)."""
        new_chunks, new_meta = [], []
        for did, chs in docs.items():
            start = sum(1 for m in self.meta if m['doc_id'] == did)
            for i, c in enumerate(chs):
                new_chunks.append(c)
                new_meta.append({'doc_id': did, 'chunk_id': start + i, 'text': c})
        if not new_chunks:
            return 0

        embs = self.model.encode(new_chunks)
        if FAISS_AVAILABLE:
            embs = np.ascontiguousarray(embs.astype('float32'))
            faiss.normalize_L2(embs)
            if self.index is None:
                self.index = faiss.IndexFlatIP(embs.shape[1])
            self.index.add(embs)
        else:
            if self.index is None:
                self.index = embs
            else:
                self.index = np.vstack([self.index, embs])

        self.chunks.extend(new_chunks)
        self.meta.extend(new_meta)
        self._persist()
        return len(new_chunks)
    

# LLMDecisionEngine
import google.generativeai as genai

genai.configure(api_key=Config.GOOGLE_API_KEY)

class LLMDecisionEngine:
    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name=Config.GEMINI_MODEL,
            generation_config={"response_mime_type":"application/json"}
        )

    def create_decision_prompt(self, info: Dict[str, Any], chunks: List[dict]) -> str:
        chunk_data = "\n".join(f"--- {c['doc_id']}:{c['chunk_id']} ---\n{c['text']}"
                            for c in chunks)
        
        return f"""
    You are an insurance claims processor. Based on the policy excerpts and user information, make a claim decision.

    User Information:
    - Age: {info.get('age', 'Not specified')}
    - Gender: {info.get('gender', 'Not specified')}
    - Procedure: {info.get('procedure', 'Not specified')}
    - Location: {info.get('location', 'Not specified')}
    - Policy Duration: {info.get('policy_duration', 'Not specified')}

    Policy Excerpts:
    {chunk_data}

    Respond with JSON:
    {{
        "decision": "approved" | "rejected",
        "amount": <number or null>,
        "confidence": "high" | "medium" | "low",
        "justification": "<reason>",
        "referenced_clauses": ["<doc_id:chunk_id>", ...],
        "waiting_period_status": "applicable" | "not_applicable" | "expired",
        "additional_notes": "<any extra info>"
    }}
    """


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
from threading import Lock

class DocumentQuerySystem:
    def __init__(self):
        self.dp = DocumentProcessor()
        self.qp = QueryParser()
        self.ss = SemanticSearchEngine()
        self.de = LLMDecisionEngine()
        self.inited = False
        self._lock = Lock()
        self.doc_registry: set[str] = set()  # to avoid re-ingesting same URL

    def initialize(self) -> bool:
        docs = self.dp.process_documents(Config.DOCUMENT_URLS)
        if not docs:
            return False
        self.ss.build_index(docs)
        self.doc_registry.update(Config.DOCUMENT_URLS)
        self.inited = True
        return True

    def ingest_documents(self, urls: List[str]) -> int:
        """Download → chunk → embed → append to global index. Skips URLs already ingested."""
        if not urls:
            return 0
        with self._lock:
            new_urls = [u for u in urls if u not in self.doc_registry]
            if not new_urls:
                return 0
            docs = self.dp.process_documents(new_urls)
            added = self.ss.add_to_index(docs)
            self.doc_registry.update(new_urls)
            self.inited = True
            return added

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
    

    def answer_questions(self, documents: Optional[Union[str, List[str]]], questions: List[str]) -> Tuple[Dict[str, Any], int]:
        if not self.inited:
            return {"error": "Service not initialized"}, 503

        # If new docs were sent with this request, ingest them now (incremental)
        if documents:
            urls = [documents] if isinstance(documents, str) else list(documents)
            self.ingest_documents(urls)

        answers = []
        for q in questions:
            hits = self.ss.search(q, Config.TOP_K_RETRIEVAL)
            if not hits:
                answers.append("The policy document does not specify this information.")
                continue
            # Current code tries to get 'summary' from direct answer
            info = self.de.answer_question_direct(q, hits)
            answers.append(info.get("summary") or "The policy document does not specify this information.")

            # Should be:
            answer = self.de.answer_question_direct(q, hits)
            answers.append(answer if answer else "The policy document does not specify this information.")
    

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
def employee_query(req: QueryRequest, cur=Depends(get_current_user)):  # Add this dependency
    if cur['role'] not in ['employee', 'admin']: 
        raise HTTPException(403, 'Employee access required')
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

from fastapi import Request

class HackrxRunIn(BaseModel):
    documents: Optional[Union[str, List[str]]] = None
    questions: List[str]

class HackrxRunOut(BaseModel):
    answers: List[str]

from fastapi import Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

bearer_scheme = HTTPBearer(auto_error=False)

from fastapi import Header

def require_webhook_bearer(authorization: Optional[str] = Header(None)):
    api_key = os.getenv("WEBHOOK_API_KEY")
    if not api_key:  # open route in local dev if you prefer
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")

@app.post('/hackrx/run', response_model=HackrxRunOut, tags=['Webhook'])
def hackrx_run(payload: HackrxRunIn, _=Depends(require_webhook_bearer)):
    res, st = system.answer_questions(payload.documents, payload.questions)
    if st != status.HTTP_200_OK:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, res)
    return res

@app.middleware("http")
async def _debug_auth(request, call_next):
    if request.url.path == "/hackrx/run":
        print("AUTH HEADER:", request.headers.get("authorization"))
    return await call_next(request)

