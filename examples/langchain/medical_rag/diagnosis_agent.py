
import os
import warnings
from typing import List, Dict
from pypdf import PdfReader
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer

from dotenv import load_dotenv
load_dotenv()

catalyst = RagaAICatalyst(
    access_key=os.getenv('CATALYST_ACCESS_KEY'), 
    secret_key=os.getenv('CATALYST_SECRET_KEY'), 
    base_url=os.getenv('CATALYST_BASE_URL')
)
tracer = Tracer(
    project_name=os.environ['PROJECT_NAME'],
    dataset_name=os.environ['DATASET_NAME'],
    tracer_type="agentic/langchain",
)

init_tracing(catalyst=catalyst, tracer=tracer)

MEDICAL_TEXTS_DIR = "data/medical_texts"
SYMPTOM_MAP_CSV = "data/symptom_disease_map.csv"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
MODEL_TYPE = "openai"

class MedicalDataLoader:
    @staticmethod
    def load_pdfs() -> List[str]:
        texts = []
        for pdf_file in os.listdir(MEDICAL_TEXTS_DIR):
            reader = PdfReader(os.path.join(MEDICAL_TEXTS_DIR, pdf_file))
            for page in reader.pages:
                texts.append(page.extract_text())
        return texts

    @staticmethod
    def load_symptom_map() -> pd.DataFrame:
        return pd.read_csv(SYMPTOM_MAP_CSV)

class DiagnosisSystem:
    def __init__(self):
        self.symptom_df = MedicalDataLoader.load_symptom_map()
        self.vector_db = self._create_vector_db()
        self.llm = self._init_llm()
        
    def _create_vector_db(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        texts = MedicalDataLoader.load_pdfs()
        chunks = text_splitter.split_text("\n\n".join(texts))
        
        return Chroma.from_texts(
            texts=chunks,
            embedding=HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL),
            persist_directory="./chroma_db"
        )
    
    def _init_llm(self):
        if MODEL_TYPE == "openai":
            return OpenAI(temperature=0.3)
        elif MODEL_TYPE == "local":
            raise NotImplementedError("Local model not implemented yet.")
    
    def _match_symptoms(self, symptoms: List[str]) -> Dict:
        matched = []
        
        for _, row in self.symptom_df.iterrows():
            if any(s in row["symptom"] for s in symptoms):
                matched.append({
                    "disease": row["disease"],
                    "confidence": row["confidence"],
                    "symptoms": row["symptom"].split(",")
                })
        return sorted(matched, key=lambda x: x["confidence"], reverse=True)
    
    def generate_diagnosis(self, symptoms: List[str], patient_history: str):
        matched = self._match_symptoms(symptoms)
        
        prompt_template = """Use these medical guidelines to explain {disease}:
        {context}
        
        Patient History: {history}
        Symptoms: {symptoms}
        
        Provide:
        1. Likely diagnosis (confidence score)
        2. Key evidence from guidelines
        3. Recommended next steps"""
        
        PROMPT =  PromptTemplate(
            template=prompt_template,
            input_variables=["context", "disease", "history", "symptoms"]
        )

        results = []
        for candidate in matched[:3]:
            retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
            qa_chain = (
                {
                    'context': retriever, 
                    'disease': lambda _: candidate["disease"],
                    'history': lambda _: patient_history,
                    'symptoms': lambda _: ", ".join(symptoms)
                 }
                 | PROMPT
                 | self.llm
                 | StrOutputParser()
            )

            response = qa_chain.invoke('Find the likely diagnosis, key evidence, and recommended next steps.')
            
            
            results.append({
                "disease": candidate["disease"],
                "confidence": candidate["confidence"],
                "evidence": response
            })
        
        return results

def main():
    system = DiagnosisSystem()
    
    print("Medical Diagnosis Assistant\n")
    symptoms = ["fever", "headache", "fatigue"]
    history = '70 years old female, no prior medical history'
    
    print("\nAnalyzing...")
    diagnoses = system.generate_diagnosis(symptoms, history)
    
    print("\nPossible Diagnoses:")
    for idx, diagnosis in enumerate(diagnoses, 1):
        print(f"\n{idx}. {diagnosis['disease'].upper()} (Confidence: {diagnosis['confidence']*100:.1f}%)")
        print(f"Evidence:\n{diagnosis['evidence']}\n")

if __name__ == "__main__":
    with tracer:
        main()