import sys
# sys.path.append('/Users/ritikagoel/workspace/synthetic-catalyst-internal-api2/ragaai-catalyst')

import pytest
from ragaai_catalyst import SyntheticDataGeneration
import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

@pytest.fixture
def synthetic_gen():
    return SyntheticDataGeneration()

@pytest.fixture
def sample_text(synthetic_gen):
    text_file = "/Users/siddharthakosti/Downloads/catalyst_error_handling/catalyst_v2/catalyst_v2_new_1/data/ai_document_061023_2.pdf"  # Update this path as needed
    return synthetic_gen.process_document(input_data=text_file)

def test_invalid_csv_processing(synthetic_gen):
    """Test processing an invalid CSV file"""
    with pytest.raises(Exception):
        synthetic_gen.process_document(input_data="/Users/siddharthakosti/Downloads/catalyst_error_handling/catalyst_v2/catalyst_v2_new_1/data/OG1.csv")

def test_special_chars_csv_processing(synthetic_gen):
    """Test processing CSV with special characters"""
    with pytest.raises(Exception):
        synthetic_gen.process_document(input_data="/Users/siddharthakosti/Downloads/catalyst_error_handling/catalyst_v2/catalyst_v2_new_1/data/OG1.csv")



def test_missing_llm_proxy(synthetic_gen, sample_text):
    """Test behavior when internal_llm_proxy is not provided"""
    print('-'*10)
    print(OPENAI_API_KEY)
    print('-'*10)
    with pytest.raises(ValueError, match="API key must be provided"):
        synthetic_gen.generate_qna(
            text=sample_text,
            question_type='mcq',
            model_config={"provider": "openai", "model": "gpt-4o-mini"},
            n=20,
            user_id="1"
        )

def test_llm_proxy(synthetic_gen, sample_text):
    result = synthetic_gen.generate_qna(
            text=sample_text,
            question_type='mcq',
            model_config={"provider": "gemini", "model": "gemini-1.5-flash"},
            n=15,
            internal_llm_proxy="http://4.247.138.221:4000/chat/completions",
            user_id="1"
        )
    assert len(result) == 15 

    

def test_invalid_llm_proxy(synthetic_gen, sample_text):
    """Test behavior with invalid internal_llm_proxy URL"""
    with pytest.raises(Exception, match="No connection adapters were found for"):
        synthetic_gen.generate_qna(
            text=sample_text,
            question_type='mcq',
            model_config={"provider": "openai", "model": "gpt-4o-mini"},
            n=2,
            internal_llm_proxy="tp://invalid.url",
            user_id="1"
        )

def test_missing_model_config(synthetic_gen, sample_text):
    """Test behavior when model_config is not provided"""
    with pytest.raises(ValueError, match="Model configuration must be provided with a valid provider and model"):
        synthetic_gen.generate_qna(
            text=sample_text,
            question_type='mcq',
            n=2,
            internal_llm_proxy="http://20.244.126.4:4000/chat/completions",
            user_id="1"
        )

def test_missing_api_key_for_external_provider(synthetic_gen, sample_text):
    """Test behavior when API key is missing for external provider"""
    with pytest.raises(ValueError, match="API key must be provided"):
        synthetic_gen.generate_qna(
            text=sample_text,
            question_type='mcq',
            model_config={"provider": "gemini", "model": "gemini/gemini-1.5-flash"},
            n=5
        )

def test_invalid_api_key(synthetic_gen, sample_text):
    """Test behavior with invalid API key"""
    with pytest.raises(Exception, match="Failed to generate valid response after 3 attempts: Invalid API key provided"):
        synthetic_gen.generate_qna(
            text=sample_text,
            question_type='mcq',
            model_config={"provider": "gemini", "model": "gemini/gemini-1.5-flash"},
            n=5,
            api_key='invalid_key'
        )

def test_default_question_count(synthetic_gen, sample_text):
    """Test default number of questions when n is not provided"""
    result = synthetic_gen.generate_qna(
        text=sample_text,
        question_type='mcq',
        model_config={"provider": "openai", "model": "gpt-4o-mini"},
        internal_llm_proxy="http://4.247.138.221:4000/chat/completions",
        user_id="1"
    )
    assert len(result) == 5  # Default should be 5 questions

def test_default_question_type(synthetic_gen, sample_text):
    """Test default question type when question_type is not provided"""
    result = synthetic_gen.generate_qna(
        text=sample_text,
        model_config={"provider": "openai", "model": "gpt-4o-mini"},
        n=5,
        internal_llm_proxy="http://20.244.126.4:4000/chat/completions",
        user_id="1"
    )
    # Verify result contains simple Q/A format without multiple choice options
    assert all('options' not in qa for qa in result)

def test_question_count_matches_n(synthetic_gen, sample_text):
    """Test if number of generated questions matches n"""
    n = 2
    result = synthetic_gen.generate_qna(
        text=sample_text,
        question_type='mcq',
        model_config={"provider": "openai", "model": "gpt-4o-mini"},
        n=n,
        internal_llm_proxy="http://4.247.138.221:4000/chat/completions",
        user_id="1"
    )
    assert len(result) == n

def test_proxy_call_check(synthetic_gen,sample_text):
    """Test compatibility when proxy script called"""

    result = synthetic_gen.generate_qna(
            text=sample_text,
            question_type='simple',
            model_config={"provider": "gemini", "model": "gemini-1.5-flash", "api_base": "http://172.172.11.158:8000/v1alpha1/v1alpha1/predictions"},
            n=5
        )
    assert len(result) == 5 



