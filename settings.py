"""Configuration commune du notebook, de Streamlit et du serveur MCP."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env', override=False)


def llm_config():
    provider = os.getenv('LLM_PROVIDER', 'ollama').lower()
    if provider not in {'ollama', 'openai'}:
        raise ValueError('LLM_PROVIDER doit être ollama ou openai.')
    if provider == 'ollama':
        return {
            'model': os.getenv('LLM_MODEL', 'qwen3-vl:4b-instruct'),
            'base_url': os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1'),
            'api_key': 'ollama',
        }
    key = os.getenv('OPENAI_API_KEY')
    if not key or key.strip() in {'....', 'votre_cle_ici'}:
        raise ValueError('Configurez OPENAI_API_KEY dans .env pour le fournisseur OpenAI.')
    return {'model': os.getenv('LLM_MODEL', 'gpt-4.1-mini'), 'api_key': key}


def create_llm_client():
    from openai import OpenAI
    config = llm_config()
    return OpenAI(**{k: v for k, v in config.items() if k != 'model'}, timeout=180, max_retries=1)


def llm_options():
    """Éviter que le raisonnement interne épuise le budget de réponse local."""
    return {'reasoning_effort': 'none'} if os.getenv('LLM_PROVIDER', 'ollama').lower() == 'ollama' else {}
