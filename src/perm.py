import os
import google.generativeai as genai
from dotenv import load_dotenv

# Carrega o .env para ler a GOOGLE_API_KEY
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Erro: GOOGLE_API_KEY não encontrada no seu arquivo .env")
else:
    genai.configure(api_key=api_key)
    print("Buscando modelos de embedding disponíveis...")
    try:
        for m in genai.list_models():
            if 'embedContent' in m.supported_generation_methods:
                print(f"Modelo disponível: {m.name}")
    except Exception as e:
        print(f"Erro ao listar modelos: {e}")