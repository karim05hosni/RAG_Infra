from sentence_transformers import SentenceTransformer
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForSequenceClassification

bge_m3_model = 'BAAI/bge-m3'
import psutil, os

def mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

print(f"before sentence_transformer_embedding_models model load: {mem():.0f} MB")
# your model load here
sentence_transformer_embedding_models = {
    'bf16': {
        bge_m3_model: SentenceTransformer(
            bge_m3_model,
            model_kwargs={"torch_dtype": bfloat16}
        )
    }
}

print(f"After sentence_transformer_embedding_models model load: {mem():.0f} MB")


print(f"before translation model load: {mem():.0f} MB")
en_to_fr_translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
en_to_fr_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

fr_to_en_translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
fr_to_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

print(f"After translation model load: {mem():.0f} MB")
print(f"before NLI model load: {mem():.0f} MB")
nli_model = AutoModelForSequenceClassification.from_pretrained("vectara/hallucination_evaluation_model", trust_remote_code=True, device_map="auto")
print(f"After NLI model load: {mem():.0f} MB")