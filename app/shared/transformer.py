from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForSequenceClassification


embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
en_to_fr_translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
en_to_fr_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

fr_to_en_translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
fr_to_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

nli_model = AutoModelForSequenceClassification.from_pretrained("vectara/hallucination_evaluation_model", trust_remote_code=True, device_map="auto")
