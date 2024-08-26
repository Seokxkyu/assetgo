import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

class KeywordExtractor:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        model = SentenceTransformer(model_name)
        self.kw_model = KeyBERT(model=model)

    def extract_keywords(self, text, keyphrase_ngram_range=(1, 1), use_mmr=True, diversity=0.5, use_maxsum=False, nr_candidates=20):
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=keyphrase_ngram_range,
            use_mmr=use_mmr,
            diversity=diversity,
            use_maxsum=use_maxsum,
            nr_candidates=nr_candidates
        )
        return {kw[0]: kw[1] for kw in keywords}

    def extract_keywords_from_parquet(self, input_parquet, output_parquet):
        df = pd.read_parquet(input_parquet)
        
        df['keywords'] = df['content'].apply(lambda x: self.extract_keywords(x))
        
        df.to_parquet(output_parquet, index=False)
        return df
