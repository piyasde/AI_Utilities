import spacy
import numpy as np

# Load spaCy's pre-trained model (for sentence tokenization & word vectors)
nlp = spacy.load("en_core_web_md")  # Medium-sized model with word embeddings

class Transformer:
    def __init__(self, num_heads=4):
        self.num_heads = num_heads

    def multi_head_attention(self, Q, K, V):
        """Multi-Head Attention Mechanism"""
        d_k = Q.shape[-1] // self.num_heads  # Dimension per head
        attention_outputs = []

        # Split Q, K, V into multiple heads
        Q_split = np.array(np.split(Q, self.num_heads, axis=1))
        K_split = np.array(np.split(K, self.num_heads, axis=1))
        V_split = np.array(np.split(V, self.num_heads, axis=1))

        for head in range(self.num_heads):
            attention_scores = np.dot(Q_split[head], K_split[head].T) / np.sqrt(d_k)
            attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)
            head_output = np.dot(attention_weights, V_split[head])
            attention_outputs.append(head_output)

        return np.concatenate(attention_outputs, axis=1)

    def encode(self, sentence_vectors):
        """Encodes each sentence using multi-head self-attention"""
        return self.multi_head_attention(sentence_vectors, sentence_vectors, sentence_vectors)

def get_sentence_vectors(doc):
    """Converts each sentence into a vector (by averaging word vectors)"""
    sentences = list(doc.sents)
    sentence_vectors = []
    for sent in sentences:
        words = [token.vector for token in sent if token.has_vector]
        if words:  # Ignore empty sentences
            sentence_vectors.append(np.mean(words, axis=0))  # Average word vectors for the sentence
    return sentences, np.array(sentence_vectors)

def summarize_text(text, top_n=3):
    """Summarizes the text using Transformer-based multi-head attention"""
    doc = nlp(text)

    # Step 1: Convert each sentence into a vector
    sentences, sentence_vectors = get_sentence_vectors(doc)

    # Step 2: Apply Transformer Encoder
    transformer = Transformer(num_heads=4)
    encoded_sentences = transformer.encode(sentence_vectors)

    # Step 3: Compute attention-based importance scores
    attention_scores = np.mean(np.dot(sentence_vectors, sentence_vectors.T) / np.sqrt(sentence_vectors.shape[1]), axis=0)
    attention_scores = np.exp(attention_scores) / np.sum(np.exp(attention_scores))  # Normalize

    # Step 4: Select top N sentences for summary
    ranked_sentences = sorted(zip(sentences, attention_scores), key=lambda x: x[1], reverse=True)
    summary_sentences = [str(sent[0]) for sent in ranked_sentences[:top_n]]

    # Print ranked sentences
    print("\n Top Sentences Selected for Summary:")
    for sent, score in ranked_sentences[:top_n]:
        print(f"{sent} [Score: {score:.2f}]")

    # Step 5: Return summary as a single paragraph
    return " ".join(summary_sentences)

# Example Document
document_text = """
Steel sales have been growing consistently over the past year. 
Recent reports indicate a significant increase in revenue due to rising demand. 
Market analysts predict further expansion in the steel industry. 
However, global economic uncertainties could impact future growth. 
Government policies are also expected to influence the steel sector. 
Investment in infrastructure projects is one key driver for the increase in sales.
"""

# Generate and print summary
print("Sentences -")
print(document_text)

summary = summarize_text(document_text, top_n=3)
print("\n Summary of the Document:")
print(summary)
