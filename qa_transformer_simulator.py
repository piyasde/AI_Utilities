
# qa_transformer_simulator.py
# Simulates a transformer encoder-decoder for Q&A with attention and memory — no ML libraries

# 1. Simulated Q&A memory (like decoder memory of learned pairs)
qa_memory = {
    "what is the capital of france": ["Paris", "is", "the", "capital", "of", "France"],
    "who is the president of usa": ["The", "President", "is", "Joe", "Biden"],
    "what is the boiling point of water": ["100", "degrees", "Celsius", "at", "sea", "level"],
    "who wrote romeo and juliet": ["William", "Shakespeare", "wrote", "Romeo", "and", "Juliet"]
}

# 2. Assign fake embeddings for each word based on importance
# Higher = more meaningful, for attention simulation
word_embeddings = {
    "what": 0.1, "is": 0.1, "the": 0.1, "of": 0.0, "who": 0.1,
    "capital": 0.9, "france": 0.95,
    "president": 0.9, "usa": 0.95,
    "boiling": 0.8, "point": 0.8, "water": 0.9,
    "wrote": 0.8, "romeo": 0.95, "juliet": 0.95
}

# 3. Attention scoring (simple dot product simulation)
def attention_score(w1, w2):
    return word_embeddings.get(w1, 0.1) * word_embeddings.get(w2, 0.1)

# 4. Encode input: tokenize + compute attention focus map
def encode_question(question):
    tokens = question.lower().split()
    focus_map = {}
    for w1 in tokens:
        focus = 0
        for w2 in tokens:
            if w1 != w2:
                focus += attention_score(w1, w2)
        focus_map[w1] = round(focus, 2)
    return tokens, focus_map

# 5. Find best memory match (decoder selection)
def find_best_match(tokens):
    joined = " ".join(tokens)
    best_score = 0
    best_key = None
    for known_question in qa_memory:
        match_score = sum(1 for word in tokens if word in known_question)
        if match_score > best_score:
            best_score = match_score
            best_key = known_question
    return best_key

# 6. Decode: generate output token by token
def decode_answer(memory_key):
    return qa_memory.get(memory_key, ["I", "don't", "know", "that", "yet."])

# 7. Full Q&A Transformer simulation
def qa_transformer_respond(user_question):
    print("\n--- ENCODER ---")
    tokens, focus_map = encode_question(user_question)
    print("Tokens:", tokens)
    print("Focus Map:", focus_map)

    print("\n--- DECODER ---")
    match_key = find_best_match(tokens)
    if match_key:
        print(f"Best memory match: '{match_key}'")
        response = decode_answer(match_key)
        print("Generated Answer: ", " ".join(response))
    else:
        print("No relevant memory found.")

# 8. Run test
if __name__ == "__main__":
    test_questions = [
        "What is the capital of France",
        "Who wrote Romeo and Juliet",
        "What is the boiling point of water",
        "president USA",
        "GDP India"
    ]

    for q in test_questions:
        print("\n============================")
        print("User Question:", q)
        qa_transformer_respond(q)
