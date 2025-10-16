# chunker.py
# Token-aware chunking helpers. Uses tiktoken to measure tokens.
import tiktoken

ENC = None
def get_encoding(name="cl100k_base"):
    global ENC
    if ENC is None:
        try:
            ENC = tiktoken.get_encoding(name)
        except Exception:
            # fallback: simple char tokenizer if tiktoken isn't available
            ENC = None
    return ENC

def count_tokens(text: str):
    enc = get_encoding()
    if enc is None:
        # rough heuristic: 4 chars per token
        return max(1, len(text)//4)
    return len(enc.encode(text))

def smart_chunk(text: str, max_tokens=300, overlap_tokens=50):
    """
    Splits text into token-bounded chunks with simple word-based accumulation.
    Returns list of chunk strings.
    """
    words = text.split()
    chunks = []
    curr = []
    curr_tokens = 0
    for w in words:
        # approximate token count for the word (include trailing space)
        tcount = count_tokens(w + " ")
        if curr_tokens + tcount > max_tokens:
            chunks.append(" ".join(curr))
            # create overlap
            if overlap_tokens > 0:
                # find overlap by stepping back words until approx overlap_tokens reached
                ov = []
                ov_tokens = 0
                for token_word in reversed(curr):
                    ov.insert(0, token_word)
                    ov_tokens += count_tokens(token_word + " ")
                    if ov_tokens >= overlap_tokens:
                        break
                curr = ov.copy()
                curr_tokens = count_tokens(" ".join(curr))
            else:
                curr = []
                curr_tokens = 0
        curr.append(w)
        curr_tokens += tcount
    if curr:
        chunks.append(" ".join(curr))
    return chunks
