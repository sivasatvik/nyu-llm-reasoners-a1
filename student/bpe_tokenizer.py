import multiprocessing as mp
import os
import time
from collections import Counter, defaultdict
from typing import BinaryIO, Iterable, Iterator

import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

_PRETOKEN_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def build_special_split_pattern(special_tokens: list[str]) -> re.Pattern | None:
    if not special_tokens:
        return None
    escaped = [re.escape(tok) for tok in special_tokens]
    return re.compile("|".join(escaped))


def pretokenize_text(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    counter: Counter[tuple[bytes, ...]] = Counter()
    special_split = build_special_split_pattern(special_tokens)

    # Check for special tokens and split text accordingly
    if special_split is None:
        segments = [text]
    else:
        segments = special_split.split(text)

    cache: dict[str, tuple[bytes, ...]] = {}
    # Go through each segment and cache pre-tokenized results
    # and update counts and finally return the total counts
    for segment in segments:
        # if not segment:
        #     continue
        for match in _PRETOKEN_PATTERN.finditer(segment):
            token = match.group(0)
            cached = cache.get(token)
            if cached is None:
                token_bytes = token.encode("utf-8")
                cached = tuple(bytes([b]) for b in token_bytes)
                cache[token] = cached
            counter[cached] += 1
    return counter


def iter_pretokenized_segments(text: str, special_tokens: list[str]) -> Iterator[tuple[str, bool]]:
    """
    Yield (segment, is_special) pairs in order, preserving special tokens as single segments.
    """
    special_split = build_special_split_pattern(special_tokens)
    if special_split is None:
        yield text, False
        return

    last_end = 0
    for match in special_split.finditer(text):
        start, end = match.span()
        if start > last_end:
            yield text[last_end:start], False
        yield text[start:end], True
        last_end = end
    if last_end < len(text):
        yield text[last_end:], False


def pretokenize_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:
    with open(input_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    text = data.decode("utf-8", errors="ignore")
    return pretokenize_text(text, special_tokens)


def pairs_in_word(word: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
    pairs: Counter[tuple[bytes, bytes]] = Counter()
    for i in range(len(word) - 1):
        pairs[(word[i], word[i + 1])] += 1
    return pairs


def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    if len(word) < 2:
        return word
    merged: list[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    verbose: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer.
    
    Args:
        input_path: Path to the input text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to add
        verbose: If True, print timing information for profiling
    
    Returns:
        Tuple of (vocab dict, merges list)
    """
    if verbose:
        print("\n" + "="*80)
        print("BPE TRAINING PROFILING")
        print("="*80)
        overall_start = time.time()
    
    # Initialize base vocabulary with all single-byte tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    # Pre-tokenize the corpus (optionally in parallel)
    if verbose:
        section_start = time.time()
        print("\n[1/4] Pre-tokenization...")
    
    file_size = os.path.getsize(input_path)
    use_parallel = file_size >= 5_000_000 and (os.cpu_count() or 1) > 1 and len(special_tokens) > 0

    if use_parallel:
        split_special_token = special_tokens[0].encode("utf-8")
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, os.cpu_count() or 1, split_special_token)
        args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        with mp.Pool(processes=len(args)) as pool:
            counters = pool.starmap(pretokenize_chunk, args)
        word_counts: Counter[tuple[bytes, ...]] = Counter()
        for c in counters:
            word_counts.update(c)
    else:
        with open(input_path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        word_counts = pretokenize_text(text, special_tokens)

    if verbose:
        print(f"  ✓ Completed in {time.time() - section_start:.2f}s")
        print(f"  ✓ Found {len(word_counts)} unique words")
        print(f"  ✓ Total word occurrences: {sum(word_counts.values())}")

    # print(word_counts.most_common(10))

    # Build initial pair counts and index
    if verbose:
        section_start = time.time()
        print("\n[2/4] Building initial pair counts...")
    
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for word, freq in word_counts.items():
        all_pairs_in_word = pairs_in_word(word)
        for pair, count in all_pairs_in_word.items():
            pair_counts[pair] += freq * count
            pair_to_words[pair].add(word)

    if verbose:
        print(f"  ✓ Completed in {time.time() - section_start:.2f}s")
        print(f"  ✓ Found {len(pair_counts)} unique pairs")

    # print(pair_counts.most_common(10))
    # print(pair_to_words)

    # Determine number of merges to perform (reserve space for special tokens)
    num_merges = max(0, vocab_size - len(vocab) - len(special_tokens))

    if verbose:
        section_start = time.time()
        print(f"\n[3/4] Performing {num_merges} BPE merges...")
        merge_times = []
        find_best_times = []
        update_times = []

    for merge_idx in range(num_merges):
        if verbose and merge_idx % 100 == 0 and merge_idx > 0:
            avg_merge = sum(merge_times[-100:]) / len(merge_times[-100:])
            avg_find = sum(find_best_times[-100:]) / len(find_best_times[-100:])
            avg_update = sum(update_times[-100:]) / len(update_times[-100:])
            print(f"  Progress: {merge_idx}/{num_merges} merges | "
                  f"Avg time per merge: {avg_merge*1000:.2f}ms "
                  f"(find: {avg_find*1000:.2f}ms, update: {avg_update*1000:.2f}ms)")
        
        if not pair_counts:
            break
        
        if verbose:
            find_start = time.time()
        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        if verbose:
            find_best_times.append(time.time() - find_start)
        
        if pair_counts[best_pair] <= 0:
            break

        merges.append(best_pair)

        if verbose:
            update_start = time.time()
        
        affected_words = list(pair_to_words.get(best_pair, set()))
        if not affected_words:
            break

        # Go over the affected words due to this merge
        # and update counts accordingly
        for word in affected_words:
            freq = word_counts.get(word, 0)
            if freq == 0:
                continue

            # Remove old pairs for this word
            old_pairs = pairs_in_word(word)
            for pair, count in old_pairs.items():
                pair_counts[pair] -= freq * count
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                pair_to_words[pair].discard(word)
                if not pair_to_words[pair]:
                    pair_to_words.pop(pair, None)

            # Merge the word and update counts
            new_word = merge_word(word, best_pair)
            if new_word != word:
                word_counts.pop(word, None)
                word_counts[new_word] += freq
            else:
                word_counts[word] = freq

            # Add new pairs for the updated word
            new_pairs = pairs_in_word(new_word)
            for pair, count in new_pairs.items():
                pair_counts[pair] += freq * count
                pair_to_words[pair].add(new_word)
        
        if verbose:
            update_times.append(time.time() - update_start)
            merge_times.append(time.time() - (find_start if 'find_start' in locals() else update_start))

    if verbose:
        print(f"  ✓ Completed in {time.time() - section_start:.2f}s")
        print(f"  ✓ Performed {len(merges)} merges")
        if merge_times:
            print(f"  ✓ Avg time per merge: {sum(merge_times)/len(merge_times)*1000:.2f}ms")
            print(f"  ✓ Avg time finding best pair: {sum(find_best_times)/len(find_best_times)*1000:.2f}ms")
            print(f"  ✓ Avg time updating counts: {sum(update_times)/len(update_times)*1000:.2f}ms")

    # Add merged tokens to vocab
    if verbose:
        section_start = time.time()
        print("\n[4/4] Building final vocabulary...")
    
    next_id = len(vocab)
    for merge in merges:
        vocab[next_id] = merge[0] + merge[1]
        next_id += 1

    # Add special tokens at the end
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[next_id] = token_bytes
            next_id += 1

    if verbose:
        print(f"  ✓ Completed in {time.time() - section_start:.2f}s")
        print(f"  ✓ Final vocabulary size: {len(vocab)}")
        print("="*80)
        print(f"TOTAL TIME: {time.time() - overall_start:.2f}s")
        print("="*80 + "\n")

    return vocab, merges
    

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self._encode_cache: dict[str, list[int]] = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        import json

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        vocab = {int(k): v.encode("latin-1") for k, v in vocab_data.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                cleaned = line.strip()
                if not cleaned:
                    continue
                parts = cleaned.split(" ", 1)
                if len(parts) != 2:
                    continue
                a_str, b_str = parts
                try:
                    a_bytes = bytes.fromhex(a_str)
                    b_bytes = bytes.fromhex(b_str)
                except ValueError:
                    # Fallback for old latin-1 format
                    a_bytes = a_str.encode("latin-1")
                    b_bytes = b_str.encode("latin-1")
                merges.append((a_bytes, b_bytes))

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs using BPE.
        
        Step 1: Pre-tokenize the text into pre-tokens (word pieces)
        Step 2: Apply the merges from training in order to each pre-token
        Step 3: Convert final tokens to vocabulary IDs
        
        Args:
            text: The text to encode
            
        Returns:
            List of token IDs
        """
        tokens: list[int] = []
        special_tokens = self.special_tokens or []

        def bpe_encode_token(token: str) -> list[int]:
            cached = self._encode_cache.get(token)
            if cached is not None:
                return cached

            token_bytes = token.encode("utf-8")
            word = tuple(bytes([b]) for b in token_bytes)

            while True:
                pairs = pairs_in_word(word)
                if not pairs:
                    break
                best_pair = min(
                    (pair for pair in pairs if pair in self.merge_ranks),
                    key=lambda pair: self.merge_ranks[pair],
                    default=None,
                )
                if best_pair is None:
                    break
                word = merge_word(word, best_pair)

            encoded: list[int] = []
            for tok in word:
                token_id = self.token_to_id.get(tok)
                if token_id is not None:
                    encoded.append(token_id)
                else:
                    # Fallback: encode missing merged tokens as individual bytes
                    for byte in tok:
                        encoded.append(byte)

            self._encode_cache[token] = encoded
            return encoded

        # Step 1: Pre-tokenize the text while preserving special tokens
        for segment, is_special in iter_pretokenized_segments(text, special_tokens):
            if not segment:
                continue
            if is_special:
                token_id = self.token_to_id.get(segment.encode("utf-8"))
                if token_id is not None:
                    tokens.append(token_id)
                continue

            for match in _PRETOKEN_PATTERN.finditer(segment):
                token = match.group(0)
                tokens.extend(bpe_encode_token(token))

        return tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of strings to token IDs.
        
        Args:
            iterable: An iterable of strings
            
        Yields:
            Token IDs one by one
        """
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        byte_chunks = [self.vocab[token_id] for token_id in token_ids if token_id in self.vocab]
        byte_stream = b"".join(byte_chunks)
        return byte_stream.decode("utf-8", errors="ignore")

## Usage
# if __name__ == "__main__":
#     import pathlib
#     vocab, merges = train_bpe(
#         input_path=(pathlib.Path(__file__).resolve().parent.parent) / "tests" / "fixtures" / "tinystories_sample_5M.txt",
#         vocab_size=10000,
#         special_tokens=["<|endoftext|>"],
#     )
#     print(vocab)