import multiprocessing as mp
import os
from collections import Counter, defaultdict
from typing import BinaryIO

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


## Usage
if __name__ == "__main__":
    with open(..., "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token


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
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Initialize base vocabulary with all single-byte tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    # Pre-tokenize the corpus (optionally in parallel)
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

    # print(word_counts.most_common(10))

    # Build initial pair counts and index
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for word, freq in word_counts.items():
        all_pairs_in_word = pairs_in_word(word)
        for pair, count in all_pairs_in_word.items():
            pair_counts[pair] += freq * count
            pair_to_words[pair].add(word)

    # print(pair_counts.most_common(10))
    # print(pair_to_words)

    # Determine number of merges to perform (reserve space for special tokens)
    num_merges = max(0, vocab_size - len(vocab) - len(special_tokens))

    for _ in range(num_merges):
        if not pair_counts:
            break
        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        if pair_counts[best_pair] <= 0:
            break

        merges.append(best_pair)

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

    # Add merged tokens to vocab
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

    return vocab, merges