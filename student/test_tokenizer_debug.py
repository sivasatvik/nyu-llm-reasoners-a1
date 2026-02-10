# #!/usr/bin/env python3
# """Debug tokenizer encoding to verify BPE merges are applied."""

# import sys
# sys.path.insert(0, 'student')
# from bpe_tokenizer import Tokenizer

# # Load tokenizer
# tokenizer = Tokenizer.from_files(
#     vocab_filepath='artifacts/vocab.json',
#     merges_filepath='artifacts/merges.txt',
#     special_tokens=['<|endoftext|>']
# )

# # Test encoding
# test_text = "Once upon a time, there was a little boy named Tim."
# print(f"Test text: {test_text}")
# print()

# encoded = tokenizer.encode(test_text)
# print(f"Encoded token IDs ({len(encoded)} tokens): {encoded}")
# print()

# # Decode to verify round-trip
# decoded = tokenizer.decode(encoded)
# print(f"Decoded: {decoded}")
# print(f"Round-trip match: {test_text == decoded}")
# print()

# # Check token distribution
# num_base = sum(1 for tid in encoded if tid < 256)
# num_merged = sum(1 for tid in encoded if tid >= 256)
# print(f"Base tokens (0-255): {num_base} ({num_base/len(encoded)*100:.1f}%)")
# print(f"Merged tokens (256+): {num_merged} ({num_merged/len(encoded)*100:.1f}%)")
# print()

# # Show what each token decodes to
# print("Token-by-token breakdown:")
# for i, tid in enumerate(encoded[:20]):
#     token_bytes = tokenizer.vocab.get(tid, b'<UNK>')
#     try:
#         token_text = token_bytes.decode('utf-8')
#     except:
#         token_text = f"<INVALID UTF-8: {token_bytes.hex()}>"
#     print(f"  {i}: ID={tid:4d} bytes={token_bytes.hex():20s} text={repr(token_text)}")

# print("\n" + "="*80)
# print("If you see mostly base tokens (0-255), the tokenizer is NOT applying merges!")
# print("Expected: mostly merged tokens (256-9999) for common words")
# print("="*80)
