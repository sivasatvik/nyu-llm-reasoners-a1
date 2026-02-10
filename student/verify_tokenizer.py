# #!/usr/bin/env python3
# """Comprehensive tokenizer verification before training."""

# import sys
# sys.path.insert(0, 'student')
# from bpe_tokenizer import Tokenizer

# def test_tokenizer():
#     print("="*80)
#     print("TOKENIZER VERIFICATION")
#     print("="*80)
    
#     # Load tokenizer
#     print("\n1. Loading tokenizer...")
#     tokenizer = Tokenizer.from_files(
#         vocab_filepath='artifacts/vocab.json',
#         merges_filepath='artifacts/merges.txt',
#         special_tokens=['<|endoftext|>']
#     )
#     print(f"   ✓ Vocab size: {len(tokenizer.vocab)}")
#     print(f"   ✓ Number of merges: {len(tokenizer.merges)}")
    
#     # Check merge consistency
#     print("\n2. Checking vocab/merges consistency...")
#     missing_in_vocab = []
#     for i, (a, b) in enumerate(tokenizer.merges[:100]):  # Check first 100
#         merged = a + b
#         if merged not in tokenizer.token_to_id:
#             missing_in_vocab.append((i, merged))
    
#     if missing_in_vocab:
#         print(f"   ✗ ERROR: {len(missing_in_vocab)} merged tokens NOT in vocab!")
#         print(f"   First few missing: {missing_in_vocab[:5]}")
#         return False
#     else:
#         print(f"   ✓ All merge results exist in vocab")
    
#     # Test common words
#     print("\n3. Testing common word encoding...")
#     test_words = [
#         "the", "and", "was", "said", "once", "upon", "little", "time",
#         "there", "happy", "friend", "played", "wanted", "could"
#     ]
    
#     total_tokens = 0
#     merged_tokens = 0
    
#     for word in test_words:
#         ids = tokenizer.encode(" " + word)  # Space prefix like in text
#         total_tokens += len(ids)
#         merged_tokens += sum(1 for tid in ids if tid >= 256)
        
#         if len(ids) == 1 and ids[0] >= 256:
#             status = "✓ MERGED"
#         elif all(tid < 256 for tid in ids):
#             status = "✗ ALL BYTES"
#         else:
#             status = "~ PARTIAL"
        
#         token_str = " ".join(str(tid) for tid in ids)
#         print(f"   '{word}': {token_str:30s} {status}")
    
#     merge_pct = (merged_tokens / total_tokens * 100) if total_tokens > 0 else 0
#     print(f"\n   Merged token %: {merge_pct:.1f}%")
    
#     if merge_pct < 30:
#         print("   ✗ WARNING: Very low merge usage! Tokenizer may be broken.")
#         return False
#     elif merge_pct < 50:
#         print("   ⚠ Low merge usage. May need retraining.")
#     else:
#         print("   ✓ Good merge usage")
    
#     # Test TinyStories sample
#     print("\n4. Testing TinyStories sample...")
#     sample = """Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the park. She wanted to play with it, but it was too high up in a tree.
# Lily's mom came to help. She said, "Let me get the ball for you." She climbed up the tree and got the ball. Lily was so happy! She said, "Thank you, Mommy!"
# They played with the ball all day long. They had so much fun together."""
    
#     ids = tokenizer.encode(sample)
#     base_count = sum(1 for tid in ids if tid < 256)
#     merged_count = sum(1 for tid in ids if tid >= 256)
#     total = len(ids)
    
#     print(f"   Total tokens: {total}")
#     print(f"   Base tokens (0-255): {base_count} ({base_count/total*100:.1f}%)")
#     print(f"   Merged tokens (256+): {merged_count} ({merged_count/total*100:.1f}%)")
    
#     # Expected: For a good BPE tokenizer, merged tokens should be 40-70%
#     if merged_count / total < 0.30:
#         print("   ✗ FAIL: Too few merged tokens! Tokenizer is broken.")
#         return False
#     elif merged_count / total < 0.40:
#         print("   ⚠ WARNING: Low merge rate. Consider retraining.")
#         success = False
#     else:
#         print("   ✓ PASS: Good merge rate")
#         success = True
    
#     # Test round-trip
#     print("\n5. Testing round-trip...")
#     decoded = tokenizer.decode(ids)
#     if sample == decoded:
#         print("   ✓ Perfect round-trip")
#     else:
#         print("   ✗ Round-trip FAILED")
#         print(f"   Original length: {len(sample)}")
#         print(f"   Decoded length: {len(decoded)}")
#         # Find first difference
#         for i, (c1, c2) in enumerate(zip(sample, decoded)):
#             if c1 != c2:
#                 print(f"   First diff at pos {i}: {repr(c1)} vs {repr(c2)}")
#                 break
#         return False
    
#     # Summary
#     print("\n" + "="*80)
#     if success and merge_pct >= 40:
#         print("✓ TOKENIZER IS READY FOR TRAINING")
#         print("="*80)
#         return True
#     else:
#         print("✗ TOKENIZER HAS ISSUES - REGENERATE VOCAB/MERGES")
#         print("="*80)
#         print("\nTo fix, run:")
#         print("  python student/train_bpe_tinystories.py \\")
#         print("    --input data/TinyStoriesV2-GPT4-train.txt \\")
#         print("    --vocab-size 10000 \\")
#         print("    --out-dir artifacts")
#         return False


# if __name__ == "__main__":
#     success = test_tokenizer()
#     sys.exit(0 if success else 1)
