# ═══════════════════════════════════════════════════════════════
# BATCH 1 — Build FAISS Index from existing embeddings
# Run this ONCE from your project folder:
#   python build_faiss_index.py
#
# Input  : src/combined_embeddings.npy  (213,314 x 384 float32)
# Output : src/combined_faiss.index     (FAISS binary index)
# ═══════════════════════════════════════════════════════════════

import numpy as np
import faiss
import time
import os

# ── Paths (resolved dynamically — works on any machine) ───────────
BASE_PATH       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_PATH = os.path.join(BASE_PATH, "src", "combined_embeddings.npy")
INDEX_PATH      = os.path.join(BASE_PATH, "src", "combined_faiss.index")

print("=" * 55)
print("  FAISS Index Builder — StoryAnimate Project")
print("=" * 55)

# ── Step 1: Load existing embeddings ─────────────────────────────
print(f"\nStep 1: Loading embeddings from:")
print(f"  {EMBEDDINGS_PATH}")

t0 = time.time()
embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
load_time  = time.time() - t0

print(f"  Shape  : {embeddings.shape}")
print(f"  Dtype  : {embeddings.dtype}")
print(f"  Size   : {embeddings.nbytes / 1024 / 1024:.1f} MB")
print(f"  Loaded : {load_time:.2f}s")

N, DIM = embeddings.shape
print(f"\n  Videos : {N:,}")
print(f"  Dims   : {DIM}")

# ── Step 2: Normalise for cosine similarity ───────────────────────
print("\nStep 2: Normalising vectors for cosine similarity...")
t0 = time.time()
faiss.normalize_L2(embeddings)           # in-place L2 normalisation
print(f"  Done in {time.time()-t0:.2f}s")

# ── Step 3: Build FAISS index ─────────────────────────────────────
# IndexFlatIP = Inner Product (= cosine similarity after L2 norm)
# This is exact search — same accuracy as your current numpy search
# but ~100x faster
print("\nStep 3: Building FAISS IndexFlatIP...")
t0    = time.time()
index = faiss.IndexFlatIP(DIM)           # IP = inner product = cosine
index.add(embeddings)                    # add all 213K vectors
build = time.time() - t0

print(f"  Vectors added : {index.ntotal:,}")
print(f"  Build time    : {build:.2f}s")

# ── Step 4: Save index to disk ────────────────────────────────────
print(f"\nStep 4: Saving index to:")
print(f"  {INDEX_PATH}")
t0 = time.time()
faiss.write_index(index, INDEX_PATH)
save_time = time.time() - t0

size_mb = os.path.getsize(INDEX_PATH) / 1024 / 1024
print(f"  Index size : {size_mb:.1f} MB")
print(f"  Saved in   : {save_time:.2f}s")

# ── Step 5: Verify — quick test search ───────────────────────────
print("\nStep 5: Verification — test search on first vector...")
test_vec = embeddings[:1].copy()         # use first embedding as query
D, I    = index.search(test_vec, 5)     # find top 5

print(f"  Top-5 indices    : {I[0].tolist()}")
print(f"  Top-5 scores     : {[round(float(s),4) for s in D[0]]}")
print(f"  First match idx  : {I[0][0]}  (should be 0 — exact self-match)")

if I[0][0] == 0:
    print("  Self-match check : PASSED ✓")
else:
    print("  Self-match check : FAILED — check embeddings file")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  BATCH 1 COMPLETE")
print("=" * 55)
print(f"  Embeddings loaded : {N:,} vectors × {DIM} dims")
print(f"  Index built       : IndexFlatIP (exact cosine)")
print(f"  Index saved to    : combined_faiss.index")
print(f"  Index size        : {size_mb:.1f} MB")
print(f"  Total time        : {load_time+build+save_time:.1f}s")
print()
print("  Next step → Run Batch 2 to load this index into app.py")
print("=" * 55)