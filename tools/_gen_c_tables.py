#!/usr/bin/env python3
"""Generate src/model/tokenizer_tables.h from the frozen Qwen2Tokenizer.

Reads models/dev/vocab.json (151643 entries), models/dev/merges.txt
(151387 merges, rank = line index) and models/dev/tokenizer.json
(added tokens + pretokenizer regex) and emits the FULL embedded C tables:

  * hd_tok_byte_piece[256]   GPT-2 byte -> BPE piece string
  * hd_tok_vocab_id/str[]    every vocab entry, sorted by id
  * hd_tok_merge_left/right/rank[]  every merge, sorted by rank
  * hd_tok_special_str/id[]  all 31 added tokens

The Python-side oracle encode() below reproduces the fast-tokenizer
pipeline (NFC normalize -> added-token extraction -> Unicode regex
pre-tokenization -> greedy priority-queue BPE) and is used to keep the
canonical-prompt assertion green.
"""

import heapq
import json
import re
import sys
import unicodedata

import regex  # supports \p{L}/\p{N} like the onig backend

sys.setrecursionlimit(10**6)

vocab = json.load(open("models/dev/vocab.json"))
merges_lines = open("models/dev/merges.txt").read().splitlines()
merges = []
for line in merges_lines:
    line = line.replace("\ufeff", "").rstrip("\n")
    if line == "":
        continue
    p = line.split(" ")
    if len(p) == 2:
        merges.append((p[0], p[1]))
    else:
        raise SystemExit("bad merge line: %r" % line)
assert len(merges) == 151387, len(merges)
merges_rank = {m: i for i, m in enumerate(merges)}
# merge info keyed by token-id pair: (rank, new_id)
merge_info = {}
for (a, b), rank in merges_rank.items():
    merge_info[(vocab[a], vocab[b])] = (rank, vocab[a + b])
assert len(merge_info) == 151387

# GPT-2 byte -> unicode mapping (same as the original generator).
bs = list(range(0x21, 0x7E + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
b2u = {b: chr(c) for b, c in zip(bs, cs)}
u2b = {c: b for b, c in b2u.items()}

# Oracle pretokenizer regex (from models/dev/tokenizer.json, onig backend).
PAT = regex.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# All 31 added tokens from tokenizer.json (authoritative; tokenizer_config.json
# only carries 26 of them).
added = json.load(open("models/dev/tokenizer.json"))["added_tokens"]
added.sort(key=lambda a: a["id"])
specials = {a["content"]: a["id"] for a in added}
assert len(specials) == 31

template = "<|im_start|>user\na red fox sits under a cherry blossom tree<|im_end|>\n<|im_start|>assistant\n"
expect = [151644, 872, 198, 64, 2518, 38835, 23011, 1212, 264, 40880, 88758, 4916, 151645, 198, 151644, 77091, 198]


def bpe_word(pieces):
    """Greedy BPE on a list of piece strings, fast-tokenizer merge_all
    semantics: min-heap ordered by (rank, pos); after each merge only the two
    new neighbor pairs are re-pushed; expired entries are skipped."""
    # symbol: [c, prev, next, len]  (c = vocab id of the piece string)
    syms = []
    for p in pieces:
        i = len(syms)
        if syms:
            syms[-1][2] = i
        syms.append([vocab[p], i - 1, -1, 1])

    heap = []
    for i in range(len(syms) - 1):
        pair = (syms[i][0], syms[i + 1][0])
        if pair in merge_info:
            rank, new_id = merge_info[pair]
            heap.append((rank, i, new_id))
    heapq.heapify(heap)

    while heap:
        rank, pos, new_id = heapq.heappop(heap)
        if syms[pos][3] == 0:
            continue
        if syms[pos][2] == -1:
            continue
        nxt = syms[pos][2]
        right = syms[nxt]
        target = (syms[pos][0], right[0])
        if target not in merge_info or merge_info[target][1] != new_id:
            continue
        # merge left into right position
        syms[pos][0] = new_id
        syms[pos][3] += right[3]
        syms[pos][2] = right[2]
        right[3] = 0
        if right[2] > -1:
            syms[right[2]][1] = pos
        # new pair with previous
        if syms[pos][1] >= 0:
            prev = syms[pos][1]
            pair = (syms[prev][0], syms[pos][0])
            if pair in merge_info:
                heapq.heappush(heap, (merge_info[pair][0], prev, merge_info[pair][1]))
        # new pair with next
        nxt2 = syms[pos][2]
        if nxt2 < len(syms):
            pair = (syms[pos][0], syms[nxt2][0])
            if pair in merge_info:
                heapq.heappush(heap, (merge_info[pair][0], pos, merge_info[pair][1]))

    return [syms[i][0] for i in range(len(syms)) if syms[i][3] != 0]


def encode(text):
    text = unicodedata.normalize("NFC", text)
    ids = []
    i = 0
    n = len(text)
    while i < n:
        # added-token extraction: longest match at position i (all 31 tokens)
        matched = None
        for s, sid in specials.items():
            if text.startswith(s, i):
                if matched is None or len(s) > len(matched[0]):
                    matched = (s, sid)
        if matched:
            ids.append(matched[1])
            i += len(matched[0])
            continue
        j = i
        while i < n and not any(text.startswith(s, i) for s in specials):
            i += 1
        seg = text[j:i]
        for m in PAT.finditer(seg):
            w = seg[m.start():m.end()]
            pieces = [b2u[b] for b in w.encode("utf-8")]
            ids.extend(bpe_word(pieces))
    return ids


got = encode(template)
assert got == expect, (got, expect)

# ---------------------------------------------------------------------------
# Emit C tables
# ---------------------------------------------------------------------------


def esc_c(s):
    b = s.encode("utf-8")
    out = '"'
    for byte in b:
        if byte == 0x22:
            out += '\\"'
        elif byte == 0x5C:
            out += "\\\\"
        elif 0x20 <= byte < 0x7F:
            out += chr(byte)
        else:
            out += "\\%03o" % byte
    out += '"'
    return out


byte_pieces = [b2u[b] for b in range(256)]
# Vocab sorted by piece string for binary-search lookup in C.
subset = sorted(vocab.items(), key=lambda kv: kv[0])  # (piece, id) sorted by piece
assert len(subset) == 151643
# Merge table sorted by (left, right) for binary-search rank lookup in C.
merges_sorted = sorted(merges, key=lambda m: (m[0], m[1]))
assert len(merges_sorted) == 151387

lines = []
lines.append("/* Generated by tools/_gen_c_tables.py from the frozen Qwen2Tokenizer")
lines.append("   (models/dev/vocab.json + merges.txt + tokenizer.json).")
lines.append("   DO NOT EDIT BY HAND. */")
lines.append("#ifndef HIDREAM_TOKENIZER_TABLES_H")
lines.append("#define HIDREAM_TOKENIZER_TABLES_H")
lines.append("")
lines.append("/* Byte -> BPE piece string (UTF-8), GPT-2 byte-to-unicode mapping. */")
lines.append("static const char *const hd_tok_byte_piece[256] = {")
for b in range(256):
    lines.append("    %s%s" % (esc_c(byte_pieces[b]), "," if b < 255 else ""))
lines.append("};")
lines.append("")
lines.append("/* Full vocab: (id, piece string). Sorted by piece string for binary search. */")
lines.append("static const int hd_tok_vocab_count = %d;" % len(subset))
lines.append("static const int hd_tok_vocab_id[%d] = {" % len(subset))
for i, (p, t) in enumerate(subset):
    lines.append("    %d%s" % (t, "," if i < len(subset) - 1 else ""))
lines.append("};")
lines.append("static const char *const hd_tok_vocab_str[%d] = {" % len(subset))
for i, (p, t) in enumerate(subset):
    lines.append("    %s%s" % (esc_c(p), "," if i < len(subset) - 1 else ""))
lines.append("};")
lines.append("")
lines.append("/* Full merge table: left, right, rank. Sorted by (left,right) for binary search. */")
lines.append("static const int hd_tok_merge_count = %d;" % len(merges_sorted))
lines.append("static const char *const hd_tok_merge_left[%d] = {" % len(merges_sorted))
for i, (a, b) in enumerate(merges_sorted):
    lines.append("    %s%s" % (esc_c(a), "," if i < len(merges_sorted) - 1 else ""))
lines.append("};")
lines.append("static const char *const hd_tok_merge_right[%d] = {" % len(merges_sorted))
for i, (a, b) in enumerate(merges_sorted):
    lines.append("    %s%s" % (esc_c(b), "," if i < len(merges_sorted) - 1 else ""))
lines.append("};")
lines.append("static const int hd_tok_merge_rank[%d] = {" % len(merges_sorted))
for i, (a, b) in enumerate(merges_sorted):
    lines.append("    %d%s" % (merges_rank[(a, b)], "," if i < len(merges_sorted) - 1 else ""))
lines.append("};")
lines.append("")
lines.append("/* Added tokens (content, id): all 31, matched longest-first. */")
spec_sel = [a["content"] for a in added]
lines.append("static const int hd_tok_special_count = %d;" % len(spec_sel))
lines.append("static const char *const hd_tok_special_str[%d] = {" % len(spec_sel))
for i, s in enumerate(spec_sel):
    lines.append("    %s%s" % (esc_c(s), "," if i < len(spec_sel) - 1 else ""))
lines.append("};")
lines.append("static const int hd_tok_special_id[%d] = {" % len(spec_sel))
for i, s in enumerate(spec_sel):
    lines.append("    %d%s" % (specials[s], "," if i < len(spec_sel) - 1 else ""))
lines.append("};")
lines.append("")
lines.append("#endif")
out = "\n".join(lines) + "\n"
open("src/model/tokenizer_tables.h", "w").write(out)
print("bytes:", len(out))
print("vocab entries:", len(subset))
print("merges:", len(merges))
print("special tokens:", len(spec_sel))
print("GOT:", got)
print("MATCH:", got == expect)
