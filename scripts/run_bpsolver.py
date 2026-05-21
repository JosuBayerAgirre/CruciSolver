import argparse
import csv
import math
import pickle
import string
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

DEVICE = 'cpu'  # overridden in main() to 'cuda' if available

UNIGRAM_PROBS = [
    ('A', 0.1253), ('B', 0.0142), ('C', 0.0468), ('D', 0.0586),
    ('E', 0.1368), ('F', 0.0069), ('G', 0.0101), ('H', 0.0070),
    ('I', 0.0625), ('J', 0.0044), ('K', 0.0002), ('L', 0.0497),
    ('M', 0.0315), ('N', 0.0671), ('Ñ', 0.0031), ('O', 0.0868),
    ('P', 0.0251), ('Q', 0.0088), ('R', 0.0687), ('S', 0.0798),
    ('T', 0.0463), ('U', 0.0393), ('V', 0.0090), ('W', 0.0001),
    ('X', 0.0022), ('Y', 0.0090), ('Z', 0.0052),
]
LETTER_SMOOTHING_FACTOR = [
    0.0, 0.0, 0.04, 0.001, 0.006, 0.002, 0.005, 0.013,
    0.027, 0.065, 0.125, 0.220, 0.232, 0.255, 0.399, 0.276,
    0.673, 0.682, 0.857, 0.825, 0.800, 0.719, 0.0,
]
# Spanish alphabet: A-Z plus Ñ (inserted after N)
CHARS = list(string.ascii_uppercase[:14]) + ['Ñ'] + list(string.ascii_uppercase[14:])
CHAR_IDX = {c: i for i, c in enumerate(CHARS)}
N_CHARS = len(CHARS)  # 27


def normalize(text):
    """Uppercase and keep only valid Spanish crossword characters (A-Z + Ñ)."""
    result = []
    for c in text.upper():
        if c in CHAR_IDX:
            result.append(c)
        else:
            # Try stripping combining marks (e.g. Á→A, É→E, but Ñ is already handled above)
            nfd = unicodedata.normalize('NFD', c)
            base = ''.join(x for x in nfd if unicodedata.category(x) != 'Mn')
            if base in CHAR_IDX:
                result.append(base)
    return ''.join(result)


class BPVar:
    def __init__(self, name, variable, candidates, cells):
        self.name = name
        cells_by_pos = {cell.position: cell for cell in cells}
        for cell in cells:
            cell._connect(self)
        self.length = len(cells)
        self.ordered_cells = [cells_by_pos[pos] for pos in variable['cells']]
        self.candidates = candidates
        self.words = list(candidates['words'])
        # (W, L) char index per word per position
        self.word_indices = torch.tensor(
            [[CHAR_IDX[c] for c in w] for w in self.words],
            dtype=torch.long, device=DEVICE,
        )
        scores = torch.tensor(
            [-candidates['weights'][w] for w in self.words],
            dtype=torch.float32, device=DEVICE,
        )
        self.prior_log_probs = F.log_softmax(scores, dim=0)
        self.log_probs = self.prior_log_probs.clone()
        self.directional_scores = [
            torch.zeros(len(self.log_probs), device=DEVICE)
            for _ in self.ordered_cells
        ]
        self.unigram = torch.tensor(
            [p for _, p in UNIGRAM_PROBS], dtype=torch.float32, device=DEVICE,
        )

    def _propagate_to_var(self, other, belief_state):
        idx = self.ordered_cells.index(other)
        self.directional_scores[idx] = belief_state[self.word_indices[:, idx]]

    def sync_state(self):
        self.log_probs = F.log_softmax(
            sum(self.directional_scores) + self.prior_log_probs, dim=0,
        )

    def propagate(self):
        sf = LETTER_SMOOTHING_FACTOR[min(self.length, len(LETTER_SMOOTHING_FACTOR) - 1)]
        bit_array = self.candidates['bit_array']  # (N_CHARS, L, W) on DEVICE
        for i, cell in enumerate(self.ordered_cells):
            word_probs = F.softmax(self.log_probs - self.directional_scores[i], dim=0)  # (W,)
            lp = (bit_array[:, i, :] * word_probs).sum(dim=1) + 1e-8  # (N_CHARS,)
            lp = (1 - sf) * lp + sf * self.unigram
            cell._propagate_to_cell(self, torch.log(lp))


class BPCell:
    def __init__(self, position, clue_pair):
        self.crossing_clues = clue_pair
        self.position = tuple(position)
        self.log_probs = torch.full((N_CHARS,), -math.log(N_CHARS), dtype=torch.float32, device=DEVICE)
        self.crossing_vars = []
        self.directional_scores = []

    def _connect(self, other):
        self.crossing_vars.append(other)
        self.directional_scores.append(None)
        assert len(self.crossing_vars) <= 2

    def _propagate_to_cell(self, other, belief_state):
        self.directional_scores[self.crossing_vars.index(other)] = belief_state

    def sync_state(self):
        valid = [s for s in self.directional_scores if s is not None]
        if valid:
            self.log_probs = F.log_softmax(sum(valid), dim=0)

    def propagate(self):
        if len(self.crossing_vars) == 2 and all(s is not None for s in self.directional_scores):
            for i, v in enumerate(self.crossing_vars):
                v._propagate_to_var(self, self.directional_scores[1 - i])


def parse_puzzle(puz_path, clue_lookup):
    """
    Grid structure and gold answers come from the .puz file (authoritative).
    Clue text comes from clue_lookup (CSV data) keyed by (puzzle_id, direction, answer),
    because the clue text stored in the .puz file is misaligned from the grid answers.
    """
    import puz as puzlib

    p = puzlib.read(str(puz_path))
    numbering = p.clue_numbering()
    height, width = p.height, p.width
    solution = p.solution
    puzzle_id = puz_path.stem

    letter_grid = []
    for r in range(height):
        row = []
        for c in range(width):
            ch = solution[r * width + c]
            row.append('' if ch == '.' else normalize(ch))
        letter_grid.append(row)

    variables = {}
    grid_cells = defaultdict(list)

    for i, pe in enumerate(numbering.across):
        r0 = pe['cell'] // width
        c0 = pe['cell'] % width
        puz_len = pe['len']
        cells = [(r0, c0 + j) for j in range(puz_len)]
        gold = ''.join(normalize(solution[r * width + c]) for r, c in cells)
        clue = clue_lookup.get((puzzle_id, 'across', gold), pe['clue'])
        wid = f'{i+1}A'
        variables[wid] = {'clue': clue, 'gold': gold, 'cells': cells, 'crossing': []}
        for cell in cells:
            if wid not in grid_cells[cell]:
                grid_cells[cell].append(wid)

    for i, pe in enumerate(numbering.down):
        r0 = pe['cell'] // width
        c0 = pe['cell'] % width
        puz_len = pe['len']
        cells = [(r0 + j, c0) for j in range(puz_len)]
        gold = ''.join(normalize(solution[r * width + c]) for r, c in cells)
        clue = clue_lookup.get((puzzle_id, 'down', gold), pe['clue'])
        wid = f'{i+1}D'
        variables[wid] = {'clue': clue, 'gold': gold, 'cells': cells, 'crossing': []}
        for cell in cells:
            if wid not in grid_cells[cell]:
                grid_cells[cell].append(wid)

    for wid, var in variables.items():
        for cell in var['cells']:
            for oid in grid_cells[cell]:
                if oid != wid and oid not in var['crossing']:
                    var['crossing'].append(oid)

    return variables, dict(grid_cells), letter_grid


def build_candidates(variables, vectorizer, train_matrix, train_answers, top_n):
    var_keys = list(variables.keys())
    vecs = vectorizer.transform([variables[v]['clue'] for v in var_keys])
    sims = cosine_similarity(vecs, train_matrix)

    candidates = {}
    for i, vk in enumerate(var_keys):
        length = len(variables[vk]['gold'])
        sim_row = sims[i]
        seen = {}
        for idx in sim_row.argsort()[::-1]:
            ans = normalize(train_answers[idx])
            if len(ans) == length and ans not in seen:
                seen[ans] = float(sim_row[idx])
            if len(seen) == top_n:
                break
        if not seen:
            for ans in set(normalize(a) for a in train_answers):
                if len(ans) == length:
                    seen[ans] = 1e-6
                if len(seen) >= top_n:
                    break

        words = list(seen.keys())
        sims_t = torch.tensor([seen[w] for w in words], dtype=torch.float32)
        cost_t = -torch.log(F.softmax(sims_t / 0.75, dim=0) + 1e-12)
        weights = {w: c.item() for w, c in zip(words, cost_t)}
        words_sorted = sorted(weights, key=weights.get)

        arr = np.zeros((N_CHARS, length, len(words_sorted)), dtype=np.float32)
        for wi, word in enumerate(words_sorted):
            for pi, ch in enumerate(word):
                if ch in CHAR_IDX:
                    arr[CHAR_IDX[ch], pi, wi] = 1.0

        candidates[vk] = {
            'words': words_sorted,
            'weights': weights,
            'bit_array': torch.tensor(arr, device=DEVICE),
        }
    return candidates


def generate_candidates_mt5(variables, model, tokenizer, device, num_beams=20, batch_size=8, max_src_len=64):
    def template(clue, length):
        return f"La pista de {length} letras es: {clue} Respuesta:"

    var_keys = list(variables.keys())
    all_generated = {}

    for i in range(0, len(var_keys), batch_size):
        batch_keys = var_keys[i:i + batch_size]
        batch_inputs = [template(variables[vk]['clue'], len(variables[vk]['gold'])) for vk in batch_keys]
        batch_lengths = [len(variables[vk]['gold']) for vk in batch_keys]

        enc = tokenizer(batch_inputs, return_tensors='pt', max_length=max_src_len,
                        truncation=True, padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=20,
                output_scores=True,
                return_dict_in_generate=True,
            )
        seqs = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        scores = out.sequences_scores.cpu().tolist()

        for j, (vk, length) in enumerate(zip(batch_keys, batch_lengths)):
            seen = {}
            for k in range(num_beams):
                idx = j * num_beams + k
                word = normalize(seqs[idx].strip())
                if len(word) == length and word not in seen:
                    seen[word] = scores[idx]
            all_generated[vk] = sorted(seen.items(), key=lambda x: -x[1])

    candidates = {}
    for vk, var in variables.items():
        cands = all_generated.get(vk, [])
        if not cands:
            cands = [(var['gold'], -1.0)]
        words = [w for w, _ in cands]
        weights = {w: -s for w, s in cands}   # cost = -log_prob, lower = better
        length = len(var['gold'])
        arr = np.zeros((N_CHARS, length, len(words)), dtype=np.float32)
        for wi, word in enumerate(words):
            for pi, ch in enumerate(word):
                if ch in CHAR_IDX:
                    arr[CHAR_IDX[ch], pi, wi] = 1.0
        candidates[vk] = {
            'words': words,
            'weights': weights,
            'bit_array': torch.tensor(arr, device=DEVICE),
        }
    return candidates


def build_candidates_combined(variables, model, tokenizer, device,
                              vectorizer, train_matrix, train_answers,
                              num_beams=20, gen_batch_size=8, top_n=200, max_src_len=64):
    """
    Merge ByT5 beam search candidates (quality) with TF-IDF candidates (recall).
    ByT5 words keep their log-prob scores; TF-IDF-only words get a penalty below all ByT5 scores.
    """
    mt5_cands = generate_candidates_mt5(
        variables, model, tokenizer, device,
        num_beams=num_beams, batch_size=gen_batch_size, max_src_len=max_src_len,
    )
    tfidf_cands = build_candidates(variables, vectorizer, train_matrix, train_answers, top_n)

    combined = {}
    for vk, var in variables.items():
        mt5_words  = mt5_cands[vk]['words']
        mt5_scores = {w: -mt5_cands[vk]['weights'][w] for w in mt5_words}  # log-prob (higher=better)

        tfidf_words  = tfidf_cands[vk]['words']
        tfidf_scores_raw = {w: -tfidf_cands[vk]['weights'][w] for w in tfidf_words}

        # TF-IDF-only words sit just below the worst ByT5 score to preserve ByT5 priority
        if mt5_scores:
            floor = min(mt5_scores.values()) - 2.0
        else:
            floor = -10.0

        if tfidf_scores_raw:
            tfidf_min = min(tfidf_scores_raw.values())
            tfidf_max = max(tfidf_scores_raw.values())
            tfidf_range = max(tfidf_max - tfidf_min, 1e-6)
            # Map TF-IDF scores to [floor-1, floor]
            tfidf_scores = {
                w: floor - 1.0 + (tfidf_scores_raw[w] - tfidf_min) / tfidf_range
                for w in tfidf_words
            }
        else:
            tfidf_scores = {}

        # ByT5 takes priority for shared words
        merged = {**tfidf_scores, **mt5_scores}
        words_sorted = sorted(merged, key=merged.get, reverse=True)

        length = len(var['gold'])
        weights = {w: -merged[w] for w in words_sorted}   # cost = -log_prob
        arr = np.zeros((N_CHARS, length, len(words_sorted)), dtype=np.float32)
        for wi, word in enumerate(words_sorted):
            for pi, ch in enumerate(word):
                if ch in CHAR_IDX:
                    arr[CHAR_IDX[ch], pi, wi] = 1.0

        combined[vk] = {
            'words': words_sorted,
            'weights': weights,
            'bit_array': torch.tensor(arr, device=DEVICE),
        }

    return combined


def direct_bp_fill(bp_vars, bp_cells, letter_grid):
    """
    Fill grid from BP results without hard letter constraints.
    Each variable picks its top-1 word; conflicted cells use cell marginals.
    Avoids the cascading error problem of greedy sequential fill.
    """
    grid = [['' for _ in row] for row in letter_grid]

    cell_votes = defaultdict(list)   # position → [(letter, var_log_prob)]
    for var in bp_vars:
        if not var.words:
            continue
        best_idx = var.log_probs.argmax().item()
        best_lp = var.log_probs[best_idx].item()
        best_word = var.words[best_idx]
        for j, cell in enumerate(var.ordered_cells):
            cell_votes[cell.position].append((best_word[j], best_lp))

    for cell in bp_cells:
        r, c = cell.position
        votes = cell_votes.get((r, c), [])
        if len(votes) == 1:
            grid[r][c] = votes[0][0]
        elif len(votes) > 1:
            # Conflict: use cell marginal (combines messages from all crossing vars)
            grid[r][c] = CHARS[cell.log_probs.argmax().item()]
        else:
            grid[r][c] = CHARS[cell.log_probs.argmax().item()]

    return grid


def greedy_sequential_fill(bp_vars, bp_cells, letter_grid):
    grid = [['' for _ in row] for row in letter_grid]
    unfilled = {cell.position for cell in bp_cells}
    cache = [(list(v.words), v.log_probs.clone(), v.word_indices.clone()) for v in bp_vars]

    sf = LETTER_SMOOTHING_FACTOR
    for var in bp_vars:
        var.log_probs = var.log_probs + math.log(
            max(1 - sf[min(var.length, len(sf) - 1)], 1e-12)
        )

    best_per_var = [v.log_probs.max().item() if v.log_probs.numel() else None for v in bp_vars]

    while any(x is not None for x in best_per_var):
        best_idx = max(
            (i for i, x in enumerate(best_per_var) if x is not None),
            key=lambda i: best_per_var[i],
        )
        best_var = bp_vars[best_idx]
        best_word = best_var.words[best_var.log_probs.argmax().item()]

        for i, cell in enumerate(best_var.ordered_cells):
            letter = best_word[i]
            grid[cell.position[0]][cell.position[1]] = letter
            unfilled.discard(cell.position)
            target_idx = CHAR_IDX[letter]
            for other in cell.crossing_vars:
                if other is not best_var:
                    ci = other.ordered_cells.index(cell)
                    keep = (other.word_indices[:, ci] == target_idx).nonzero(as_tuple=True)[0]
                    other.words = [other.words[j] for j in keep.tolist()]
                    other.log_probs = other.log_probs[keep]
                    other.word_indices = other.word_indices[keep]
                    vi = bp_vars.index(other)
                    best_per_var[vi] = other.log_probs.max().item() if keep.numel() > 0 else None

        best_var.words = []
        best_var.log_probs = torch.zeros(0, device=DEVICE)
        best_var.word_indices = torch.zeros((0, best_var.length), dtype=torch.long, device=DEVICE)
        best_per_var[best_idx] = None

    for cell in bp_cells:
        if cell.position in unfilled:
            grid[cell.position[0]][cell.position[1]] = CHARS[cell.log_probs.argmax().item()]

    for var, (words, lp, wi) in zip(bp_vars, cache):
        var.words = words
        var.log_probs = lp
        var.word_indices = wi

    return grid


def top1_fill(variables, candidates, letter_grid):
    grid = [['' for _ in row] for row in letter_grid]
    for vk, var in variables.items():
        word = candidates[vk]['words'][0]
        for j, (r, c) in enumerate(var['cells']):
            grid[r][c] = word[j]
    return grid


def evaluate(grid, letter_grid, variables):
    lc = lt = wc = wt = 0
    for r, row in enumerate(letter_grid):
        for c, gold in enumerate(row):
            if gold:
                lt += 1
                lc += gold == grid[r][c]
    for var in variables.values():
        wt += 1
        wc += ''.join(grid[r][c] for r, c in var['cells']) == var['gold']
    return lc, lt, wc, wt


def rerank_with_mt5(variables, candidates, model, tokenizer, device, batch_size=64):
    """Score every (clue, candidate_word) pair with mT5 and replace TF-IDF weights."""
    def template(clue, length):
        return f"La pista de {length} letras es: {clue} Respuesta:"

    all_inputs, all_targets, meta = [], [], []
    for vk, var in variables.items():
        inp = template(var['clue'], len(var['gold']))
        for wi, word in enumerate(candidates[vk]['words']):
            all_inputs.append(inp)
            all_targets.append(word)
            meta.append((vk, wi))

    all_scores = []
    for i in range(0, len(all_inputs), batch_size):
        b_inp = all_inputs[i:i + batch_size]
        b_tgt = all_targets[i:i + batch_size]
        enc = tokenizer(b_inp, return_tensors='pt', max_length=64,
                        truncation=True, padding=True).to(device)
        lab_enc = tokenizer(b_tgt, return_tensors='pt', max_length=16,
                            truncation=True, padding=True).to(device)
        labels = lab_enc.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            logits = model(**enc, labels=labels).logits        # (B, T, V)
            log_p = F.log_softmax(logits, dim=-1)
            safe_labels = labels.clamp(min=0)
            tok_lp = log_p.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
            mask = (labels != -100).float()
            scores = (tok_lp * mask).sum(dim=1)               # (B,)
        all_scores.extend(scores.cpu().tolist())

    var_word_scores = defaultdict(dict)
    for (vk, wi), score in zip(meta, all_scores):
        word = candidates[vk]['words'][wi]
        var_word_scores[vk][word] = score

    for vk, var in variables.items():
        scored = var_word_scores[vk]
        if not scored:
            continue
        words_sorted = sorted(scored, key=scored.get, reverse=True)
        weights = {w: -scored[w] for w in words_sorted}   # cost = -log_prob
        length = len(var['gold'])
        arr = np.zeros((N_CHARS, length, len(words_sorted)), dtype=np.float32)
        for wi, word in enumerate(words_sorted):
            for pi, ch in enumerate(word):
                if ch in CHAR_IDX:
                    arr[CHAR_IDX[ch], pi, wi] = 1.0
        candidates[vk] = {
            'words': words_sorted,
            'weights': weights,
            'bit_array': torch.tensor(arr, device=DEVICE),
        }
    return candidates


def local_search(grid, variables, bp_vars, candidates, steps=5, top_k=5):
    """
    Word-level iterative local search.
    For each variable (least-confident first), try the top-k candidates by post-BP
    log_probs and accept the swap that maximises the joint log-prob of that word
    plus all its crossing words.  Uses scores already in candidates — no extra
    forward passes required.
    """
    var_to_bp = {vk: bpv for vk, bpv in zip(variables.keys(), bp_vars)}

    cell_to_vars = defaultdict(list)
    for vk, var in variables.items():
        for cell in var['cells']:
            cell_to_vars[cell].append(vk)

    def read_word(g, cells):
        return ''.join(g[r][c] for r, c in cells)

    def word_log_prob(vk, word):
        w = candidates[vk]['weights']
        return -w[word] if word in w else -1000.0

    for step in range(steps):
        any_improved = False
        # Least-confident variables first (most room to improve)
        var_order = sorted(
            list(variables.keys()),
            key=lambda vk: var_to_bp[vk].log_probs.max().item(),
        )

        for vk in var_order:
            var = variables[vk]
            bpv = var_to_bp[vk]
            current_word = read_word(grid, var['cells'])

            order = bpv.log_probs.argsort(descending=True).tolist()
            proposals = [bpv.words[i] for i in order[:top_k] if bpv.words[i] != current_word]
            if not proposals:
                continue

            crossing_vks = set()
            for cell in var['cells']:
                crossing_vks.update(cell_to_vars.get(cell, []))
            crossing_vks.discard(vk)
            crossing_vks = list(crossing_vks)

            def score_placement(test_word):
                s = word_log_prob(vk, test_word)
                temp = [row[:] for row in grid]
                for j, (r, c) in enumerate(var['cells']):
                    temp[r][c] = test_word[j]
                for cvk in crossing_vks:
                    s += word_log_prob(cvk, read_word(temp, variables[cvk]['cells']))
                return s

            current_score = score_placement(current_word)
            best_delta = 0.0
            best_word = None

            for proposal in proposals:
                delta = score_placement(proposal) - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_word = proposal

            if best_word:
                for j, (r, c) in enumerate(var['cells']):
                    grid[r][c] = best_word[j]
                any_improved = True

        if not any_improved:
            break

    return grid


def local_search_mt5(grid, variables, bp_cells, model, tokenizer, device,
                     steps=5, batch_size=32, flip_threshold=0.01, max_src_len=64):
    """
    Letter-flip local search scored by ByT5.
    At each step proposes all single-letter flips at cells where BP assigns
    probability >= flip_threshold to an alternative letter, scores each proposal
    with ByT5 in one batched forward pass, and accepts the best improvement.
    Can recover words not present in the original candidate list.
    """
    def template(clue, length):
        return f"La pista de {length} letras es: {clue} Respuesta:"

    def read_word(g, cells):
        return ''.join(g[r][c] for r, c in cells)

    cell_to_vars = defaultdict(list)
    for vk, var in variables.items():
        for pos in var['cells']:
            cell_to_vars[pos].append(vk)

    def score_batch(inputs, targets):
        scores = []
        for i in range(0, len(inputs), batch_size):
            b_inp = inputs[i:i + batch_size]
            b_tgt = targets[i:i + batch_size]
            enc = tokenizer(b_inp, return_tensors='pt', max_length=max_src_len,
                            truncation=True, padding=True).to(device)
            lab = tokenizer(b_tgt, return_tensors='pt', max_length=16,
                            truncation=True, padding=True).to(device)
            labels = lab.input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            with torch.no_grad():
                logits = model(**enc, labels=labels).logits
                log_p = F.log_softmax(logits, dim=-1)
                tok_lp = log_p.gather(2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                scores.extend(
                    ((tok_lp * (labels != -100).float()).sum(dim=1)).cpu().tolist()
                )
        return scores

    for step in range(steps):
        # Build proposals: every letter flip with BP prob >= flip_threshold
        proposals = []  # (r, c, new_letter, affected_vks, new_words_dict)
        for cell in bp_cells:
            r, c = cell.position
            current = grid[r][c]
            probs = cell.log_probs.exp()
            affected = cell_to_vars.get((r, c), [])
            if not affected:
                continue
            for li, letter in enumerate(CHARS):
                if letter == current or probs[li].item() < flip_threshold:
                    continue
                new_words = {}
                for vk in affected:
                    cells = variables[vk]['cells']
                    pos_idx = cells.index((r, c))
                    chars = [grid[rr][cc] for rr, cc in cells]
                    chars[pos_idx] = letter
                    new_words[vk] = ''.join(chars)
                proposals.append((r, c, letter, affected, new_words))

        if not proposals:
            break

        unique_vks = list({vk for _, _, _, aff, _ in proposals for vk in aff})
        cur_scores = dict(zip(
            unique_vks,
            score_batch(
                [template(variables[vk]['clue'], len(variables[vk]['gold'])) for vk in unique_vks],
                [read_word(grid, variables[vk]['cells']) for vk in unique_vks],
            ),
        ))

        prop_inp, prop_tgt, prop_meta = [], [], []
        for pi, (r, c, letter, affected, new_words) in enumerate(proposals):
            for vk in affected:
                prop_inp.append(template(variables[vk]['clue'], len(variables[vk]['gold'])))
                prop_tgt.append(new_words[vk])
                prop_meta.append((pi, vk))

        prop_scores = score_batch(prop_inp, prop_tgt)

        deltas = defaultdict(float)
        for (pi, vk), s in zip(prop_meta, prop_scores):
            deltas[pi] += s - cur_scores[vk]

        best_pi = max(deltas, key=deltas.get)
        best_delta = deltas[best_pi]

        if best_delta <= 0:
            break

        r, c, letter, _, _ = proposals[best_pi]
        grid[r][c] = letter
        print(f'  LS-mT5 step {step+1}: [{r},{c}]→{letter} (Δ={best_delta:.3f})', flush=True)

    return grid


def load_mt5(model_dir, device):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).eval().to(device)
    return model, tokenizer


def _mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _encode(enc_model, input_ids, attention_mask):
    """Encode a batch — uses pooler_output for DPR models, mean pool otherwise."""
    from transformers import DPRQuestionEncoder, DPRContextEncoder
    out = enc_model(input_ids=input_ids, attention_mask=attention_mask)
    if isinstance(enc_model, (DPRQuestionEncoder, DPRContextEncoder)):
        return out.pooler_output
    return _mean_pool(out.last_hidden_state, attention_mask)


def load_biencoder(model_dir, device):
    from transformers import (AutoModel, AutoTokenizer,
                               DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
                               DPRContextEncoder, DPRContextEncoderTokenizer)
    import json, os
    def _is_dpr(path, kind):
        cfg = os.path.join(path, "config.json")
        if os.path.exists(cfg):
            with open(cfg) as f:
                return json.load(f).get("model_type", "") == "dpr"
        return False

    q_path = f"{model_dir}/query_encoder"
    d_path = f"{model_dir}/doc_encoder"

    if _is_dpr(q_path, "query"):
        q_tok = DPRQuestionEncoderTokenizer.from_pretrained(q_path)
        q_enc = DPRQuestionEncoder.from_pretrained(q_path).eval().to(device)
    else:
        q_tok = AutoTokenizer.from_pretrained(q_path)
        q_enc = AutoModel.from_pretrained(q_path).eval().to(device)

    if _is_dpr(d_path, "doc"):
        d_tok = DPRContextEncoderTokenizer.from_pretrained(d_path)
        d_enc = DPRContextEncoder.from_pretrained(d_path).eval().to(device)
    else:
        d_tok = AutoTokenizer.from_pretrained(d_path)
        d_enc = AutoModel.from_pretrained(d_path).eval().to(device)

    return q_enc, d_enc, q_tok, d_tok


def build_candidates_biencoder(variables, q_enc, d_enc, q_tok, d_tok,
                                train_answers, device, top_n=200,
                                batch_size=256, max_q_len=128, max_d_len=32):
    """
    Dense retrieval using a fine-tuned bi-encoder.
    For each clue, encodes the query and retrieves top-N training answers of
    the correct length by cosine similarity.
    """
    answers_by_len = defaultdict(list)
    seen = set()
    for a in train_answers:
        na = normalize(a)
        if na and na not in seen:
            seen.add(na)
            answers_by_len[len(na)].append(na)

    doc_cache = {}  # length → (emb tensor [N, D], list of words)

    @torch.no_grad()
    def get_doc_embs(length):
        if length in doc_cache:
            return doc_cache[length]
        words = answers_by_len.get(length, [])
        if not words:
            doc_cache[length] = (None, [])
            return None, []
        all_embs = []
        for i in range(0, len(words), batch_size):
            chunk = words[i:i + batch_size]
            enc = d_tok(chunk, padding=True, truncation=True,
                        max_length=max_d_len, return_tensors="pt").to(device)
            emb = F.normalize(_encode(d_enc, enc["input_ids"], enc["attention_mask"]), dim=-1)
            all_embs.append(emb.cpu())
        embs = torch.cat(all_embs, dim=0)
        doc_cache[length] = (embs, words)
        return embs, words

    var_keys = list(variables.keys())
    candidates = {}

    for i in range(0, len(var_keys), batch_size):
        batch_keys = var_keys[i:i + batch_size]
        queries = [f"Pista de {len(variables[vk]['gold'])} letras: {variables[vk]['clue']}"
                   for vk in batch_keys]
        with torch.no_grad():
            enc = q_tok(queries, padding=True, truncation=True,
                        max_length=max_q_len, return_tensors="pt").to(device)
            q_embs = F.normalize(_encode(q_enc, enc["input_ids"], enc["attention_mask"]), dim=-1).cpu()

        for j, vk in enumerate(batch_keys):
            length = len(variables[vk]['gold'])
            d_embs, words = get_doc_embs(length)

            if d_embs is None or len(words) == 0:
                candidates[vk] = {
                    'words': [variables[vk]['gold']],
                    'weights': {variables[vk]['gold']: 0.0},
                    'bit_array': torch.zeros((N_CHARS, length, 1), device=DEVICE),
                }
                continue

            scores = torch.mv(d_embs, q_embs[j])  # cosine sim (already normalised)
            top_k = min(top_n, len(words))
            top_idx = scores.topk(top_k).indices.tolist()

            seen_w = {}
            for idx in top_idx:
                w = words[idx]
                seen_w[w] = scores[idx].item()

            words_sorted = sorted(seen_w, key=seen_w.get, reverse=True)
            sim_t = torch.tensor([seen_w[w] for w in words_sorted])
            cost_t = -torch.log(F.softmax(sim_t / 0.5, dim=0) + 1e-12)
            weights = {w: c.item() for w, c in zip(words_sorted, cost_t)}

            arr = np.zeros((N_CHARS, length, len(words_sorted)), dtype=np.float32)
            for wi, word in enumerate(words_sorted):
                for pi, ch in enumerate(word):
                    if ch in CHAR_IDX:
                        arr[CHAR_IDX[ch], pi, wi] = 1.0

            candidates[vk] = {
                'words': words_sorted,
                'weights': weights,
                'bit_array': torch.tensor(arr, device=DEVICE),
            }

    return candidates


def build_candidates_combined_be(variables, mt5_model, mt5_tok, mt5_device,
                                  q_enc, d_enc, q_tok, d_tok, train_answers,
                                  num_beams=20, gen_batch_size=8, top_n=200,
                                  max_src_len=64, be_batch_size=256):
    """
    ByT5 beam search (quality) merged with bi-encoder retrieval (recall).
    Same merging logic as build_candidates_combined but replaces TF-IDF.
    """
    mt5_cands = generate_candidates_mt5(
        variables, mt5_model, mt5_tok, mt5_device,
        num_beams=num_beams, batch_size=gen_batch_size, max_src_len=max_src_len,
    )
    be_cands = build_candidates_biencoder(
        variables, q_enc, d_enc, q_tok, d_tok, train_answers,
        device=mt5_device, top_n=top_n, batch_size=be_batch_size,
    )

    combined = {}
    for vk, var in variables.items():
        mt5_words  = mt5_cands[vk]['words']
        mt5_scores = {w: -mt5_cands[vk]['weights'][w] for w in mt5_words}

        be_words  = be_cands[vk]['words']
        be_scores_raw = {w: -be_cands[vk]['weights'][w] for w in be_words}

        # BiEncoder-only words sit just below the worst ByT5 score
        if mt5_scores:
            floor = min(mt5_scores.values()) - 2.0
        else:
            floor = -10.0

        if be_scores_raw:
            be_min = min(be_scores_raw.values())
            be_max = max(be_scores_raw.values())
            be_range = max(be_max - be_min, 1e-6)
            # Map BiEncoder scores to [floor-1, floor]
            be_scores = {
                w: floor - 1.0 + (be_scores_raw[w] - be_min) / be_range
                for w in be_words
            }
        else:
            be_scores = {}

        # ByT5 takes priority for shared words
        merged = {**be_scores, **mt5_scores}
        words_sorted = sorted(merged, key=merged.get, reverse=True)

        length = len(var['gold'])
        weights = {w: -merged[w] for w in words_sorted}
        arr = np.zeros((N_CHARS, length, len(words_sorted)), dtype=np.float32)
        for wi, word in enumerate(words_sorted):
            for pi, ch in enumerate(word):
                if ch in CHAR_IDX:
                    arr[CHAR_IDX[ch], pi, wi] = 1.0

        combined[vk] = {
            'words': words_sorted,
            'weights': weights,
            'bit_array': torch.tensor(arr, device=DEVICE),
        }

    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfidf-model', default='output/tfidf_model.pkl')
    parser.add_argument('--puzzles-dir', default='output/puzzles')
    parser.add_argument('--split-file', default='data/dev.csv')
    parser.add_argument('--top-n', type=int, default=500)
    parser.add_argument('--bp-iters', type=int, default=10)
    parser.add_argument('--max-puzzles', type=int, default=None)
    parser.add_argument('--no-bp', action='store_true',
                        help='Skip BP entirely — use top-1 TF-IDF per clue (retrieval baseline)')
    parser.add_argument('--rerank', action='store_true',
                        help='Rerank TF-IDF candidates with ByT5 before BP')
    parser.add_argument('--rerank-batch-size', type=int, default=64)
    parser.add_argument('--mt5-generate', action='store_true',
                        help='Use ByT5 beam search to generate candidates (replaces TF-IDF retrieval)')
    parser.add_argument('--combine', action='store_true',
                        help='Merge ByT5 beam candidates (quality) with TF-IDF candidates (recall)')
    parser.add_argument('--num-beams', type=int, default=20)
    parser.add_argument('--gen-batch-size', type=int, default=8)
    parser.add_argument('--no-greedy', action='store_true',
                        help='Use direct BP argmax fill instead of greedy sequential fill')
    parser.add_argument('--local-search', action='store_true',
                        help='Enable word-level local search after BP (no extra ByT5 calls)')
    parser.add_argument('--ls-steps', type=int, default=10)
    parser.add_argument('--ls-top-k', type=int, default=5)
    parser.add_argument('--ls-mt5', action='store_true',
                        help='Enable ByT5-scored letter-flip local search after word-level LS')
    parser.add_argument('--ls-mt5-steps', type=int, default=10)
    parser.add_argument('--ls-mt5-batch-size', type=int, default=32)
    parser.add_argument('--mt5-model', default='checkpoints/byt5_base_aug',
                        help='Path to fine-tuned ByT5 checkpoint')
    parser.add_argument('--mt5-max-src-len', type=int, default=256,
                        help='Max source length for tokenizer (256 for ByT5 byte-level)')
    parser.add_argument('--biencoder', default=None,
                        help='Path to bi-encoder checkpoint dir (expects query_encoder/ and doc_encoder/ subdirs)')
    parser.add_argument('--be-top-n', type=int, default=200)
    parser.add_argument('--be-batch-size', type=int, default=256)
    args = parser.parse_args()

    global DEVICE
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {DEVICE}', flush=True)

    print('Loading TF-IDF model...', flush=True)
    with open(args.tfidf_model, 'rb') as f:
        tfidf = pickle.load(f)
    vectorizer = tfidf['vectorizer']
    train_matrix = tfidf['train_matrix']
    train_answers = tfidf['train_answers']

    mt5_model = mt5_tok = mt5_device = None
    if args.rerank or args.ls_mt5 or args.mt5_generate or args.combine:
        mt5_device = DEVICE
        print(f'Loading ByT5 from {args.mt5_model} on {mt5_device}...', flush=True)
        mt5_model, mt5_tok = load_mt5(args.mt5_model, mt5_device)

    be_q_enc = be_d_enc = be_q_tok = be_d_tok = None
    if args.biencoder:
        print(f'Loading bi-encoder from {args.biencoder}...', flush=True)
        be_q_enc, be_d_enc, be_q_tok, be_d_tok = load_biencoder(args.biencoder, DEVICE)

    # Build clue lookup from ALL csv splits so every puzzle's clues are covered.
    # Key: (puzzle_id, direction, normalized_answer) → clue text
    clue_lookup = {}
    data_dir = Path(args.split_file).parent
    for split in ('train', 'dev', 'test'):
        split_path = data_dir / f'{split}.csv'
        if not split_path.exists():
            continue
        with open(split_path) as f:
            for row in csv.DictReader(f):
                key = (row['puzzle_id'], row['direction'], normalize(row['answer']))
                clue_lookup[key] = row['clue']
    print(f'Clue lookup: {len(clue_lookup)} entries', flush=True)

    test_ids = set()
    with open(args.split_file) as f:
        for row in csv.DictReader(f):
            test_ids.add(row['puzzle_id'])
    print(f'Test puzzles: {len(test_ids)}', flush=True)

    total_lc = total_lt = total_wc = total_wt = perfect = evaluated = 0

    for puz_path in sorted(Path(args.puzzles_dir).glob('*.puz')):
        if puz_path.stem not in test_ids:
            continue
        try:
            variables, grid_cells, letter_grid = parse_puzzle(puz_path, clue_lookup)
        except Exception as e:
            print(f'  skip {puz_path.stem}: {e}', flush=True)
            continue
        if not variables:
            continue

        if args.combine and args.biencoder:
            candidates = build_candidates_combined_be(
                variables, mt5_model, mt5_tok, mt5_device,
                be_q_enc, be_d_enc, be_q_tok, be_d_tok, train_answers,
                num_beams=args.num_beams, gen_batch_size=args.gen_batch_size,
                top_n=args.be_top_n, max_src_len=args.mt5_max_src_len,
                be_batch_size=args.be_batch_size,
            )
        elif args.combine:
            candidates = build_candidates_combined(
                variables, mt5_model, mt5_tok, mt5_device,
                vectorizer, train_matrix, train_answers,
                num_beams=args.num_beams, gen_batch_size=args.gen_batch_size,
                top_n=args.top_n, max_src_len=args.mt5_max_src_len,
            )
        elif args.biencoder:
            candidates = build_candidates_biencoder(
                variables, be_q_enc, be_d_enc, be_q_tok, be_d_tok,
                train_answers, device=DEVICE, top_n=args.be_top_n,
                batch_size=args.be_batch_size,
            )
        elif args.mt5_generate:
            candidates = generate_candidates_mt5(
                variables, mt5_model, mt5_tok, mt5_device,
                num_beams=args.num_beams, batch_size=args.gen_batch_size,
                max_src_len=args.mt5_max_src_len,
            )
        else:
            candidates = build_candidates(variables, vectorizer, train_matrix, train_answers, args.top_n)
            if args.rerank:
                candidates = rerank_with_mt5(
                    variables, candidates, mt5_model, mt5_tok, mt5_device,
                    batch_size=args.rerank_batch_size,
                )

        if args.no_bp:
            grid = top1_fill(variables, candidates, letter_grid)
        else:
            bp_cells_map = {pos: BPCell(pos, cp) for pos, cp in grid_cells.items()}
            bp_cells_by_clue = defaultdict(list)
            for pos, cell in bp_cells_map.items():
                for cid in grid_cells[pos]:
                    bp_cells_by_clue[cid].append(cell)

            bp_vars = [BPVar(vk, vv, candidates[vk], bp_cells_by_clue[vk]) for vk, vv in variables.items()]
            bp_cells = list(bp_cells_map.values())

            for _ in range(args.bp_iters):
                for var in bp_vars:
                    var.propagate()
                for cell in bp_cells:
                    cell.sync_state()
                for cell in bp_cells:
                    cell.propagate()
                for var in bp_vars:
                    var.sync_state()

            if args.no_greedy:
                grid = direct_bp_fill(bp_vars, bp_cells, letter_grid)
            else:
                grid = greedy_sequential_fill(bp_vars, bp_cells, letter_grid)

        if args.local_search and not args.no_bp:
            grid = local_search(
                grid, variables, bp_vars, candidates,
                steps=args.ls_steps, top_k=args.ls_top_k,
            )
        if args.ls_mt5 and not args.no_bp:
            grid = local_search_mt5(
                grid, variables, bp_cells,
                mt5_model, mt5_tok, mt5_device,
                steps=args.ls_mt5_steps, batch_size=args.ls_mt5_batch_size,
                max_src_len=args.mt5_max_src_len,
            )

        lc, lt, wc, wt = evaluate(grid, letter_grid, variables)
        total_lc += lc; total_lt += lt; total_wc += wc; total_wt += wt
        evaluated += 1
        if wc == wt:
            perfect += 1

        if evaluated % 10 == 0:
            print(
                f'  {evaluated} puzzles | '
                f'letter={total_lc/max(total_lt,1):.4f} | '
                f'word={total_wc/max(total_wt,1):.4f} | '
                f'perfect={perfect}/{evaluated}',
                flush=True,
            )

        if args.max_puzzles and evaluated >= args.max_puzzles:
            break

    fill = 'direct' if args.no_greedy else 'greedy'
    ls_suffix = ''
    if args.local_search:
        ls_suffix += ' + local-search'
    if args.ls_mt5:
        ls_suffix += ' + ls-mt5'
    if args.no_bp:
        tag = 'TF-IDF top-1 (no BP)'
    elif args.combine and args.biencoder:
        tag = f'ByT5+BiEncoder combined + BP ({fill}){ls_suffix}'
    elif args.combine:
        tag = f'ByT5+TF-IDF combined + BP ({fill}){ls_suffix}'
    elif args.biencoder:
        tag = f'BiEncoder + BP ({fill}){ls_suffix}'
    elif args.mt5_generate:
        tag = f'ByT5-generate + BP ({fill}){ls_suffix}'
    elif args.rerank:
        tag = f'TF-IDF + ByT5-rerank + BP ({fill}){ls_suffix}'
    else:
        tag = f'TF-IDF + BP ({fill}){ls_suffix}'
    print(flush=True)
    print(f'=== BPSolver ({tag}) Results ===', flush=True)
    print(f'Puzzles evaluated:  {evaluated}', flush=True)
    print(f'Letter accuracy:    {total_lc/max(total_lt,1):.4f}', flush=True)
    print(f'Word accuracy:      {total_wc/max(total_wt,1):.4f}', flush=True)
    print(f'Perfect puzzles:    {perfect}/{evaluated} ({perfect/max(evaluated,1):.4f})', flush=True)


if __name__ == '__main__':
    main()
