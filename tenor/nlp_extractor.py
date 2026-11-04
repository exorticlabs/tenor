"""
Linguistic NLP Extractor & Salience Gating (nlp_extractor.py)
============================================================

Provides deterministic, high-speed linguistic entity extraction via Direct spaCy NER,
universal state-transition (mutation) detection, slotted attribute parsing,
syntactic "what" category resolution, and turn salience scoring without
external LLM dependencies, regex matchers, or hardcoded dictionaries.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Module-level spaCy singleton
_SPACY_NLP = None
_SPACY_LOCK = threading.Lock()



class _DummyDoc:
    def __init__(self, text):
        self.text = text
        self.ents = []
        self.sents = [self]
    def __iter__(self):
        return iter([])
    def __len__(self):
        return 0

class _DummyNLP:
    def __call__(self, text):
        return _DummyDoc(text)
    def pipe(self, texts, *args, **kwargs):
        for t in texts:
            yield _DummyDoc(t)

def get_spacy_nlp():
    """Lazily loads and caches the spaCy language model."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        with _SPACY_LOCK:
            if _SPACY_NLP is None:
                try:
                    import spacy
                    try:
                        _SPACY_NLP = spacy.load("en_core_web_sm")
                    except Exception as e:
                        logger.warning(f"Could not load en_core_web_sm directly: {e}. Falling back to blank...")
                        _SPACY_NLP = spacy.blank("en")
                        if "sentencizer" not in _SPACY_NLP.pipe_names:
                            _SPACY_NLP.add_pipe("sentencizer")
                except Exception as e:
                    logger.warning(f"spaCy unavailable ({e}). Using robust fallback...")
                    _SPACY_NLP = _DummyNLP()
    return _SPACY_NLP


class FastNLPExtractor:
    """
    High-throughput linguistic NLP extractor for agent execution loops:
    - Extracts named entities via Direct spaCy NER (doc.ents) with grammatical guards.
    - Zero regex patterns and zero hardcoded word lists: uses universal POS tags and dependency grammar.
    - Detects state-transition mutations using prepositional dependency grammar.
    - Resolves 'what' category using Syntactic Copula / Apposition / Compound parsing.
    - Evaluates turn salience using grammatical markers and entity density.
    """

    @staticmethod
    def is_valid_entity(name: str) -> bool:
        """
        Validates whether a candidate string is a genuine named entity/concept rather
        than a conversational sentence fragment, punctuation, or scalar number.
        Pure grammatical check: requires substantive noun/proper noun head and no non-noun root.
        Zero regex matchers.
        """
        clean = name.strip(" ,.:;\"'()[]{}<>`~*#\n\r\t")
        if not clean or len(clean) < 2 or len(clean) > 50:
            return False

        # Reject file paths, directories, and URI/API endpoints
        if "/" in clean or "\\" in clean:
            return False

        # Reject HTML tags or markup syntax
        if "<" in clean or ">" in clean:
            return False

        # Reject multiline text
        if "\n" in clean or "\r" in clean:
            return False

        nlp = get_spacy_nlp()
        doc = nlp(clean)
        if not doc or len(doc) == 0:
            return False

        # Reject if entirely composed of stopwords, numbers, or punctuation
        if all(t.is_stop or t.like_num or t.is_punct for t in doc):
            return False

        # Reject sentences ending with interrogative/exclamatory punctuation
        if any(t.text in ("?", "!", ";") and t.pos_ == "PUNCT" for t in doc):
            return False

        is_named_or_tech = (
            any(c.isupper() for c in clean[1:])
            or clean[0].isupper()
        )

        # Must have at least one substantive noun, proper noun, or technical identifier
        has_substantive = any(t.pos_ in ("PROPN", "NOUN") and not t.is_stop and not t.like_num for t in doc) or is_named_or_tech
        if not has_substantive:
            return False

        # If not a recognized named/tech token, head/root token must not be a verb, auxiliary, etc.
        if not is_named_or_tech:
            root = doc[0]
            for t in doc:
                if t.dep_ == "ROOT":
                    root = t
                    break
            if root.pos_ in ("VERB", "AUX", "ADV", "INTJ", "PRON", "CCONJ", "SCONJ", "ADP"):
                return False

            # Initial token cannot be a verb, auxiliary, adverb, interjection, or pronoun
            if doc[0].pos_ in ("VERB", "AUX", "ADV", "INTJ", "PRON", "CCONJ", "SCONJ", "ADP"):
                return False

        return True

    def __init__(self, max_entities_per_turn: int = 10, embedder: Any = None) -> None:
        self.max_entities_per_turn = max_entities_per_turn
        self.embedder = embedder

    @staticmethod
    def clean_text_for_parsing(text: str) -> str:
        """
        Strips code blocks, bullet points, and headers without regex.
        Uses fast string manipulation.
        """
        if not text:
            return ""

        # 1. Strip code fences (keep only non-code segments)
        if "```" in text:
            parts = text.split("```")
            text = " ".join(parts[i] for i in range(0, len(parts), 2))

        # 3. Strip line-leading bullets, numbers, and markdown headers
        lines = []
        for line in text.splitlines():
            clean_l = line.lstrip(" \t#*->1234567890.")
            if clean_l:
                lines.append(clean_l)
        text = " ".join(lines)

        # 4. Strip inline markdown symbols
        for ch in "`*#_~<>":
            text = text.replace(ch, " ")

        return " ".join(text.split()).strip()

    def extract_entities(
        self,
        text: str,
        active_entities: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Deterministic entity extraction using Direct spaCy NER (doc.ents) + Grammatical Guard.
        Zero noun chunks, zero regexes.
        
        If active_entities is provided:
        - Pre-checks turn against active entities (Resolution-First).
        - Decomposes compound mentions into existing active entities.
        - Resolves existing active entities before generating new ones.
        """
        if not text or not text.strip():
            return []

        clean_text = self.clean_text_for_parsing(text)
        if not clean_text:
            return []

        candidates: Dict[str, str] = {}
        active_set = {a.lower().strip() for a in (active_entities or []) if a}
        turn_lower = clean_text.lower()

        # -------------------------------------------------------------
        # 1. Pre-check: Resolve active entities already present in turn
        # -------------------------------------------------------------
        for a_lower in active_set:
            if a_lower in turn_lower:
                candidates[a_lower] = a_lower

        # -------------------------------------------------------------
        # 2. Direct spaCy NER (doc.ents) with 2-line Grammatical Guard
        # -------------------------------------------------------------
        nlp = get_spacy_nlp()
        doc = nlp(clean_text)

        for ent in doc.ents:
            # Exclude non-conceptual entities (pure dates, times, quantities, and numeric cardinals)
            if ent.root.like_num or ent.root.pos_ in ("NUM", "SYM"):
                continue
            if ent.label_ in ("DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"):
                continue

            # Guard 1: Root must NOT be a verb (drops "Define Schema", "Schedule Interviews", "Refine")
            if ent.root.pos_ == "VERB" or ent.root.tag_ in {"VB", "VBP", "VBD", "VBZ", "VBG"}:
                continue

            # Guard 2: Root must NOT be a scalar numeral / quantity
            if ent.root.like_num or ent.root.tag_ == "CD":
                continue

            c_text = ent.text.strip(" ,.:;\"'()[]{}<>`~*#\n\r\t")
            if len(c_text) < 2:
                continue

            # Decompose compound mentions against active entities (e.g. 'Pandas NumPy' -> Pandas, NumPy)
            c_lower = c_text.lower()
            decomposed = [a for a in active_set if a in c_lower and a != c_lower]
            if len(decomposed) >= 2:
                for d in decomposed:
                    candidates[d] = d
                continue

            if self.is_valid_entity(c_text):
                candidates[c_lower] = c_text

        # Capture proper nouns and capitalized named tokens from POS parse
        for token in doc:
            if token.pos_ in ("PROPN", "NOUN") and not token.is_stop and not token.like_num:
                if token.ent_type_ in ("DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"):
                    continue
                t_clean = token.text.strip(" ,.:;\"'()[]{}<>`~*#\n\r\t")
                if len(t_clean) >= 2 and self.is_valid_entity(t_clean):
                    if token.pos_ == "PROPN" or any(c.isupper() for c in t_clean[1:]):
                        candidates[t_clean.lower()] = t_clean

        # -------------------------------------------------------------
        # 3. Subsumption Pruning & Deduplication
        # -------------------------------------------------------------
        final_entities: List[str] = []
        cand_keys = sorted(list(candidates.keys()), key=lambda k: len(k), reverse=True)
        for k in cand_keys:
            cand = candidates[k]
            subsumed = False
            for ex in final_entities:
                if k != ex.lower() and k in ex.lower():
                    subsumed = True
                    break
            if not subsumed:
                final_entities.append(cand)

        return final_entities[:self.max_entities_per_turn]

    def extract_what_syntactic(self, entity_name: str, doc) -> Optional[str]:
        """
        Syntactic Dependency Parsing for nominal predicates and head modifiers.
        Matches copular structures (X is a Y), appositions (X, a Y), and head modifiers (X Y).
        Zero regex matchers.
        """
        def _is_valid_what(phrase: str) -> bool:
            p_clean = phrase.strip()
            if not p_clean or len(p_clean) < 2:
                return False
            # Ensure phrase has a substantive common noun head and is not all stopwords or numbers
            p_doc = get_spacy_nlp()(p_clean)
            if all(t.is_stop or t.like_num or t.is_punct for t in p_doc):
                return False
            return any(t.pos_ in ("NOUN", "PROPN") and not t.is_stop and not t.like_num for t in p_doc)

        ent_lower = entity_name.lower().strip()
        for sent in doc.sents:
            if ent_lower not in sent.text.lower():
                continue

            for token in sent:
                if token.text.lower() in ent_lower and token.pos_ in {"PROPN", "NOUN"}:
                    # 1. Copular predicate: X is [attr]
                    if token.dep_ == "nsubj" and token.head.lemma_ == "be":
                        for child in token.head.children:
                            if child.dep_ in {"attr", "acomp"} and child.pos_ in {"NOUN", "PROPN"}:
                                mods = [t.text for t in child.lefts if t.pos_ in {"ADJ", "NOUN", "PROPN"} and not t.is_stop]
                                phrase = " ".join(mods + [child.text]).strip()
                                if _is_valid_what(phrase):
                                    return phrase

                    # 2. Apposition: X, a [appos]
                    for child in token.children:
                        if child.dep_ == "appos" and child.pos_ in {"NOUN", "PROPN"}:
                            mods = [t.text for t in child.lefts if t.pos_ in {"ADJ", "NOUN", "PROPN"} and not t.is_stop]
                            phrase = " ".join(mods + [child.text]).strip()
                            if _is_valid_what(phrase):
                                return phrase

                    # 3. Direct head modification: e.g. "PostgreSQL database" -> "database"
                    if token.dep_ in {"nmod", "compound", "amod"} and token.head.pos_ in {"NOUN", "PROPN"}:
                        h_text = token.head.text.strip()
                        if h_text.lower() not in ent_lower and _is_valid_what(h_text):
                            return h_text

        return None

    def extract_what(
        self,
        entity_name: str,
        sample_context: str,
        doc=None,
        embedder: Any = None,
    ) -> Tuple[str, float]:
        """
        Dynamic 'What' Category Resolver:
        - Extracts grammatical nominal predicates via Syntactic Dependency Parsing (copula, apposition, head modification).
        - If embedder is provided, projects against contextual common noun candidates.
        """
        clean_ctx = self.clean_text_for_parsing(sample_context) if sample_context else ""
        if doc is None and clean_ctx:
            nlp = get_spacy_nlp()
            doc = nlp(clean_ctx)

        # 1. Syntactic Candidate Extraction (Copula, Apposition, Compound)
        if doc is not None:
            cand = self.extract_what_syntactic(entity_name, doc)
            if cand:
                return cand, 0.85

        # 2. Contextual Common Noun Projection via Embedder
        emb = embedder or self.embedder
        if doc is not None and emb is not None:
            noun_candidates = []
            for token in doc:
                if token.pos_ == "NOUN" and not token.is_stop and not token.like_num:
                    t_text = token.text.strip()
                    if t_text.lower() != entity_name.lower() and t_text not in noun_candidates:
                        noun_candidates.append(t_text)

            if noun_candidates:
                ent_vec = np.asarray(emb.embed_text(entity_name), dtype=np.float32).ravel()
                norm_e = np.linalg.norm(ent_vec)
                if norm_e > 1e-9:
                    ent_vec = ent_vec / norm_e
                    cand_vecs = np.asarray(emb.embed_text(noun_candidates), dtype=np.float32)
                    if cand_vecs.ndim == 1:
                        cand_vecs = cand_vecs.reshape(1, -1)
                    norms_c = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
                    norms_c[norms_c < 1e-9] = 1.0
                    cand_vecs = cand_vecs / norms_c
                    sims = np.dot(cand_vecs, ent_vec)
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    if best_sim > 0.15:
                        return noun_candidates[best_idx], round(best_sim, 4)

        return "concept", 0.50

    def detect_mutations(self, text: str) -> List[Dict[str, Any]]:
        """
        Dependency-driven detection of structural migrations and substitutions.
        Zero regex matchers: matches prepositional transition structures (from X to Y, X with Y).
        """
        if not text or not text.strip():
            return []

        clean_text = self.clean_text_for_parsing(text)
        doc = get_spacy_nlp()(clean_text)
        mutations: List[Dict[str, Any]] = []

        for tok in doc:
            if tok.pos_ in ("VERB", "NOUN"):
                from_prep = next((c for c in tok.children if c.dep_ == "prep" and c.lemma_ == "from"), None)
                to_prep = next((c for c in tok.children if c.dep_ == "prep" and c.lemma_ == "to"), None)
                if from_prep and not to_prep:
                    to_prep = next((c for c in from_prep.children if c.dep_ == "prep" and c.lemma_ == "to"), None)
                with_prep = next((c for c in tok.children if c.dep_ == "prep" and c.lemma_ == "with"), None)
                dobj = next((c for c in tok.children if c.dep_ in ("dobj", "obj")), None)
                neg = next((c for c in tok.children if c.dep_ == "neg"), None)

                # Transition: "from X to Y" (switched, migrated, moved, changed, upgraded, ...)
                if from_prep and to_prep:
                    obs = " ".join([c.text for c in from_prep.children if c.dep_ == "pobj"]).strip()
                    rep = " ".join([c.text for c in to_prep.children if c.dep_ == "pobj"]).strip()
                    if obs and rep:
                        mutations.append({
                            "obsolete": obs,
                            "replacement": rep,
                            "reason": f"Syntactic transition ({tok.lemma_})",
                        })

                # Substitution: "replace/substitute/swap X with Y"
                elif dobj and with_prep:
                    obs = dobj.text.strip()
                    rep = " ".join([c.text for c in with_prep.children if c.dep_ == "pobj"]).strip()
                    if obs and rep:
                        mutations.append({
                            "obsolete": obs,
                            "replacement": rep,
                            "reason": f"Syntactic substitution ({tok.lemma_})",
                        })

                # Deprecation: negated verb governing entity
                elif neg and dobj:
                    obs = dobj.text.strip()
                    if obs:
                        mutations.append({
                            "obsolete": obs,
                            "replacement": None,
                            "reason": f"Syntactic deprecation ({tok.lemma_})",
                        })

        return mutations

    def extract_slotted_attributes(self, text: str) -> List[Dict[str, str]]:
        """
        Extracts key-value attributes from grammatical frames and dependency trees.
        Zero regex matchers.
        """
        if not text or not text.strip():
            return []

        doc = get_spacy_nlp()(text)
        slots: List[Dict[str, str]] = []

        # 1. Copular frames: X is [attr]
        for sent in doc.sents:
            for tok in sent:
                if tok.pos_ in ("PROPN", "NOUN") and tok.dep_ in ("nsubj", "nsubjpass"):
                    head = tok.head
                    if head.lemma_ == "be":
                        attr = next((c for c in head.children if c.dep_ in ("attr", "acomp")), None)
                        if attr:
                            val_text = " ".join([t.text for t in attr.subtree if not t.is_punct]).strip()
                            if val_text and not attr.is_stop and self.is_valid_entity(tok.text):
                                slots.append({
                                    "entity": tok.text,
                                    "aspect": "state",
                                    "value": val_text,
                                    "source": "copula",
                                })

        # 2. Date entities (Deadlines / Milestones)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                head_sent = ent.root.sent
                subj = next((t for t in head_sent if t.pos_ in ("PROPN", "NOUN") and t.dep_ in ("nsubj", "dobj", "pobj") and t.text != ent.text and self.is_valid_entity(t.text)), None)
                if subj:
                    slots.append({
                        "entity": subj.text,
                        "aspect": "scheduled_date",
                        "value": ent.text,
                        "source": "date_dependency",
                    })

        # 3. Action and transitive verb status frames
        for tok in doc:
            if tok.pos_ == "VERB" and tok.dep_ not in ("aux", "auxpass"):
                obj = next((c for c in tok.children if c.dep_ in ("dobj", "obj")), None)
                if obj and self.is_valid_entity(obj.text):
                    neg = any(c.dep_ == "neg" for c in tok.children)
                    prep_instr = next((c for c in tok.children if c.dep_ == "prep" and c.lemma_ in ("using", "with", "via")), None)
                    using_obj = " ".join([c.text for c in prep_instr.children if c.dep_ == "pobj"]) if prep_instr else ""
                    status = f"not {tok.lemma_}" if neg else tok.lemma_
                    if using_obj:
                        status += f" using {using_obj}"
                    aspect = "implementation_status" if tok.lemma_ == "implement" else f"{tok.lemma_}_status"
                    slots.append({
                        "entity": obj.text,
                        "aspect": aspect,
                        "value": status,
                        "source": "verb_frame",
                    })

        return slots

    def compute_turn_salience(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        has_mutation: bool = False,
        is_action_or_pref: Optional[bool] = None,
    ) -> float:
        """
        Computes turn salience score S in [0.0, 1.0].
        """
        ents = entities if entities is not None else self.extract_entities(text)
        num_ents = len(ents)

        is_action = is_action_or_pref
        if is_action is None:
            doc = get_spacy_nlp()(text[:200])
            is_action = any(t.pos_ == "AUX" and t.tag_ == "MD" for t in doc)

        length_factor = min(1.0, len(text.strip()) / 120.0)

        salience = (
            0.20 * min(num_ents, 4)
            + (0.35 if has_mutation else 0.0)
            + (0.25 if is_action else 0.0)
            + 0.10 * length_factor
        )

        return float(min(1.0, max(0.0, salience)))
