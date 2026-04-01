#!/usr/bin/env python3
"""
Algoritmo di semantic decomposition per parole italiane.

Caratteristiche:
- Ricerca dei sensi (synset) in WordNet/Open Multilingual WordNet.
- Decomposizione per ogni senso, utile per parole polisemiche.
- Fusione ponderata di segnali semantici:
  1) relazioni lessicali WordNet (iperonimi/iponimi/meronimi/sinonimi)
  2) similarità distribuzionale tramite embeddings (se disponibili)
  3) overlap lessicale con gloss ed esempi

Uso:
    python semantic_decomposition.py "banca"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TOKEN_RE = re.compile(r"[a-zàèéìòù]+", re.IGNORECASE)
STOPWORDS_IT = {
    "il",
    "lo",
    "la",
    "i",
    "gli",
    "le",
    "un",
    "una",
    "uno",
    "di",
    "a",
    "da",
    "in",
    "con",
    "su",
    "per",
    "tra",
    "fra",
    "e",
    "o",
    "che",
    "del",
    "della",
    "dello",
    "dei",
    "degli",
    "delle",
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if t.lower() not in STOPWORDS_IT]


@dataclasses.dataclass
class DecompositionWeights:
    lexical_relations: float = 0.45
    embeddings: float = 0.35
    gloss_overlap: float = 0.20

    def normalized(self) -> "DecompositionWeights":
        s = self.lexical_relations + self.embeddings + self.gloss_overlap
        if s <= 0:
            return DecompositionWeights(1 / 3, 1 / 3, 1 / 3)
        return DecompositionWeights(
            lexical_relations=self.lexical_relations / s,
            embeddings=self.embeddings / s,
            gloss_overlap=self.gloss_overlap / s,
        )


class EmbeddingBackend:
    """Backend embeddings opzionale.

    Tenta di usare spaCy italiano con vettori (`it_core_news_lg`),
    altrimenti degrada senza interrompere il flusso.
    """

    def __init__(self) -> None:
        self._nlp = None
        self.available = False
        try:
            import spacy

            self._nlp = spacy.load("it_core_news_lg")
            self.available = True
        except Exception:
            self._nlp = None
            self.available = False

    def similarity(self, a: str, b: str) -> float:
        if not self.available or self._nlp is None:
            return 0.0
        va = self._nlp(a)
        vb = self._nlp(b)
        if va.vector_norm == 0 or vb.vector_norm == 0:
            return 0.0
        sim = float(va.similarity(vb))
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))


class ItalianSemanticDecomposer:
    def __init__(self, weights: Optional[DecompositionWeights] = None) -> None:
        self.weights = (weights or DecompositionWeights()).normalized()
        self.embedding_backend = EmbeddingBackend()
        self.wordnet = self._load_wordnet()

    @staticmethod
    def _load_wordnet():
        try:
            from nltk.corpus import wordnet as wn

            # Test rapido: verifica se il corpus è disponibile.
            _ = wn.synsets("cane", lang="ita")
            return wn
        except Exception as exc:
            raise RuntimeError(
                "WordNet non disponibile. Installa NLTK e scarica i corpora: "
                "`python -m nltk.downloader wordnet omw-1.4`"
            ) from exc

    def _italian_lemmas(self, synset) -> List[str]:
        lemmas = sorted({lemma.replace("_", " ") for lemma in synset.lemma_names("ita")})
        if not lemmas:
            # Fallback: lemma inglesi se manca mapping italiano
            lemmas = sorted({lemma.replace("_", " ") for lemma in synset.lemma_names()})
        return lemmas

    def _collect_related_terms(self, synset) -> Counter:
        cnt = Counter()

        def add_lemmas(ss, weight: float) -> None:
            for lm in self._italian_lemmas(ss):
                cnt[lm] += weight

        # Sinonimi nel synset corrente
        add_lemmas(synset, 1.0)

        # Relazioni semantiche tipiche
        for h in synset.hypernyms():
            add_lemmas(h, 0.9)
        for h in synset.hyponyms():
            add_lemmas(h, 0.7)
        for m in synset.part_meronyms() + synset.substance_meronyms() + synset.member_meronyms():
            add_lemmas(m, 0.65)
        for m in synset.part_holonyms() + synset.substance_holonyms() + synset.member_holonyms():
            add_lemmas(m, 0.5)

        if synset.instance_hypernyms():
            for ih in synset.instance_hypernyms():
                add_lemmas(ih, 0.75)

        return cnt

    def _gloss_context(self, synset) -> List[str]:
        pieces = [synset.definition()] + list(synset.examples())
        return _tokenize(" ".join(pieces))

    def _score_term(
        self,
        target_word: str,
        term: str,
        lexical_score: float,
        gloss_terms: Sequence[str],
    ) -> float:
        emb = self.embedding_backend.similarity(target_word, term)
        overlap = 1.0 if term.lower() in set(gloss_terms) else 0.0
        return (
            self.weights.lexical_relations * lexical_score
            + self.weights.embeddings * emb
            + self.weights.gloss_overlap * overlap
        )

    def decompose(self, word: str, top_k: int = 12) -> Dict:
        synsets = self.wordnet.synsets(word, lang="ita")
        if not synsets:
            return {
                "word": word,
                "senses": [],
                "note": "Nessun senso WordNet trovato per la parola italiana fornita.",
            }

        results = []
        for i, syn in enumerate(synsets, start=1):
            related = self._collect_related_terms(syn)
            gloss_terms = self._gloss_context(syn)

            # normalizzazione lessicale in [0,1]
            max_rel = max(related.values()) if related else 1.0
            scored_terms: List[Tuple[str, float, Dict[str, float]]] = []

            for term, raw_rel in related.items():
                if term.lower() == word.lower():
                    continue
                lex_norm = raw_rel / max_rel
                emb = self.embedding_backend.similarity(word, term)
                overlap = 1.0 if term.lower() in set(gloss_terms) else 0.0
                total = self._score_term(word, term, lex_norm, gloss_terms)
                scored_terms.append(
                    (
                        term,
                        total,
                        {
                            "lexical_relations": round(lex_norm, 4),
                            "embeddings": round(emb, 4),
                            "gloss_overlap": round(overlap, 4),
                        },
                    )
                )

            scored_terms.sort(key=lambda x: x[1], reverse=True)
            top_terms = [
                {
                    "term": t,
                    "score": round(s, 4),
                    "components": c,
                }
                for t, s, c in scored_terms[:top_k]
            ]

            results.append(
                {
                    "sense_index": i,
                    "synset": syn.name(),
                    "pos": syn.pos(),
                    "definition": syn.definition(),
                    "examples": list(syn.examples()),
                    "italian_synonyms": self._italian_lemmas(syn),
                    "semantic_decomposition": top_terms,
                }
            )

        return {
            "word": word,
            "weights": dataclasses.asdict(self.weights),
            "embeddings_available": self.embedding_backend.available,
            "senses": results,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic decomposition per parole italiane")
    parser.add_argument("word", help="Parola italiana da decomporre")
    parser.add_argument("--top-k", type=int, default=12, help="Numero massimo di termini per senso")
    parser.add_argument("--pretty", action="store_true", help="Output JSON formattato")
    args = parser.parse_args()

    decomposer = ItalianSemanticDecomposer()
    data = decomposer.decompose(args.word, top_k=max(1, args.top_k))

    if args.pretty:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    main()
