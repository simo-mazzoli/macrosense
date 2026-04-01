#!/usr/bin/env python3
"""
Algoritmo per stimare se due parole italiane appartengono allo stesso macro-concetto.

Output principale:
- label proposta del macro-senso condiviso
- probabilità [0, 1] di appartenenza allo stesso macro-concetto
- dettagli dei segnali usati e loro contributi

Segnali supportati:
1) WordNet / OMW (sensi, iperonimi comuni, similarità di percorso)
2) Embeddings (spaCy it_core_news_lg o fastText locale)
3) Overlap lessicale delle gloss

Nota modello consigliato (GitHub):
- fastText (facebookresearch/fastText) con vettori italiani `cc.it.300.bin`
  https://github.com/facebookresearch/fastText
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

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


@dataclasses.dataclass
class InfluenceWeights:
    lexical_wordnet: float = 0.45
    embeddings: float = 0.35
    gloss_overlap: float = 0.20

    def normalized(self) -> "InfluenceWeights":
        total = self.lexical_wordnet + self.embeddings + self.gloss_overlap
        if total <= 0:
            return InfluenceWeights(1 / 3, 1 / 3, 1 / 3)
        return InfluenceWeights(
            lexical_wordnet=self.lexical_wordnet / total,
            embeddings=self.embeddings / total,
            gloss_overlap=self.gloss_overlap / total,
        )


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if t.lower() not in STOPWORDS_IT]


class EmbeddingBackend:
    def __init__(self, fasttext_path: Optional[str] = None) -> None:
        self.available = False
        self.source = "none"
        self._nlp = None
        self._ft_model = None

        if fasttext_path:
            try:
                from gensim.models import KeyedVectors

                self._ft_model = KeyedVectors.load_word2vec_format(fasttext_path, binary=fasttext_path.endswith(".bin"))
                self.available = True
                self.source = "fasttext"
                return
            except Exception:
                self._ft_model = None

        try:
            import spacy

            self._nlp = spacy.load("it_core_news_lg")
            self.available = True
            self.source = "spacy"
        except Exception:
            self._nlp = None

    def similarity(self, a: str, b: str) -> float:
        if not self.available:
            return 0.0

        if self.source == "fasttext" and self._ft_model is not None:
            if a in self._ft_model and b in self._ft_model:
                sim = float(self._ft_model.similarity(a, b))
                return max(0.0, min(1.0, (sim + 1.0) / 2.0))
            return 0.0

        if self.source == "spacy" and self._nlp is not None:
            va = self._nlp(a)
            vb = self._nlp(b)
            if va.vector_norm == 0 or vb.vector_norm == 0:
                return 0.0
            sim = float(va.similarity(vb))
            return max(0.0, min(1.0, (sim + 1.0) / 2.0))

        return 0.0


class MacroSenseMatcher:
    def __init__(self, weights: Optional[InfluenceWeights] = None, fasttext_path: Optional[str] = None) -> None:
        self.weights = (weights or InfluenceWeights()).normalized()
        self.embedding_backend = EmbeddingBackend(fasttext_path=fasttext_path)
        self.wordnet = self._load_wordnet()

    @staticmethod
    def _load_wordnet():
        try:
            from nltk.corpus import wordnet as wn

            _ = wn.synsets("cane", lang="ita")
            return wn
        except Exception as exc:
            raise RuntimeError(
                "WordNet non disponibile. Installa NLTK e scarica i corpora: "
                "`python -m nltk.downloader wordnet omw-1.4`"
            ) from exc

    def _lemmas_it(self, synset) -> List[str]:
        lemmas = sorted({x.replace("_", " ") for x in synset.lemma_names("ita")})
        if not lemmas:
            lemmas = sorted({x.replace("_", " ") for x in synset.lemma_names()})
        return lemmas

    def _gloss_tokens(self, synset) -> set[str]:
        text = " ".join([synset.definition(), *synset.examples()])
        return set(_tokenize(text))

    def _safe_path_similarity(self, s1, s2) -> float:
        sim = s1.path_similarity(s2)
        if sim is None:
            return 0.0
        return max(0.0, min(1.0, float(sim)))

    def _shared_hypernym_score_and_label(self, s1, s2) -> Tuple[float, Optional[str]]:
        hypers = s1.lowest_common_hypernyms(s2)
        if not hypers:
            return 0.0, None

        depths = [h.max_depth() for h in hypers]
        max_depth = max(depths) if depths else 0
        depth_score = 1.0 - math.exp(-max_depth / 8.0)

        best = hypers[depths.index(max_depth)]
        label_candidates = self._lemmas_it(best)
        label = label_candidates[0] if label_candidates else best.name().split(".")[0]
        return max(0.0, min(1.0, depth_score)), label

    def _lexical_component(self, s1, s2) -> Tuple[float, Optional[str], Dict[str, float]]:
        path_sim = self._safe_path_similarity(s1, s2)
        hypernym_score, label = self._shared_hypernym_score_and_label(s1, s2)
        same_synset = 1.0 if s1.name() == s2.name() else 0.0

        lexical = 0.55 * path_sim + 0.35 * hypernym_score + 0.10 * same_synset
        details = {
            "path_similarity": round(path_sim, 4),
            "shared_hypernym_depth": round(hypernym_score, 4),
            "same_synset": round(same_synset, 4),
        }
        return max(0.0, min(1.0, lexical)), label, details

    def _gloss_overlap_component(self, s1, s2) -> float:
        t1 = self._gloss_tokens(s1)
        t2 = self._gloss_tokens(s2)
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)

    def compare_words(self, word1: str, word2: str) -> Dict:
        synsets1 = self.wordnet.synsets(word1, lang="ita")
        synsets2 = self.wordnet.synsets(word2, lang="ita")

        if not synsets1 or not synsets2:
            return {
                "word1": word1,
                "word2": word2,
                "weights": dataclasses.asdict(self.weights),
                "embeddings": {
                    "available": self.embedding_backend.available,
                    "source": self.embedding_backend.source,
                },
                "error": "Almeno una parola non ha sensi WordNet in italiano.",
                "missing": {
                    "word1_has_synsets": bool(synsets1),
                    "word2_has_synsets": bool(synsets2),
                },
            }

        word_embedding_sim = self.embedding_backend.similarity(word1, word2)

        best = None
        for s1 in synsets1:
            for s2 in synsets2:
                lexical, candidate_label, lexical_details = self._lexical_component(s1, s2)
                gloss = self._gloss_overlap_component(s1, s2)
                emb = word_embedding_sim

                score = (
                    self.weights.lexical_wordnet * lexical
                    + self.weights.embeddings * emb
                    + self.weights.gloss_overlap * gloss
                )

                label = candidate_label or self._lemmas_it(s1)[0] if self._lemmas_it(s1) else s1.name().split(".")[0]

                candidate = {
                    "macro_label": label,
                    "probability": max(0.0, min(1.0, score)),
                    "sense_pair": {
                        "word1_synset": s1.name(),
                        "word2_synset": s2.name(),
                        "word1_definition": s1.definition(),
                        "word2_definition": s2.definition(),
                    },
                    "components": {
                        "lexical_wordnet": round(lexical, 4),
                        "embeddings": round(emb, 4),
                        "gloss_overlap": round(gloss, 4),
                        "lexical_details": lexical_details,
                    },
                }

                if best is None or candidate["probability"] > best["probability"]:
                    best = candidate

        assert best is not None
        best["probability"] = round(best["probability"], 4)

        return {
            "word1": word1,
            "word2": word2,
            "weights": dataclasses.asdict(self.weights),
            "embeddings": {
                "available": self.embedding_backend.available,
                "source": self.embedding_backend.source,
                "github_recommendation": "facebookresearch/fastText + cc.it.300.bin",
            },
            "result": best,
            "meta": {
                "synsets_word1": len(synsets1),
                "synsets_word2": len(synsets2),
            },
        }


def _interactive_weights(defaults: InfluenceWeights) -> InfluenceWeights:
    print("Inserisci i pesi di influenza (Invio = default).")
    print(f"- lexical_wordnet [default {defaults.lexical_wordnet}]: ", end="")
    lw = input().strip()
    print(f"- embeddings [default {defaults.embeddings}]: ", end="")
    ew = input().strip()
    print(f"- gloss_overlap [default {defaults.gloss_overlap}]: ", end="")
    gw = input().strip()

    return InfluenceWeights(
        lexical_wordnet=float(lw) if lw else defaults.lexical_wordnet,
        embeddings=float(ew) if ew else defaults.embeddings,
        gloss_overlap=float(gw) if gw else defaults.gloss_overlap,
    ).normalized()


def main() -> None:
    parser = argparse.ArgumentParser(description="Confronto macro-senso tra due parole italiane")
    parser.add_argument("word1", help="Prima parola italiana")
    parser.add_argument("word2", help="Seconda parola italiana")
    parser.add_argument("--lexical-weight", type=float, default=0.45)
    parser.add_argument("--embedding-weight", type=float, default=0.35)
    parser.add_argument("--gloss-weight", type=float, default=0.20)
    parser.add_argument("--interactive-weights", action="store_true", help="Chiede i pesi all'utente via prompt")
    parser.add_argument("--fasttext-path", help="Path locale a modello fastText (.bin/.vec) opzionale")
    parser.add_argument("--pretty", action="store_true", help="Output JSON formattato")
    args = parser.parse_args()

    weights = InfluenceWeights(
        lexical_wordnet=args.lexical_weight,
        embeddings=args.embedding_weight,
        gloss_overlap=args.gloss_weight,
    )
    if args.interactive_weights:
        weights = _interactive_weights(weights)

    matcher = MacroSenseMatcher(weights=weights, fasttext_path=args.fasttext_path)
    result = matcher.compare_words(args.word1, args.word2)

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
