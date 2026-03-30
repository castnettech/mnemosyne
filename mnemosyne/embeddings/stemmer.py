# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Porter stemmer algorithm: public domain (Martin Porter, 1980).
# This is a minimal, dependency-free implementation of the standard
# Porter stemming rules sufficient for TF-IDF token normalisation.

"""
Minimal Porter stemmer for Mnemosyne token normalisation.

Provides a single public function :func:`stem` that reduces an English
word to its approximate root form.  Used by the TF-IDF backend to align
tokenisation with BM25/FTS5 ``porter`` stemming.
"""

from __future__ import annotations

import re


def _measure(stem: str) -> int:
    """Count the number of consonant-vowel sequences (m) in *stem*."""
    # Reduce to C/V pattern then count VC pairs
    cv = re.sub(r"[aeiou]+", "V", stem.lower())
    cv = re.sub(r"[^V]+", "C", cv)
    return cv.count("CV")


def _has_vowel(stem: str) -> bool:
    """Return True if *stem* contains at least one vowel."""
    return bool(re.search(r"[aeiou]", stem.lower()))


def _ends_double_consonant(word: str) -> bool:
    """Return True if *word* ends with a double consonant."""
    if len(word) < 2:
        return False
    return (
        word[-1] == word[-2]
        and word[-1].lower() not in "aeiou"
    )


def _ends_cvc(word: str) -> bool:
    """Return True if *word* ends consonant-vowel-consonant (not w/x/y)."""
    if len(word) < 3:
        return False
    w = word.lower()
    return (
        w[-1] not in "aeiouwxy"
        and w[-2] in "aeiou"
        and w[-3] not in "aeiou"
    )


def stem(word: str) -> str:
    """Apply Porter stemming rules to a single word. Returns the stemmed form."""
    if len(word) <= 2:
        return word

    w = word.lower()

    # Step 1a — plurals
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]

    # Step 1b — -eed, -ed, -ing
    if w.endswith("eed"):
        base = w[:-3]
        if _measure(base) > 0:
            w = w[:-1]
    elif w.endswith("ed"):
        base = w[:-2]
        if _has_vowel(base):
            w = base
            w = _step1b_fixup(w)
    elif w.endswith("ing"):
        base = w[:-3]
        if _has_vowel(base):
            w = base
            w = _step1b_fixup(w)

    # Step 1c — y -> i
    if w.endswith("y") and _has_vowel(w[:-1]) and len(w) > 2:
        w = w[:-1] + "i"

    # Step 2 — double suffixes
    w = _step2(w)

    # Step 3
    w = _step3(w)

    # Step 4
    w = _step4(w)

    # Step 5a — remove trailing e
    if w.endswith("e"):
        base = w[:-1]
        m = _measure(base)
        if m > 1:
            w = base
        elif m == 1 and not _ends_cvc(base):
            w = base

    # Step 5b — double consonant with m > 1
    if _ends_double_consonant(w) and w[-1] == "l" and _measure(w[:-1]) > 1:
        w = w[:-1]

    return w


def _step1b_fixup(w: str) -> str:
    """Fixup after -ed/-ing removal: handle -at, -bl, -iz and double letters."""
    if w.endswith("at") or w.endswith("bl") or w.endswith("iz"):
        return w + "e"
    if _ends_double_consonant(w) and w[-1] not in "lsz":
        return w[:-1]
    if _measure(w) == 1 and _ends_cvc(w):
        return w + "e"
    return w


_STEP2_MAP: list[tuple[str, str]] = [
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("abli", "able"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
]


def _step2(w: str) -> str:
    """Step 2: map double suffixes to single. Requires m > 0."""
    for suffix, replacement in _STEP2_MAP:
        if w.endswith(suffix):
            base = w[: -len(suffix)]
            if _measure(base) > 0:
                return base + replacement
            return w
    return w


_STEP3_MAP: list[tuple[str, str]] = [
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
]


def _step3(w: str) -> str:
    """Step 3: longer suffix removal. Requires m > 0."""
    for suffix, replacement in _STEP3_MAP:
        if w.endswith(suffix):
            base = w[: -len(suffix)]
            if _measure(base) > 0:
                return base + replacement
            return w
    return w


_STEP4_SUFFIXES: list[str] = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
    "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
    "ous", "ive", "ize",
]


def _step4(w: str) -> str:
    """Step 4: final suffix stripping. Requires m > 1."""
    for suffix in _STEP4_SUFFIXES:
        if w.endswith(suffix):
            base = w[: -len(suffix)]
            if suffix == "ion" and base and base[-1] in "st":
                if _measure(base) > 1:
                    return base
            elif _measure(base) > 1:
                return base
            return w
    return w
