# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Pure-Python Bloom filter using double hashing (Kirsch-Mitzenmacher trick).

The Kirsch-Mitzenmacher construction uses two independent hash functions
(here: MD5 and SHA-1) to simulate k independent hash functions with:

    h_i(x) = (h1(x) + i * h2(x)) mod m

This avoids computing k separate hashes, keeping ``add`` and
``might_contain`` O(k) in hash operations but O(1) in hash-function calls.

Bit storage uses a compact ``bytearray`` (1 byte per 8 bits) for memory
efficiency.

Persistence is a simple binary format:
    [8 bytes: capacity (uint64 LE)]
    [8 bytes: fp_rate as IEEE-754 double LE]
    [8 bytes: size in bits (uint64 LE)]
    [8 bytes: hash_count (uint64 LE)]
    [N bytes: bit array (bytearray)]
"""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path

# ---------------------------------------------------------------------------
# Header layout for serialisation
# ---------------------------------------------------------------------------

# Format: capacity, fp_rate (double), size_bits, hash_count
_HEADER_FORMAT = "<QdQQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)  # 32 bytes


def _optimal_size(capacity: int, fp_rate: float) -> int:
    """
    Return the optimal bit-array size m for the given *capacity* and
    false-positive rate *fp_rate*.

    Formula:  m = -n * ln(p) / (ln 2)^2
    """
    return max(1, math.ceil(-capacity * math.log(fp_rate) / (math.log(2) ** 2)))


def _optimal_hash_count(size_bits: int, capacity: int) -> int:
    """
    Return the optimal number of hash functions k.

    Formula:  k = (m / n) * ln 2
    """
    return max(1, round((size_bits / capacity) * math.log(2)))


def _double_hash(item: str) -> tuple[int, int]:
    """
    Compute the two independent hash values for *item*.

    Returns:
        ``(h1, h2)`` as unsigned 64-bit integers derived from MD5 and SHA-1
        digests respectively.
    """
    encoded = item.encode("utf-8")

    md5_digest = hashlib.md5(encoded, usedforsecurity=False).digest()
    sha1_digest = hashlib.sha1(encoded, usedforsecurity=False).digest()

    # Interpret the first 8 bytes of each digest as a little-endian uint64.
    h1 = struct.unpack_from("<Q", md5_digest, 0)[0]
    h2 = struct.unpack_from("<Q", sha1_digest, 0)[0]

    return h1, h2


class BloomFilter:
    """
    A space-efficient probabilistic set membership structure.

    After adding *n* items the probability of a false positive is
    approximately *fp_rate* when *n <= capacity*.  False negatives are
    impossible — if ``might_contain`` returns False the item was definitely
    never added.

    Args:
        capacity:  Expected maximum number of distinct items to insert.
                   Setting this too low increases the actual false-positive
                   rate beyond *fp_rate*.
        fp_rate:   Target false-positive probability (0 < fp_rate < 1).
                   Smaller values require more memory.

    Example::

        bf = BloomFilter(capacity=50_000, fp_rate=0.01)
        bf.add("hello.py")
        assert bf.might_contain("hello.py")    # True
        assert not bf.might_contain("other")   # probably False
    """

    def __init__(
        self, capacity: int = 100_000, fp_rate: float = 0.001
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if not (0 < fp_rate < 1):
            raise ValueError(f"fp_rate must be in (0, 1), got {fp_rate}")

        self.capacity = capacity
        self.fp_rate = fp_rate
        self.size_bits: int = _optimal_size(capacity, fp_rate)
        self.hash_count: int = _optimal_hash_count(self.size_bits, capacity)

        # Allocate bit array — ceil(size_bits / 8) bytes, all zeros.
        byte_count = (self.size_bits + 7) // 8
        self._bits = bytearray(byte_count)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, item: str) -> None:
        """
        Add *item* to the filter.

        Args:
            item: Any string to insert.  The empty string is valid.
        """
        h1, h2 = _double_hash(item)
        m = self.size_bits
        for i in range(self.hash_count):
            bit_index = (h1 + i * h2) % m
            byte_pos, bit_offset = divmod(bit_index, 8)
            self._bits[byte_pos] |= 1 << bit_offset

    def might_contain(self, item: str) -> bool:
        """
        Test membership of *item*.

        Returns:
            False  — *item* was **definitely** never added.
            True   — *item* was **probably** added (false positives are possible
                     at rate ≈ *fp_rate* once the filter holds *capacity* items).
        """
        h1, h2 = _double_hash(item)
        m = self.size_bits
        for i in range(self.hash_count):
            bit_index = (h1 + i * h2) % m
            byte_pos, bit_offset = divmod(bit_index, 8)
            if not (self._bits[byte_pos] & (1 << bit_offset)):
                return False
        return True

    # ------------------------------------------------------------------
    # Convenience alias
    # ------------------------------------------------------------------

    def __contains__(self, item: str) -> bool:
        """Support ``item in bloom_filter`` syntax."""
        return self.might_contain(item)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Serialise the filter to *path* in a compact binary format.

        The output file is self-describing — :meth:`load` does not need
        the original constructor parameters.

        Args:
            path: Destination file path (created or overwritten).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        header = struct.pack(
            _HEADER_FORMAT,
            self.capacity,
            self.fp_rate,
            self.size_bits,
            self.hash_count,
        )
        with open(path, "wb") as fh:
            fh.write(header)
            fh.write(self._bits)

    @classmethod
    def load(cls, path: str | Path) -> "BloomFilter":
        """
        Deserialise a filter previously saved with :meth:`save`.

        Args:
            path: Path to the saved filter file.

        Returns:
            A :class:`BloomFilter` instance with the same state as when it
            was saved.

        Raises:
            OSError:       If the file cannot be read.
            struct.error:  If the file header is malformed / truncated.
            ValueError:    If the stored parameters are logically invalid.
        """
        path = Path(path)
        with open(path, "rb") as fh:
            header_bytes = fh.read(_HEADER_SIZE)
            if len(header_bytes) < _HEADER_SIZE:
                raise struct.error(
                    f"Bloom filter file too short: expected {_HEADER_SIZE} header bytes"
                )
            capacity, fp_rate, size_bits, hash_count = struct.unpack(
                _HEADER_FORMAT, header_bytes
            )
            bits_data = fh.read()

        # Validate.
        expected_bytes = (size_bits + 7) // 8
        if len(bits_data) != expected_bytes:
            raise ValueError(
                f"Bloom filter bit array size mismatch: "
                f"expected {expected_bytes} bytes, got {len(bits_data)}"
            )

        # Reconstruct without re-allocating (bypass __init__ calculation).
        bf = cls.__new__(cls)
        bf.capacity = capacity
        bf.fp_rate = fp_rate
        bf.size_bits = size_bits
        bf.hash_count = hash_count
        bf._bits = bytearray(bits_data)
        return bf

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def memory_bytes(self) -> int:
        """Byte size of the internal bit array."""
        return len(self._bits)

    @property
    def fill_ratio(self) -> float:
        """
        Fraction of bits that are set.

        A fill ratio near 1.0 means the filter is over-capacity and the
        actual false-positive rate is much higher than *fp_rate*.
        """
        set_bits = sum(bin(b).count("1") for b in self._bits)
        return set_bits / self.size_bits if self.size_bits else 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BloomFilter(capacity={self.capacity}, fp_rate={self.fp_rate}, "
            f"size_bits={self.size_bits}, hash_count={self.hash_count}, "
            f"memory_kb={self.memory_bytes / 1024:.1f})"
        )
