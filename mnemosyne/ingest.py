# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Ingestion orchestrator for Mnemosyne.

Scans the project directory, chunks modified files, deduplicates by content
hash, stores chunks, updates the FTS5 index, computes embeddings, and
maintains the Bloom filter and audit log.
"""

from __future__ import annotations

import ast
import fnmatch
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from mnemosyne.models import FileRecord, Chunk, estimate_tokens

if TYPE_CHECKING:
    from mnemosyne.store import Store
    from mnemosyne.bloom import BloomFilter
    from mnemosyne.audit import AuditLog


class Ingester:
    """
    File ingestion orchestrator.

    Walks the project root, applies ignore rules, checks the Bloom filter
    and hash for changes, chunks each file, deduplicates by content hash,
    stores everything, and records an audit entry.

    Args:
        project_root:  Absolute path to the project directory to index.
        config:        Mnemosyne :class:`~mnemosyne.config.Config` instance.
        store:         Persistent :class:`~mnemosyne.store.Store` instance.
        bloom:         :class:`~mnemosyne.bloom.BloomFilter` for fast
                       "already indexed" checks.
        tfidf_backend: TF-IDF backend with a ``fit(chunk_id, text)``
                       method for updating the embedding index.
        audit:         :class:`~mnemosyne.audit.AuditLog` for operation
                       provenance records.
    """

    def __init__(
        self,
        project_root: str,
        config,
        store: "Store",
        bloom: "BloomFilter",
        tfidf_backend,
        audit: "AuditLog",
        dense_backend=None,
        doc_store=None,
        doc_tfidf=None,
    ) -> None:
        self.root = os.path.abspath(project_root)
        self.config = config
        self.store = store
        self.bloom = bloom
        self.tfidf = tfidf_backend
        self.audit = audit
        self.dense = dense_backend
        self.doc_store = doc_store
        self.doc_tfidf = doc_tfidf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        paths: list[str] | None = None,
        full: bool = False,
        dry_run: bool = False,
        progress=None,
    ) -> dict:
        """
        Index files in the project.

        Args:
            paths:   Explicit list of file paths to process.  When ``None``
                     all files under ``project_root`` are scanned.
            full:    Force a full re-index of every file, ignoring hashes
                     and mtimes.
            dry_run: Scan and report what would be indexed without writing
                     any data.

        Returns:
            Stats dict with keys ``files_scanned``, ``files_indexed``,
            ``files_skipped``, ``files_failed``, ``chunks_added``,
            ``chunks_deduped``, ``elapsed_seconds``.
        """
        t_start = time.monotonic()

        stats: dict[str, int | float] = {
            "files_scanned": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "chunks_added": 0,
            "chunks_deduped": 0,
            "elapsed_seconds": 0.0,
        }

        # Resolve file list
        if paths:
            file_list = self._resolve_paths(paths)
        else:
            file_list = self._scan_files()

        stats["files_scanned"] = len(file_list)

        # On full re-index of the entire project, purge stale file records
        # (files that were previously indexed but are no longer in the scan
        # list -- e.g. because they were deleted or newly match an ignore
        # pattern).  This prevents ghost chunks from polluting retrieval.
        if full and not paths and not dry_run:
            scanned_rel_paths = {
                os.path.relpath(f, self.root).replace(os.sep, "/")
                for f in file_list
            }
            for rec in self.store.list_files(include_deleted=False):
                if rec.rel_path not in scanned_rel_paths:
                    self.store.delete_chunks_for_file(rec.file_id)
                    self.store.mark_deleted(rec.file_id)

        total = len(file_list)
        for i, abs_path in enumerate(file_list):
            rel_path = os.path.relpath(abs_path, self.root).replace(os.sep, "/")

            if progress is not None:
                progress(i + 1, total, rel_path, stats)

            try:
                if not self._needs_indexing(abs_path, rel_path, full):
                    stats["files_skipped"] += 1
                    continue

                if dry_run:
                    stats["files_indexed"] += 1
                    continue

                added, deduped = self._index_file(abs_path, rel_path)
                stats["files_indexed"] += 1
                stats["chunks_added"] += added
                stats["chunks_deduped"] += deduped

            except Exception as exc:
                stats["files_failed"] += 1
                try:
                    self.audit.log(
                        "ingest_error",
                        file=rel_path,
                        error=str(exc),
                    )
                except Exception:
                    pass

        # Rebuild TF-IDF vocabulary and re-embed all chunks now that we have
        # the full corpus.  This is cheap for typical project sizes and ensures
        # IDF values are meaningful (single-file embeddings produce empty
        # vectors because min_df filtering removes all terms).
        if not dry_run and stats["chunks_added"] > 0:
            self._rebuild_tfidf()

        elapsed = time.monotonic() - t_start
        stats["elapsed_seconds"] = round(elapsed, 3)

        # Audit summary
        if not dry_run:
            try:
                self.audit.log("ingest_complete", **{str(k): v for k, v in stats.items()})
            except Exception:
                pass

        return stats

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_paths(self, paths: list[str]) -> list[str]:
        """
        Resolve user-supplied paths to a deduplicated list of indexable files.

        For each path in *paths*:

        - Relative paths are joined to ``self.root`` before resolving.
        - Absolute paths are resolved directly.
        - Symlinks are resolved with ``os.path.realpath()``.
        - Paths that resolve outside the project root raise ``ValueError``.
        - Files are included only if they pass extension and size filters.
        - Directories are walked using ``_scan_dir()`` (same logic as full scan).
        - Non-existent paths are silently skipped.

        Args:
            paths: Raw paths from the caller (CLI arguments, API, etc.).

        Returns:
            Deduplicated list of absolute file paths.

        Raises:
            ValueError: If any path resolves outside the project root.
        """
        from mnemosyne.hasher import is_document

        supported_exts: set[str] = set(self.config.general.supported_extensions)
        max_code_bytes: int = self.config.general.max_file_size_kb * 1024
        max_doc_bytes: int = getattr(
            getattr(self.config, "extraction", None),
            "max_file_size_kb", 10240,
        ) * 1024

        seen: set[str] = set()
        results: list[str] = []

        for p in paths:
            if os.path.isabs(p):
                real = os.path.realpath(p)
            else:
                real = os.path.realpath(os.path.join(self.root, p))

            if real != self.root and not real.startswith(self.root + os.sep):
                raise ValueError(f"Path '{p}' resolves outside project root")

            if not os.path.exists(real):
                continue

            if os.path.isdir(real):
                for f in self._scan_dir(real):
                    if f not in seen:
                        seen.add(f)
                        results.append(f)
            elif os.path.isfile(real):
                rel_path = os.path.relpath(real, self.root).replace(os.sep, "/")
                if self._should_ignore(rel_path):
                    continue

                _, ext = os.path.splitext(real)
                if ext.lower() not in supported_exts:
                    continue

                try:
                    size = os.path.getsize(real)
                except OSError:
                    continue
                max_bytes = max_doc_bytes if is_document(real) else max_code_bytes
                if size > max_bytes:
                    continue

                if real not in seen:
                    seen.add(real)
                    results.append(real)

        return results

    # ------------------------------------------------------------------
    # File scanning
    # ------------------------------------------------------------------

    def _scan_files(self) -> list[str]:
        """
        Walk the project directory and return absolute paths of candidate files.

        Applies extension filter, size limit, and ignore patterns.
        """
        return self._scan_dir(self.root)

    def _scan_dir(self, root_dir: str) -> list[str]:
        """
        Walk *root_dir* and return absolute paths of candidate files.

        Applies extension filter, size limit, and ignore patterns.  All
        relative-path checks use ``self.root`` as the base so that ignore
        patterns behave consistently regardless of which sub-directory is
        being scanned.

        Args:
            root_dir: Absolute path of the directory to walk.

        Returns:
            List of absolute file paths that pass all filters.
        """
        from mnemosyne.hasher import is_document

        supported_exts: set[str] = set(self.config.general.supported_extensions)
        max_code_bytes: int = self.config.general.max_file_size_kb * 1024
        max_doc_bytes: int = getattr(
            getattr(self.config, "extraction", None),
            "max_file_size_kb", 10240,
        ) * 1024

        results: list[str] = []

        for dirpath, dirnames, filenames in os.walk(root_dir):
            rel_dir = os.path.relpath(dirpath, self.root).replace(os.sep, "/")

            # Prune ignored directories in-place to prevent descent
            dirnames[:] = [
                d for d in dirnames
                if not self._should_ignore(
                    (rel_dir + "/" + d).lstrip("./") if rel_dir != "." else d
                )
            ]

            for fname in filenames:
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, self.root).replace(os.sep, "/")

                if self._should_ignore(rel_path):
                    continue

                _, ext = os.path.splitext(fname)
                if ext.lower() not in supported_exts:
                    continue

                try:
                    size = os.path.getsize(abs_path)
                except OSError:
                    continue

                # Documents get a higher size limit than code files
                max_bytes = max_doc_bytes if is_document(abs_path) else max_code_bytes
                if size > max_bytes:
                    continue

                results.append(abs_path)

        return results

    def _should_ignore(self, rel_path: str) -> bool:
        """
        Return True if *rel_path* matches any configured ignore pattern.

        Patterns are matched against both the full relative path and the
        basename, using ``fnmatch.fnmatch`` glob semantics.
        """
        patterns: list[str] = self.config.general.ignore_patterns
        basename = os.path.basename(rel_path)

        for pattern in patterns:
            # Match against basename
            if fnmatch.fnmatch(basename, pattern):
                return True
            # Match against full relative path (e.g. "node_modules/..." matches "node_modules")
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            # Match any path component against the pattern (for directory names)
            parts = rel_path.replace("\\", "/").split("/")
            for part in parts:
                if fnmatch.fnmatch(part, pattern):
                    return True

        return False

    def _needs_indexing(self, abs_path: str, rel_path: str, full: bool) -> bool:
        """
        Determine whether *abs_path* should be (re-)indexed.

        A file needs indexing when:

        * ``full`` is True (always re-index), or
        * The file is not in the Bloom filter, or
        * The stored ``FileRecord`` is absent, or
        * The file mtime or content hash differs from the stored record.

        Args:
            abs_path: Absolute filesystem path.
            rel_path: Relative path from project root.
            full:     Force re-index flag.

        Returns:
            True if the file should be (re-)indexed.
        """
        if full:
            return True

        # Quick Bloom filter check -- if definitely not present, needs indexing
        if not self.bloom.might_contain(rel_path):
            return True

        # Look up stored record
        file_record = self.store.get_file_record_by_path(rel_path)
        if file_record is None:
            return True

        # Compare mtime
        try:
            mtime = os.path.getmtime(abs_path)
        except OSError:
            return False

        if mtime != file_record.last_modified:
            # mtime changed -- check hash to confirm content change
            try:
                from mnemosyne.hasher import file_hash
                current_hash = file_hash(abs_path)
            except Exception:
                return True
            return current_hash != file_record.content_hash

        return False

    # ------------------------------------------------------------------
    # Enrichment -- context prepended to embedding input only
    # ------------------------------------------------------------------

    def _build_enriched_text(
        self,
        cand,
        rel_path: str,
        module_doc: str,
    ) -> str:
        """Build enriched text for embedding.  Raw ``cand.content`` is stored
        in the DB unchanged (for display); only the embedding input changes.

        Enrichment layers:
        1. File path context
        2. Module docstring (first 100 chars, if available)
        3. Parent class name (if the chunk is a method)
        4. Symbol name and chunk type
        5. Original content
        """
        parts: list[str] = []

        # 1. File path context
        parts.append(f"# File: {rel_path}")

        # 2. Module docstring (trimmed)
        if module_doc:
            trimmed = module_doc[:100].replace("\n", " ").strip()
            parts.append(f"# Module: {trimmed}")

        # 3. Parent class context
        if cand.parent_symbol:
            parts.append(f"# Class: {cand.parent_symbol}")

        # 4. Symbol name context
        if cand.symbol_name:
            parts.append(f"# Symbol: {cand.symbol_name} ({cand.chunk_type})")

        # 5. Original content
        parts.append(cand.content)

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Core indexing logic
    # ------------------------------------------------------------------

    def _index_file(self, abs_path: str, rel_path: str) -> tuple[int, int]:
        """
        Read, chunk, dedup, store, and embed a single file.

        Binary/document files are routed through the extractor pipeline
        when an appropriate extractor is available.  Source code files
        continue through the existing chunker pipeline.

        Returns:
            ``(chunks_added, chunks_deduped)`` counts.
        """
        from mnemosyne.hasher import file_hash, is_binary, is_document
        from mnemosyne.chunkers import get_chunker, detect_language

        # Route document files through the extractor pipeline
        if is_document(abs_path) or is_binary(abs_path):
            return self._index_document(abs_path, rel_path)

        # Gate: check extraction config
        if not getattr(self.config, "extraction", None):
            pass  # no extraction config = code-only mode

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError:
            return 0, 0

        if not source.strip():
            return 0, 0

        mtime = os.path.getmtime(abs_path)
        size = os.path.getsize(abs_path)
        content_hash_val = file_hash(abs_path)
        language = detect_language(rel_path)

        # Upsert FileRecord
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        file_record = FileRecord(
            file_id=None,
            rel_path=rel_path,
            content_hash=content_hash_val,
            size_bytes=size,
            language=language,
            last_modified=mtime,
            last_indexed=now_iso,
            is_deleted=False,
        )
        file_id = self.store.upsert_file(file_record)

        # Delete old chunks for this file before re-chunking
        self.store.delete_chunks_for_file(file_id)

        # Extract module docstring once per file for enrichment
        module_doc = ""
        if language == "python":
            try:
                tree = ast.parse(source)
                module_doc = ast.get_docstring(tree) or ""
            except SyntaxError:
                pass

        # Chunk the file
        chunker = get_chunker(language, self.config)
        candidates = chunker.chunk(source, language)

        chunks_added = 0
        chunks_deduped = 0

        for cand in candidates:
            from mnemosyne.hasher import content_hash as compute_content_hash
            chunk_hash = compute_content_hash(cand.content)

            # Deduplicate by content hash across the whole index
            existing_chunk = self.store.get_chunk_by_hash(chunk_hash)
            if existing_chunk is not None:
                chunks_deduped += 1
                continue

            token_count = estimate_tokens(cand.content)
            chunk = Chunk(
                chunk_id=None,
                file_id=file_id,
                content_hash=chunk_hash,
                chunk_type=cand.chunk_type,
                line_start=cand.line_start,
                line_end=cand.line_end,
                token_count=token_count,
                content=cand.content,
                compressed=None,
                compression_ratio=None,
                symbol_name=cand.symbol_name,
                parent_chunk_id=None,
                page_number=getattr(cand, "page_number", None),
            )

            chunk_id = self.store.save_chunk(chunk)

            # Build enriched text for embedding (content stored raw for display)
            enriched = self._build_enriched_text(cand, rel_path, module_doc)

            # Update sparse embedding index (TF-IDF term weights for this chunk)
            try:
                terms = self.tfidf.embed(enriched)
                self.store.insert_sparse_embedding(chunk_id, terms)
            except Exception:
                pass

            # Dense embedding (optional -- requires onnxruntime + model)
            if self.dense is not None:
                try:
                    vec_bytes = self.dense.embed_to_bytes(enriched)
                    if vec_bytes:
                        self.store.insert_dense_embedding(chunk_id, vec_bytes, dim=self.dense.dim)
                except Exception:
                    pass

            # Update Bloom filter with the content hash
            self.bloom.add(chunk_hash)

            chunks_added += 1

        # Also add the file path to the Bloom filter
        self.bloom.add(rel_path)

        return chunks_added, chunks_deduped

    def _index_document(self, abs_path: str, rel_path: str) -> tuple[int, int]:
        """Extract text from a document file, chunk it, and index.

        Routes through the extractor pipeline and writes to the document
        partition (doc_store) when available.  If no doc_store is configured,
        skips the file -- documents do not enter the code partition.

        Returns:
            ``(chunks_added, chunks_deduped)`` counts.
        """
        if self.doc_store is None:
            return 0, 0

        from mnemosyne.extractors import extract_file
        from mnemosyne.hasher import file_hash_binary, content_hash as compute_content_hash
        from mnemosyne.chunkers.document_chunker import DocumentChunker

        extracted = extract_file(abs_path, self.config)
        if extracted is None or not extracted.pages:
            return 0, 0

        if extracted.extraction_quality == "failed" and not extracted.full_text.strip():
            return 0, 0

        mtime = os.path.getmtime(abs_path)
        size = os.path.getsize(abs_path)
        content_hash_val = file_hash_binary(abs_path)

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        file_record = FileRecord(
            file_id=None,
            rel_path=rel_path,
            content_hash=content_hash_val,
            size_bytes=size,
            language="document",
            last_modified=mtime,
            last_indexed=now_iso,
            is_deleted=False,
            source_type="document",
            extraction_method=extracted.extraction_method,
            extraction_quality=extracted.extraction_quality,
            page_count=extracted.page_count,
        )
        # FileRecord goes to the shared files table (via code store)
        file_id = self.store.upsert_file(file_record)

        # Chunks go to the document partition
        self.doc_store.delete_chunks_for_file(file_id)

        doc_chunker = DocumentChunker(self.config)
        candidates = doc_chunker.chunk_extracted(extracted)

        chunks_added = 0
        chunks_deduped = 0
        doc_tfidf = self.doc_tfidf or self.tfidf

        for cand in candidates:
            chunk_hash = compute_content_hash(cand.content)

            existing_chunk = self.doc_store.get_chunk_by_hash(chunk_hash)
            if existing_chunk is not None:
                chunks_deduped += 1
                continue

            token_count = estimate_tokens(cand.content)
            chunk = Chunk(
                chunk_id=None,
                file_id=file_id,
                content_hash=chunk_hash,
                chunk_type=cand.chunk_type,
                line_start=cand.line_start,
                line_end=cand.line_end,
                token_count=token_count,
                content=cand.content,
                compressed=None,
                compression_ratio=None,
                symbol_name=cand.symbol_name,
                parent_chunk_id=None,
                page_number=getattr(cand, "page_number", None),
            )

            chunk_id = self.doc_store.insert_chunk(chunk)

            enriched = self._build_enriched_text(cand, rel_path, "")

            try:
                terms = doc_tfidf.embed(enriched)
                self.doc_store.insert_sparse_embedding(chunk_id, terms)
            except Exception:
                pass

            self.bloom.add(chunk_hash)
            chunks_added += 1

        self.bloom.add(rel_path)
        return chunks_added, chunks_deduped

    def _rebuild_tfidf(self) -> None:
        """Rebuild TF-IDF vocabulary and re-embed all chunks.

        This must run after all files are ingested so IDF values reflect
        the full corpus rather than a single file at a time.  Enriched
        text (file path, module docstring, symbol context) is used for
        both vocabulary building and per-chunk embedding so that the
        TF-IDF index stays consistent with the initial ingest embeddings.
        """
        rows = self.store.conn.execute(
            "SELECT c.chunk_id, c.content, c.chunk_type, c.symbol_name, "
            "       f.rel_path "
            "FROM chunks c JOIN files f ON c.file_id = f.file_id"
        ).fetchall()
        if not rows:
            return

        # Build a minimal enriched text for each chunk.  We cannot
        # recover parent_symbol from the DB (it is not stored), so we
        # skip that layer.  Module docstring extraction is also skipped
        # here to avoid re-reading every source file; the path, symbol,
        # and type context still provide a significant boost.
        enriched_texts: list[str] = []
        for row in rows:
            parts: list[str] = [f"# File: {row[4]}"]
            if row[3]:  # symbol_name
                parts.append(f"# Symbol: {row[3]} ({row[2]})")
            parts.append(row[1])  # content
            enriched_texts.append("\n".join(parts))

        # Build vocabulary from the full corpus
        self.tfidf.build_vocabulary(enriched_texts)

        # Re-embed every chunk with the updated IDF values
        for row, enriched in zip(rows, enriched_texts):
            terms = self.tfidf.embed(enriched)
            if terms:
                self.store.insert_sparse_embedding(row[0], terms)

        # Persist vocabulary for future sessions
        try:
            self.tfidf._save_vocabulary()
        except Exception:
            pass

        # Rebuild document partition TF-IDF (isolated vocabulary)
        if self.doc_store is not None and self.doc_tfidf is not None:
            self._rebuild_doc_tfidf()

    def _rebuild_doc_tfidf(self) -> None:
        """Rebuild TF-IDF for the document partition with isolated IDF."""
        rows = self.doc_store.conn.execute(
            "SELECT c.chunk_id, c.content, c.chunk_type, c.symbol_name, "
            "       f.rel_path "
            "FROM doc_chunks c JOIN files f ON c.file_id = f.file_id"
        ).fetchall()
        if not rows:
            return

        enriched_texts: list[str] = []
        for row in rows:
            parts: list[str] = [f"# File: {row[4]}"]
            if row[3]:
                parts.append(f"# Section: {row[3]}")
            parts.append(row[1])
            enriched_texts.append("\n".join(parts))

        self.doc_tfidf.build_vocabulary(enriched_texts)

        for row, enriched in zip(rows, enriched_texts):
            terms = self.doc_tfidf.embed(enriched)
            if terms:
                self.doc_store.insert_sparse_embedding(row[0], terms)

        try:
            self.doc_tfidf._save_vocabulary()
        except Exception:
            pass
