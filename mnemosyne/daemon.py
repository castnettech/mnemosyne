# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
JSON-RPC daemon for Mnemosyne.

Persistent Unix-domain-socket server keeping SQLite, TF-IDF index, analytics,
and prefetcher warm across requests.  Eliminates cold-start overhead.

Protocol: client connects to ``.mnemosyne/mnemosyne.sock``, sends
newline-terminated JSON ``{"method": "query", "params": {...}}\\n``,
receives ``{"result": ..., "error": null}\\n``, connection closes.
"""

from __future__ import annotations

import json
import logging
import os
import select
import signal
import socket
import traceback

logger = logging.getLogger("mnemosyne.daemon")
_MAX_REQUEST_BYTES = 65_536  # 64 KiB cap on request payload


class MnemosyneDaemon:
    """Persistent JSON-RPC server over a Unix domain socket."""

    def __init__(self, project_root: str) -> None:
        self.project_root = os.path.abspath(project_root)
        self._mnemosyne_dir = os.path.join(self.project_root, ".mnemosyne")
        self._socket_path = os.path.join(self._mnemosyne_dir, "mnemosyne.sock")
        self._pid_path = os.path.join(self._mnemosyne_dir, "daemon.pid")
        self._sock: socket.socket | None = None
        self._running = False
        self._init_components()

    def _init_components(self) -> None:
        """Warm-start all retrieval components."""
        from mnemosyne.config import Config
        from mnemosyne.schema import open_store
        from mnemosyne.store import Store
        from mnemosyne.embeddings import get_backend
        from mnemosyne.analytics import Analytics
        from mnemosyne.prefetch import Prefetcher
        from mnemosyne.retrieval import RetrievalEngine

        self.config = Config(root=self.project_root)
        self.conn = open_store(self._mnemosyne_dir)
        self.store = Store(self.conn)
        self.tfidf = get_backend(self.config, self.store)
        self.analytics = Analytics(self.store, self.config)
        self.prefetcher = Prefetcher(self.store)
        self.engine = RetrievalEngine(
            store=self.store, tfidf_backend=self.tfidf, config=self.config,
            analytics=self.analytics, prefetcher=self.prefetcher,
        )

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Bind socket, write PID file, enter accept loop."""
        os.makedirs(self._mnemosyne_dir, exist_ok=True)
        if os.path.exists(self._socket_path):
            try:
                os.unlink(self._socket_path)
            except OSError:
                pass

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(self._socket_path)
        # P1: Restrict socket to owner only — prevents unauthorized local access
        os.chmod(self._socket_path, 0o600)
        self._sock.listen(5)
        self._sock.setblocking(False)

        with open(self._pid_path, "w") as f:
            f.write(str(os.getpid()))

        # Signal handlers only work from main thread; tests use stop() directly.
        import threading
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

        self._running = True
        logger.info("Daemon started: pid=%d socket=%s", os.getpid(), self._socket_path)
        self._accept_loop()

    def stop(self) -> None:
        """Graceful shutdown: close socket, remove PID file, close DB."""
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        for path in (self._socket_path, self._pid_path):
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass
        if hasattr(self, "conn") and self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
        logger.info("Daemon stopped.")

    def _signal_handler(self, signum: int, frame) -> None:
        self._running = False

    # -- accept loop ------------------------------------------------------

    def _accept_loop(self) -> None:
        """Select-based accept loop, one request per connection."""
        try:
            while self._running:
                if self._sock is None:
                    break
                try:
                    readable, _, _ = select.select([self._sock], [], [], 0.5)
                except (OSError, ValueError):
                    break
                if not readable:
                    continue
                try:
                    sock = self._sock
                    if sock is None:
                        break
                    client, _ = sock.accept()
                except (OSError, AttributeError):
                    break
                try:
                    self._serve_client(client)
                except Exception:
                    logger.debug("Client error: %s", traceback.format_exc())
                finally:
                    try:
                        client.close()
                    except OSError:
                        pass
        finally:
            self.stop()

    def _serve_client(self, client: socket.socket) -> None:
        """Read one request, dispatch, send response."""
        client.settimeout(5.0)
        data = b""
        try:
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data or len(data) >= _MAX_REQUEST_BYTES:
                    break
        except (socket.timeout, OSError):
            pass
        if not data:
            return
        raw = data.strip()
        if not raw:
            return
        response = self._handle_request(raw.decode("utf-8", errors="replace"))
        try:
            client.sendall(response.encode("utf-8") + b"\n")
        except (BrokenPipeError, OSError):
            pass

    # -- request dispatch -------------------------------------------------

    def _handle_request(self, data: str) -> str:
        """Parse JSON-RPC, dispatch to handler, return JSON response."""
        try:
            request = json.loads(data)
        except (json.JSONDecodeError, ValueError) as exc:
            return self._error_response(f"Invalid JSON: {exc}", code=-32700)

        if not isinstance(request, dict):
            return self._error_response("Request must be a JSON object.", code=-32600)

        method = request.get("method")
        if not method or not isinstance(method, str):
            return self._error_response("Missing or invalid 'method' field.", code=-32600)

        params = request.get("params", {})
        if not isinstance(params, dict):
            return self._error_response("'params' must be a JSON object.", code=-32602)

        req_id = request.get("id")
        handler = {
            "query": self._handle_query, "ingest": self._handle_ingest,
            "stats": self._handle_stats, "feedback": self._handle_feedback,
        }.get(method)

        if handler is None:
            return self._error_response(f"Unknown method: {method}", code=-32601, req_id=req_id)
        try:
            return self._success_response(handler(params), req_id=req_id)
        except Exception as exc:
            logger.debug("Handler error: %s", traceback.format_exc())
            return self._error_response(str(exc), req_id=req_id)

    # -- method handlers --------------------------------------------------

    def _handle_query(self, params: dict) -> dict:
        query_text = params.get("query", "")
        if not query_text:
            raise ValueError("'query' parameter is required.")
        budget = params.get("budget")
        if budget is not None:
            budget = int(budget)
        session_id = params.get("session_id")
        if session_id is None:
            session_id = self.analytics.start_session()
        else:
            self.analytics.start_session(session_id)

        results = self.engine.query(
            query_text=query_text, budget=budget, session_id=session_id,
            use_compression=params.get("use_compression", True),
        )
        from mnemosyne.models import estimate_tokens
        return {
            "query": query_text, "session_id": session_id,
            "budget": budget if budget is not None else self.config.retrieval.token_budget,
            "tokens_used": sum(estimate_tokens(r.chunk.compressed or r.chunk.content) for r in results),
            "results": [
                {
                    "chunk_id": r.chunk.chunk_id, "file": r.file_path,
                    "lines": [r.chunk.line_start, r.chunk.line_end],
                    "symbol": r.chunk.symbol_name, "type": r.chunk.chunk_type,
                    "scores": r.scores, "content": r.chunk.compressed or r.chunk.content,
                    "is_delta": r.is_delta, "is_stale": getattr(r, "is_stale", False),
                    "tokens": estimate_tokens(r.chunk.compressed or r.chunk.content),
                }
                for r in results
            ],
        }

    def _handle_ingest(self, params: dict) -> dict:
        from mnemosyne.bloom import BloomFilter
        from mnemosyne.audit import AuditLog
        from mnemosyne.ingest import Ingester

        bloom_path = os.path.join(self._mnemosyne_dir, "bloom.bin")
        try:
            bloom = BloomFilter.load(bloom_path) if os.path.isfile(bloom_path) else BloomFilter()
        except Exception:
            logger.warning("Could not load Bloom filter in daemon — creating new")
            bloom = BloomFilter()

        ingester = Ingester(
            project_root=self.project_root, config=self.config, store=self.store,
            bloom=bloom, tfidf_backend=self.tfidf,
            audit=AuditLog(os.path.join(self._mnemosyne_dir, "audit.log")),
        )
        # P2: Validate all ingest paths are within the project root
        paths = params.get("paths")
        if paths:
            validated = []
            for p in paths:
                real = os.path.realpath(os.path.join(self.project_root, p))
                if not real.startswith(self.project_root + os.sep) and real != self.project_root:
                    raise ValueError(f"Path '{p}' resolves outside project root — rejected.")
                validated.append(real)
            paths = validated
        stats = ingester.ingest(paths=paths, full=params.get("full", False), dry_run=False)
        try:
            bloom.save(bloom_path)
        except Exception:
            logger.warning("Could not save Bloom filter in daemon")
        # Rebuild inverted index so queries reflect new content.
        try:
            all_embs = self.store.get_all_sparse_embeddings()
            if all_embs and hasattr(self.tfidf, "build_inverted_index"):
                self.tfidf.build_inverted_index(all_embs)
        except Exception:
            logger.error("Failed to rebuild inverted index after ingest — queries may return stale results")
        return stats

    def _handle_stats(self, params: dict) -> dict:
        return {
            "project_root": self.project_root,
            "files_indexed": self.store.count_files(),
            "chunks": self.store.count_chunks(),
            "total_tokens": self.store.total_tokens(),
            "usage_events": self.store.count_usage_events(),
        }

    def _handle_feedback(self, params: dict) -> dict:
        chunk_id = params.get("chunk_id")
        if chunk_id is None:
            raise ValueError("'chunk_id' parameter is required.")
        chunk_id = int(chunk_id)
        event_type = params.get("event_type", "used")
        valid_types = ("retrieved", "selected", "used", "discarded")
        if event_type not in valid_types:
            raise ValueError(f"'event_type' must be one of {valid_types}, got '{event_type}'.")
        session_id = params.get("session_id")
        if session_id is None:
            session_id = self.analytics.start_session()
        else:
            self.analytics.start_session(session_id)
        self.analytics.record(chunk_id=chunk_id, event_type=event_type, query_text=params.get("query_text"))
        return {"recorded": True, "chunk_id": chunk_id, "event_type": event_type}

    # -- response helpers -------------------------------------------------

    @staticmethod
    def _success_response(result, req_id=None) -> str:
        resp: dict = {"result": result, "error": None}
        if req_id is not None:
            resp["id"] = req_id
        return json.dumps(resp)

    @staticmethod
    def _error_response(message: str, code: int = -32000, req_id=None) -> str:
        resp: dict = {"result": None, "error": {"code": code, "message": message}}
        if req_id is not None:
            resp["id"] = req_id
        return json.dumps(resp)


# -- CLI helpers ----------------------------------------------------------

def read_pid(project_root: str) -> int | None:
    """Read daemon PID from PID file.  Returns None if absent."""
    pid_path = os.path.join(project_root, ".mnemosyne", "daemon.pid")
    if not os.path.isfile(pid_path):
        return None
    try:
        with open(pid_path, "r") as f:
            return int(f.read().strip())
    except (ValueError, OSError):
        return None


def is_daemon_alive(project_root: str) -> bool:
    """Check whether the daemon process is running."""
    pid = read_pid(project_root)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
