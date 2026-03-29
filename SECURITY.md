# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.4.x   | Yes       |
| < 0.4   | No        |

## Threat Model

Mnemosyne is a local-first tool that operates on a user's own filesystem.
It does not make network calls, host services, or process untrusted input
in its default configuration.

### Trust boundaries

- **Filesystem access:** Mnemosyne reads and writes files within the project
  root directory and the `.mnemosyne/` subdirectory. Path containment is
  enforced via `os.path.realpath()` — symlinks are resolved before
  validation. Paths outside the project root are rejected.

- **Daemon socket:** When running in daemon mode, communication occurs over
  a Unix domain socket at `.mnemosyne/mnemosyne.sock` with permissions set
  to `0600` (owner-only). The daemon validates all ingest paths against the
  project root before processing.

- **Serialization:** All persistent data uses JSON or SQLite. No pickle,
  eval, marshal, or other unsafe deserialization is used anywhere in the
  codebase.

- **Database:** SQLite with WAL mode. No network-accessible database. No
  multi-user access model. FTS5 queries are parameterized (no SQL injection).

### Out of scope

- Network security (Mnemosyne makes no network calls in normal operation)
- Authentication/authorization (single-user local tool)
- Denial of service (local tool, user controls their own resources)

## Known Limitations

- **TOCTOU on path validation:** A race condition exists between path
  resolution and file read. This is inherent to POSIX filesystems and not
  exploitable in Mnemosyne's single-user context.

- **FTS5 query injection:** While FTS5 special characters are escaped, the
  FTS5 query language is limited and cannot cause data modification. The
  worst case is unexpected empty results.

## Reporting a Vulnerability

Please report security issues to **security@castnettechnology.com**.

We will acknowledge receipt within 48 hours and provide an initial
assessment within 7 days. Critical issues affecting data integrity or
path traversal will be patched within 72 hours of confirmation.

Do not open public GitHub issues for security vulnerabilities.
