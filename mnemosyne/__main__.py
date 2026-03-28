# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Entry point for ``python -m mnemosyne``.

The full CLI is implemented in ``mnemosyne.cli`` (built separately).
This module's sole responsibility is to invoke it.
"""

from mnemosyne.cli import main

if __name__ == "__main__":
    main()
