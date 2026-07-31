"""Entry point: ``python -m yoshidacomp_web``."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Serve the viewer on localhost:8000."""
    uvicorn.run("yoshidacomp_web.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
