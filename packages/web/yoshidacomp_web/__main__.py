"""Entry point: ``python -m yoshidacomp_web``.

エントリポイント。``python -m yoshidacomp_web`` で起動する。
"""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Serve the viewer on localhost:8000.

    ビューアを localhost:8000 で配信する。
    """
    uvicorn.run("yoshidacomp_web.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
