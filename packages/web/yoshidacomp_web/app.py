"""FastAPI application factory.

API は /api 配下、WebGL2 フロントエンドは / にマウントする。同一オリジンで
配信するため CORS は不要で、バイナリ応答の ``X-*`` ヘッダをブラウザから直接読める。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routers import session

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    """Build the FastAPI app with all routers and the static mount."""
    app = FastAPI(title="Yoshida Compensation Matrix Viewer", version="0.1.0")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(session.router, prefix="/api", tags=["session"])

    # Mount the frontend last so /api/* wins. html=True serves index.html at /.
    # フロントは最後にマウントする。先にすると /api/* が奪われる。
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


app = create_app()
