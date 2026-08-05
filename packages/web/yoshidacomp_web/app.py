"""FastAPI application factory.

API は /api 配下、WebGL2 フロントエンドは / にマウントする。同一オリジンで
配信するため CORS は不要で、バイナリ応答の ``X-*`` ヘッダをブラウザから直接読める。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .routers import browse, generate, lut, session, target, view

STATIC_DIR = Path(__file__).parent / "static"

# ``Host`` ヘッダとして受け付ける名前。
#
# CORS が無いので他オリジンのページはこのサーバの応答を読めない -- が、その
# 保護は **DNS リバインディングで迂回できる**。攻撃サイトが自分のドメインを
# 127.0.0.1 に再解決させると、ブラウザから見て同一オリジンになり CORS の判定
# 自体が起きない。``/api/browse`` はマシン全体のファイル名を返すので、この
# 検査はその機能と一体である。
#
# ``testserver`` は ``TestClient`` の既定 ``base_url``。既存の ``TestClient(...)``
# は11ファイル85箇所にあり、全部に ``base_url`` を渡すのはこの変更に見合わない。
# 許して失うものは無い: DNS リバインディングは名前解決を要する攻撃であり、
# ``testserver`` は単一ラベルで登録可能なドメインではないので、ブラウザに
# この ``Host`` を送らせる経路が無い。``Host`` を自由に作れるローカルのプロセスは
# そもそもソケットへ直接届くので、この検査の対象ではない。
#
# IPv6 リテラルは入れない。``TrustedHostMiddleware`` はホスト部を
# ``host.split(":")[0]`` で取るので ``[::1]:8123`` は ``[`` になり、書いても
# 効かない。サーバは IPv4 ループバック束縛なので ``::1`` では到達もできない。
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]


def create_app() -> FastAPI:
    """Build the FastAPI app with all routers and the static mount."""
    app = FastAPI(title="Yoshida Compensation Matrix Viewer", version="0.1.0")

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(session.router, prefix="/api", tags=["session"])
    app.include_router(target.router, prefix="/api", tags=["target"])
    app.include_router(lut.router, prefix="/api", tags=["lut"])
    app.include_router(view.router, prefix="/api", tags=["view"])
    app.include_router(generate.router, prefix="/api", tags=["generate"])
    app.include_router(browse.router, prefix="/api", tags=["browse"])

    # Mount the frontend last so /api/* wins. html=True serves index.html at /.
    # フロントは最後にマウントする。先にすると /api/* が奪われる。
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


app = create_app()
