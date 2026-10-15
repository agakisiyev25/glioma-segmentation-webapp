"""Run the backend: ``python -m app`` (from the backend folder).

Host and port come from ``GLIOMA_HOST`` / ``GLIOMA_PORT`` (default
127.0.0.1:8000). Use this rather than a bare ``uvicorn`` command: it sets a
long keep-alive, so status polls are not dropped while a job is keeping the
interpreter busy.
"""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:get_app",
        factory=True,
        host=os.environ.get("GLIOMA_HOST", "127.0.0.1"),
        port=int(os.environ.get("GLIOMA_PORT", "8000")),
        timeout_keep_alive=120,
    )
