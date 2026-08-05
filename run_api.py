#!/usr/bin/env python3
"""Run the LuminaSQL FastAPI gateway."""

import uvicorn

from config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
