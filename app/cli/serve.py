#!/usr/bin/env python3
"""Start a single model worker and the employer evaluation screen."""

import argparse
from pathlib import Path

from app.config import Settings, get_settings
from app.paths import project_path


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    if args.env_file is not None:
        env_file = project_path(args.env_file)
        if not env_file.is_file():
            parser.error(f"ayar dosyasi bulunamadi: {env_file}")
        settings = Settings(_env_file=env_file)
    else:
        settings = get_settings()
    import uvicorn

    from app.api.main import create_app

    uvicorn.run(
        create_app(settings),
        host=args.host or settings.host,
        port=args.port or settings.port,
        workers=1,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
