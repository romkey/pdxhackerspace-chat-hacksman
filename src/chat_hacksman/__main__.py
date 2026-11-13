"""Command line: `chat-hacksman` or `python -m chat_hacksman`. Serves the web app."""

import argparse
import logging
import sys
from collections.abc import Sequence

import sentry_sdk
import uvicorn

from chat_hacksman import __version__
from chat_hacksman.config import ConfigError, Settings
from chat_hacksman.server import create_app

logger = logging.getLogger("chat_hacksman")


def configure_sentry(dsn: str | None) -> bool:
    """Turn on Sentry error reporting if a DSN is set."""
    if not dsn:
        return False
    sentry_sdk.init(dsn=dsn, release=f"chat-hacksman@{__version__}")
    logger.info("Sentry is enabled for error reporting")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chat-hacksman", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args(argv)

    try:
        settings = Settings.from_env()
    except ConfigError as e:
        parser.exit(2, f"chat-hacksman: {e}\n")

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    configure_sentry(settings.sentry_dsn)
    app = create_app(settings)

    logger.info("chat-hacksman %s listening on %s:%d", __version__, settings.host, settings.port)
    # uvicorn handles SIGTERM (sent by `docker stop`) and SIGINT itself: it
    # stops accepting connections, finishes in-flight requests, then returns.
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None)
    logger.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
