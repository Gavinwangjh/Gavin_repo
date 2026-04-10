"""
Logger configuration for the app.

- Provides a standard logging setup for workflows, scripts, and tests.
- Initialize once at the start of the application using "setup_logging()".
- Log initialize with command "log = structlog.get_logger(__name__)" in each file required logging.
"""

import logging
import logging.handlers
import sys
from pathlib import Path

import structlog

from config.constants import AppEnv


def _get_shared_processors():
    """
    Processors that run on every log event, regardless of environment.
    These enrich the event dict before it reaches the renderer.
    """
    return [
        structlog.contextvars.merge_contextvars,  # thread-local/async context (request_id, user_id itd.)
        structlog.stdlib.add_logger_name,  # log modul name
        structlog.stdlib.add_log_level,  # "info", "warning" ...
        structlog.stdlib.ExtraAdder(),  # stdlib extra= dict
        structlog.processors.TimeStamper(fmt="iso"),  # ISO 8601 timestamp
        structlog.processors.CallsiteParameterAdder(  # filename + lineno
            # (exchange za %(filename)s:%(lineno)d)
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
        structlog.processors.StackInfoRenderer(),  # stack info if exists
        structlog.processors.UnicodeDecoder(),
    ]


def _configure_stdlib_logging(
    level: int,
    shared_processors: list,
    log_file_name: str,
    app_env: AppEnv,
    root_path: Path,
) -> None:
    """
    Bridges structlog into the stdlib logging system.

    Any library that uses stdlib logging.getLogger() (httpx, sqlalchemy,
    temporalio, alembic...) will flow through this same pipeline,
    giving you consistent structured output from all sources.
    """
    # ProcessorFormatter is bridge between structlog processor i stdlib handler
    formatter = structlog.stdlib.ProcessorFormatter(
        # foreign_pre_chain handles logs which comes from OUTSIDE (from stdlib library)
        foreign_pre_chain=shared_processors,
        # processors are applied AT END — renderer comes here
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _get_renderer(app_env),
        ],
    )

    handlers: list[logging.Handler] = []

    if app_env == AppEnv.LOCAL:
        log_dir = root_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # RotatingFileHandler - max 10MB, 5 backup files
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / log_file_name,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,  # resets eventual handlers which are already set by libraries
    )

    _suppress_noisy_loggers()


def _suppress_noisy_loggers() -> None:
    """
    Suppress logs from external libraries that are too verbose at INFO level.
    Add here anything that clutters the output.
    """
    noisy: dict[str, int] = {
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "temporalio": logging.WARNING,
        "asyncio": logging.WARNING,
        "alembic.runtime.migration": logging.ERROR,  # migrations only log when something goes wrong
    }
    for name, lvl in noisy.items():
        logging.getLogger(name).setLevel(lvl)


def _get_renderer(app_env: AppEnv):
    """
    LOCAL → human-readable colored output in terminal.
    everything else → JSON (lako parseable u Datadog/Loki/CloudWatch)
    """
    if app_env == AppEnv.LOCAL:
        return structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )
    return structlog.processors.JSONRenderer()


def setup_logging(
    log_file_name: str = "app.log",
    log_level: int = logging.INFO,
) -> None:
    """
    Initialize structlog + stdlib logging pipeline.

    Call once, at the very start of the application (main.py or worker entry point).
    Never call on module import.

    Args:
        log_file_name: Log file name (only in LOCAL env).
        log_level: Log level (default INFO).
    """
    # Lazy import so if that it doesn't break if settings isn't available in test env
    from config.constants import _ROOT_DIR
    from config.settings import settings

    shared_processors = _get_shared_processors()

    structlog.configure(
        processors=shared_processors
        + [
            # wrapper_class mora da bude pre renderera
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,  # performance: cache logger after first use
    )

    _configure_stdlib_logging(
        level=log_level,
        shared_processors=shared_processors,
        log_file_name=log_file_name,
        app_env=settings.APP_ENV,
        root_path=_ROOT_DIR,
    )
