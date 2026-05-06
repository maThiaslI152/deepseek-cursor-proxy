from __future__ import annotations

import argparse
from dataclasses import replace
import logging
from pathlib import Path
import sys
import threading
from typing import Any
from urllib.parse import urlparse

from .config import (
    ProxyConfig,
    default_config_path,
    default_reasoning_content_path,
)
from .log_format import LOG
from .logging import configure_logging
from .reasoning_store import ReasoningStore
from .trace import TraceWriter
from .tunnel import NgrokTunnel, local_tunnel_target


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local DeepSeek Cursor proxy")
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        help=f"YAML config file, default {default_config_path()}",
    )
    parser.add_argument("--host", help="Bind host, default from config or 127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        help="Bind port, default from config or 9000",
    )
    parser.add_argument(
        "--model",
        help=(
            "Fallback DeepSeek model when the request has no model, "
            "default from config or deepseek-v4-pro"
        ),
    )
    parser.add_argument(
        "--base-url",
        help=("DeepSeek base URL, default from config or https://api.deepseek.com"),
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled", "pass-through"],
        help="DeepSeek thinking mode, default from config or enabled",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "max", "xhigh"],
        help="DeepSeek reasoning effort, default from config or high",
    )
    parser.add_argument(
        "--reasoning-content-path",
        type=Path,
        help=(
            "SQLite reasoning_content cache path, "
            f"default {default_reasoning_content_path()}"
        ),
    )
    parser.add_argument(
        "--ngrok",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Start an ngrok tunnel and print the Cursor base URL",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log detailed request metadata and full payloads",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="Write full structured request traces to this directory",
    )
    parser.add_argument(
        "--display-reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mirror reasoning_content into Cursor-visible content",
    )
    parser.add_argument(
        "--collapsible-reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Markdown details for mirrored reasoning when display is enabled",
    )
    # Support typo aliases for backward compatibility if needed, though cli.py is relatively new
    parser.add_argument(
        "--collasible-reasoning",
        dest="collapsible_reasoning",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cors",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Send permissive CORS headers",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        help="Upstream request timeout in seconds, default from config or 300",
    )
    parser.add_argument(
        "--max-request-body-bytes",
        type=int,
        help="Maximum accepted request body size, default from config",
    )
    parser.add_argument(
        "--reasoning-cache-max-age-seconds",
        type=int,
        help="Maximum reasoning cache row age in seconds, default from config",
    )
    parser.add_argument(
        "--reasoning-cache-max-rows",
        type=int,
        help="Maximum reasoning cache rows, default from config",
    )
    parser.add_argument(
        "--missing-reasoning-strategy",
        choices=["recover", "reject"],
        help=(
            "What to do when required reasoning_content is missing: "
            "recover (friendly default) or reject (strict debugging mode)"
        ),
    )
    parser.add_argument(
        "--clear-reasoning-cache",
        action="store_true",
        help="Clear the local reasoning_content SQLite cache and exit",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        help="Maximum concurrent upstream requests, default from config or 20",
    )
    return parser


def warn_if_insecure_upstream(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http":
        return
    host = parsed.hostname or ""
    if host in {"127.0.0.1", "localhost", "::1"}:
        return
    LOG.warning("upstream base_url uses plain HTTP; bearer tokens may be exposed")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        config = ProxyConfig.from_file(config_path=args.config_path)
    except ValueError as exc:
        configure_logging(verbose=bool(args.verbose))
        LOG.error("%s", exc)
        return 2
    updates: dict[str, Any] = {}
    if args.host is not None:
        updates["host"] = args.host
    if args.port is not None:
        updates["port"] = args.port
    if args.model is not None:
        updates["upstream_model"] = args.model
    if args.base_url is not None:
        updates["upstream_base_url"] = args.base_url.rstrip("/")
    if args.thinking is not None:
        updates["thinking"] = args.thinking
    if args.reasoning_effort is not None:
        updates["reasoning_effort"] = args.reasoning_effort
    if args.reasoning_content_path is not None:
        updates["reasoning_content_path"] = args.reasoning_content_path
    if args.ngrok is not None:
        updates["ngrok"] = args.ngrok
    if args.verbose is not None:
        updates["verbose"] = args.verbose
    if args.trace_dir is not None:
        updates["trace_dir"] = args.trace_dir
    if args.display_reasoning is not None:
        updates["display_reasoning"] = args.display_reasoning
    if args.collapsible_reasoning is not None:
        updates["collapsible_reasoning"] = args.collapsible_reasoning
    if args.cors is not None:
        updates["cors"] = args.cors
    if args.request_timeout is not None:
        updates["request_timeout"] = args.request_timeout
    if args.max_request_body_bytes is not None:
        updates["max_request_body_bytes"] = args.max_request_body_bytes
    if args.max_concurrent_requests is not None:
        updates["max_concurrent_requests"] = args.max_concurrent_requests
    if args.reasoning_cache_max_age_seconds is not None:
        updates["reasoning_cache_max_age_seconds"] = (
            args.reasoning_cache_max_age_seconds
        )
    if args.reasoning_cache_max_rows is not None:
        updates["reasoning_cache_max_rows"] = args.reasoning_cache_max_rows
    if args.missing_reasoning_strategy is not None:
        updates["missing_reasoning_strategy"] = args.missing_reasoning_strategy
    if updates:
        config = replace(config, **updates)

    configure_logging(verbose=config.verbose)
    warn_if_insecure_upstream(config.upstream_base_url)
    store = ReasoningStore(
        config.reasoning_content_path,
        max_age_seconds=config.reasoning_cache_max_age_seconds,
        max_rows=config.reasoning_cache_max_rows,
    )
    if args.clear_reasoning_cache:
        deleted = store.clear()
        LOG.info("cleared %s reasoning cache row(s)", deleted)
        store.close()
        return 0
    trace_writer: TraceWriter | None = None
    if config.trace_dir is not None:
        try:
            trace_writer = TraceWriter(config.trace_dir)
        except OSError as exc:
            LOG.error("failed to initialize trace directory: %s", exc)
            store.close()
            return 2
    from .server import DeepSeekProxyHandler, DeepSeekProxyServer

    server = DeepSeekProxyServer((config.host, config.port), DeepSeekProxyHandler)
    server.config = config
    server.reasoning_store = store
    server.trace_writer = trace_writer
    server.concurrency_semaphore = threading.Semaphore(config.max_concurrent_requests)

    tunnel: NgrokTunnel | None = None
    public_url: str | None = None
    if config.ngrok:
        target_url = local_tunnel_target(config.host, config.port)
        tunnel = NgrokTunnel(target_url)
        try:
            public_url = tunnel.start()
        except RuntimeError as exc:
            LOG.error("%s", exc)
            server.server_close()
            store.close()
            return 2

    local_base_url = f"http://{config.host}:{config.port}/v1"
    api_base_url = (
        f"{public_url.rstrip('/')}/v1" if public_url is not None else local_base_url
    )

    LOG.info(
        "default_model: %s (%s, %s)",
        config.upstream_model,
        "thinking" if config.thinking == "enabled" else "no thinking",
        config.reasoning_effort,
    )

    if config.verbose:
        display_reasoning = "off"
        if config.display_reasoning:
            display_reasoning = (
                "on (collapsible)" if config.collapsible_reasoning else "on"
            )
        LOG.info("display_reasoning: %s", display_reasoning)
        LOG.info("missing_reasoning_strategy: %s", config.missing_reasoning_strategy)
        LOG.info("reasoning_content_path: %s", config.reasoning_content_path)
        LOG.info("logging mode=verbose metadata=detailed bodies=true")
        LOG.warning(
            "verbose logging enabled; prompts and code may be written to stdout"
        )
    else:
        LOG.info("local_base_url: %s", local_base_url)
        LOG.info("api_base_url: %s", api_base_url)
        LOG.info("logging mode=normal metadata=safe_summaries bodies=false")

    if trace_writer is not None:
        LOG.info("trace session directory: %s", trace_writer.session_dir)
        LOG.warning("trace logging enabled; prompts and code will be written to disk")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("shutting down")
    finally:
        if tunnel is not None:
            tunnel.stop()
        server.server_close()
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
