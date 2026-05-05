from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from http.client import HTTPException
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .config import ProxyConfig
from .log_format import (
    LOG,
    cache_hit_rate,
    elapsed_ms,
    format_count,
    format_usage_count,
    log_bytes,
    log_json,
    message_count,
    read_response_body,
    reasoning_content_count,
    reasoning_token_count,
    response_headers,
    sse_data,
    summarize_chat_payload,
    tool_count,
    usage_from_body,
    user_message_count,
)
from .reasoning_store import ReasoningStore, conversation_scope
from .streaming import CursorReasoningDisplayAdapter, StreamAccumulator
from .trace import TraceRequest, TraceWriter
from .transform import (
    PreparedRequest,
    RECOVERY_NOTICE_CONTENT,
    prepare_upstream_request,
    rewrite_response_body,
)


class RequestBodyTooLarge(ValueError):
    pass


@dataclass
class ProxyResponseResult:
    sent: bool
    usage: dict[str, Any] | None = None
    response_body: bytes | None = None


@dataclass
class RequestMetrics:
    total: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    cache_hit: int = 0
    cache_miss: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, latency_ms: float, *, success: bool, cache_miss: bool) -> None:
        with self._lock:
            self.total += 1
            if success:
                self.completed += 1
            else:
                self.failed += 1
            if cache_miss:
                self.cache_miss += 1
            else:
                self.cache_hit += 1
            self.total_latency_ms += latency_ms
            if latency_ms > self.max_latency_ms:
                self.max_latency_ms = latency_ms

    def record_rejected(self) -> None:
        with self._lock:
            self.total += 1
            self.rejected += 1

    @property
    def active(self) -> int:
        return self.total - self.completed - self.failed - self.rejected

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            avg_latency = (
                self.total_latency_ms / max(self.completed, 1)
                if self.completed
                else 0.0
            )
            return {
                "total": self.total,
                "active": self.active,
                "completed": self.completed,
                "failed": self.failed,
                "rejected": self.rejected,
                "cache_hit": self.cache_hit,
                "cache_miss": self.cache_miss,
                "avg_latency_ms": round(avg_latency, 1),
                "max_latency_ms": round(self.max_latency_ms, 1),
            }


@dataclass
class InflightRequest:
    lock: threading.Event
    response_body: bytes | None = None
    content_type: str = "application/json"
    status: int = 200
    error: str | None = None


class DeepSeekProxyServer(ThreadingHTTPServer):
    config: ProxyConfig
    reasoning_store: ReasoningStore
    trace_writer: TraceWriter | None
    metrics: RequestMetrics = field(default_factory=RequestMetrics)
    concurrency_semaphore: threading.Semaphore | None = None
    _inflight: dict[str, InflightRequest] = field(default_factory=dict)
    _inflight_lock: threading.Lock = field(default_factory=threading.Lock)
    _shutdown_event: threading.Event = field(default_factory=threading.Event)
    _active_requests: int = 0
    _active_lock: threading.Lock = field(default_factory=threading.Lock)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.metrics = RequestMetrics()
        self._inflight = {}
        self._inflight_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._active_requests = 0
        self._active_lock = threading.Lock()

    def server_close(self) -> None:
        self._shutdown_event.set()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            with self._active_lock:
                if self._active_requests == 0:
                    break
            time.sleep(0.1)
        super().server_close()


class DeepSeekProxyHandler(BaseHTTPRequestHandler):
    server_version = "DeepSeekPythonProxy/0.1"

    @property
    def config(self) -> ProxyConfig:
        return self.server.config  # type: ignore[return-value]

    @property
    def reasoning_store(self) -> ReasoningStore:
        return self.server.reasoning_store  # type: ignore[return-value]

    @property
    def trace_writer(self) -> TraceWriter | None:
        return getattr(self.server, "trace_writer", None)

    @property
    def metrics(self) -> RequestMetrics:
        return self.server.metrics  # type: ignore[return-value]

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:
        request_path = urlparse(self.path).path
        if self.config.verbose:
            LOG.info(
                "incoming OPTIONS %s from %s",
                request_path,
                self.client_address[0],
            )
        self._send_response_headers(204, [], "sending CORS preflight response")

    def do_GET(self) -> None:
        request_path = urlparse(self.path).path
        if self.config.verbose:
            LOG.info("incoming GET %s from %s", request_path, self.client_address[0])
        if request_path in {"/healthz", "/v1/healthz"}:
            self._send_json(200, {"ok": True})
            return
        if request_path in {"/metrics", "/v1/metrics"}:
            self._send_metrics()
            return
        if request_path in {"/models", "/v1/models"}:
            self._send_models()
            return
        self._send_json(404, {"error": {"message": "Not found"}})

    def _send_metrics(self) -> None:
        metrics_snapshot = self.metrics.snapshot()
        if isinstance(self.reasoning_store, ReasoningStore):
            try:
                metrics_snapshot["schema_version"] = self.reasoning_store.schema_version
            except Exception:
                pass
        self._send_json(200, metrics_snapshot)

    def do_POST(self) -> None:
        if self.server._shutdown_event.is_set():
            self._send_json(503, {"error": {"message": "Server is shutting down"}})
            return

        semaphore = self.server.concurrency_semaphore
        if semaphore is not None and not semaphore.acquire(blocking=False):
            self.metrics.record_rejected()
            self._send_json(
                429,
                {
                    "error": {
                        "message": ("Too many concurrent requests. " "Try again later.")
                    }
                },
            )
            return

        try:
            self._handle_post()
        finally:
            if semaphore is not None:
                semaphore.release()

    def _handle_post(self) -> None:
        started = time.monotonic()
        request_path = urlparse(self.path).path
        trace = self._start_trace(request_path)
        if self.config.verbose:
            LOG.info(
                "incoming POST %s from %s content_length=%s user_agent=%s",
                request_path,
                self.client_address[0],
                self.headers.get("Content-Length", "0"),
                self.headers.get("User-Agent", ""),
            )
        if request_path not in {"/chat/completions", "/v1/chat/completions"}:
            LOG.warning("rejected unsupported POST path=%s status=404", request_path)
            self._send_json(
                404,
                {"error": {"message": "Only /v1/chat/completions is supported"}},
                trace=trace,
            )
            self._finish_trace(trace, "rejected", http_status=404)
            return
        cursor_authorization = self._cursor_authorization()
        if cursor_authorization is None:
            LOG.warning(
                "rejected request path=%s status=401 reason=missing_bearer_token",
                request_path,
            )
            self._send_json(
                401,
                {"error": {"message": "Missing Authorization bearer token"}},
                trace=trace,
            )
            self._finish_trace(trace, "rejected", http_status=401)
            return

        try:
            payload = self._read_json_body()
        except RequestBodyTooLarge as exc:
            LOG.warning(
                "rejected request path=%s status=413 reason=%s", request_path, exc
            )
            self._send_json(413, {"error": {"message": str(exc)}}, trace=trace)
            self._finish_trace(trace, "rejected", http_status=413, reason=str(exc))
            return
        except ValueError as exc:
            LOG.warning(
                "rejected request path=%s status=400 reason=%s", request_path, exc
            )
            self._send_json(400, {"error": {"message": str(exc)}}, trace=trace)
            self._finish_trace(trace, "rejected", http_status=400, reason=str(exc))
            return

        if trace is not None:
            trace.record_cursor_body(payload)

        if self.config.verbose:
            log_json("cursor request body", payload)

        self._log_cursor_request(payload, self.config)

        prepared = prepare_upstream_request(
            payload,
            self.config,
            self.reasoning_store,
            authorization=cursor_authorization,
        )
        if trace is not None:
            trace.record_transform(prepared)
        self._log_context_summary(prepared)
        if prepared.missing_reasoning_messages:
            LOG.warning(
                (
                    "strict missing-reasoning mode rejected request path=%s "
                    "status=409 reason=missing_reasoning_content count=%s"
                ),
                request_path,
                prepared.missing_reasoning_messages,
            )
            self._send_json(
                409,
                {
                    "error": {
                        "message": (
                            "deepseek-cursor-proxy is running in strict "
                            "missing-reasoning mode and cannot automatically "
                            "recover this thinking-mode tool-call history because "
                            "cached DeepSeek reasoning_content is missing for "
                            f"{prepared.missing_reasoning_messages} assistant "
                            "message(s). Restart without "
                            "`--missing-reasoning-strategy reject`, or pass "
                            "`--missing-reasoning-strategy recover`, so the proxy "
                            "can recover from partial chat history automatically."
                        ),
                        "type": "missing_reasoning_content",
                        "code": "missing_reasoning_content",
                        "missing_reasoning_messages": prepared.missing_reasoning_messages,
                    }
                },
                trace=trace,
            )
            self._finish_trace(trace, "rejected", http_status=409)
            return

        if self.config.verbose:
            LOG.info(
                (
                    "upstream request metadata: original_model=%s upstream_model=%s "
                    "patched_reasoning=%s missing_reasoning=%s %s"
                ),
                prepared.original_model,
                prepared.upstream_model,
                prepared.patched_reasoning_messages,
                prepared.missing_reasoning_messages,
                summarize_chat_payload(prepared.payload),
            )

        if self.config.verbose:
            log_json("upstream request body", prepared.payload)

        upstream_body = json.dumps(
            prepared.payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        upstream_url = f"{self.config.upstream_base_url}/chat/completions"
        upstream_headers = self._upstream_headers(
            stream=bool(prepared.payload.get("stream")),
            authorization=cursor_authorization,
        )
        if trace is not None:
            trace.record_upstream_request(
                url=upstream_url,
                headers=upstream_headers,
                body_bytes=upstream_body,
            )

        # Concurrent request deduplication (non-streaming only)
        dedup_key = self._dedup_key(prepared.payload)
        if dedup_key:
            inflight = self._check_or_register_inflight(dedup_key)
            if inflight is not None:
                inflight.lock.wait()
                if inflight.response_body is not None:
                    self._send_deduped_response(
                        inflight.response_body,
                        inflight.content_type,
                        inflight.status,
                        trace=trace,
                    )
                else:
                    self._send_json(
                        502,
                        {
                            "error": {
                                "message": (
                                    f"Deduplicated upstream request failed: "
                                    f"{inflight.error}"
                                )
                            }
                        },
                        trace=trace,
                    )
                if trace is not None:
                    trace.finish("deduped", http_status=inflight.status)
                return

        with self.server._active_lock:
            self.server._active_requests += 1
        try:
            request = Request(
                upstream_url,
                data=upstream_body,
                method="POST",
                headers=upstream_headers,
            )
            self._log_send_summary(prepared)

            try:
                if self.config.verbose:
                    LOG.info("forwarding to %s", upstream_url)
                response = urlopen(request, timeout=self.config.request_timeout)
            except HTTPError as exc:
                LOG.warning(
                    "request failed upstream_status=%s stream=%s elapsed_ms=%s",
                    exc.code,
                    bool(prepared.payload.get("stream")),
                    elapsed_ms(started),
                )
                self._send_upstream_error(exc, trace=trace)
                self._finish_trace(
                    trace,
                    "upstream_error",
                    http_status=exc.code,
                    stream=bool(prepared.payload.get("stream")),
                )
                self._notify_inflight(dedup_key, error=str(exc.code))
                self.metrics.record(elapsed_ms(started), success=False, cache_miss=True)
                return
            except URLError as exc:
                LOG.warning(
                    "upstream request failed elapsed_ms=%s reason=%s",
                    elapsed_ms(started),
                    exc.reason,
                )
                self._send_json(
                    502,
                    {"error": {"message": f"Upstream request failed: {exc.reason}"}},
                    trace=trace,
                )
                self._finish_trace(trace, "upstream_error", http_status=502)
                self._notify_inflight(dedup_key, error=str(exc.reason))
                self.metrics.record(elapsed_ms(started), success=False, cache_miss=True)
                return

            with response:
                upstream_status = getattr(response, "status", 200)
                if self.config.verbose:
                    LOG.info(
                        "upstream response status=%s stream=%s elapsed_ms=%s",
                        upstream_status,
                        bool(prepared.payload.get("stream")),
                        elapsed_ms(started),
                    )
                if prepared.payload.get("stream"):
                    sent_response = self._proxy_streaming_response(
                        response,
                        prepared.original_model,
                        prepared.payload["messages"],
                        prepared.cache_namespace,
                        prepared.recovery_notice,
                        trace=trace,
                    )
                else:
                    sent_response = self._proxy_regular_response(
                        response,
                        prepared.original_model,
                        prepared.payload["messages"],
                        prepared.cache_namespace,
                        prepared.recovery_notice,
                        trace=trace,
                    )
                    if sent_response.sent and dedup_key and sent_response.response_body:
                        with self.server._inflight_lock:
                            req = self.server._inflight.get(dedup_key)
                            if req is not None:
                                req.response_body = sent_response.response_body
                                req.status = upstream_status
                                req.lock.set()
                if not sent_response.sent:
                    self._finish_trace(
                        trace,
                        "client_disconnected",
                        http_status=upstream_status,
                        stream=bool(prepared.payload.get("stream")),
                    )
                    self._notify_inflight(dedup_key, error="client_disconnected")
                    self.metrics.record(
                        elapsed_ms(started), success=False, cache_miss=True
                    )
                    return
                self._log_stats_summary(sent_response.usage)
                self._finish_trace(
                    trace,
                    "completed",
                    http_status=upstream_status,
                    stream=bool(prepared.payload.get("stream")),
                )
                cache_miss = prepared.patched_reasoning_messages == 0 and (
                    prepared.missing_reasoning_messages > 0
                    or prepared.recovered_reasoning_messages > 0
                )
                self.metrics.record(
                    elapsed_ms(started), success=True, cache_miss=cache_miss
                )
        finally:
            with self.server._active_lock:
                self.server._active_requests -= 1
            if dedup_key:
                self._unregister_inflight(dedup_key)

    @staticmethod
    def _dedup_key(payload: dict[str, Any]) -> str | None:
        if payload.get("stream"):
            return None
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return None
        canonical = {
            "model": payload.get("model"),
            "messages": [
                {
                    k: v
                    for k, v in m.items()
                    if k in {"role", "content", "tool_calls", "tool_call_id", "name"}
                }
                for m in messages
            ],
            "tools": payload.get("tools"),
            "tool_choice": payload.get("tool_choice"),
            "thinking": payload.get("thinking"),
            "reasoning_effort": payload.get("reasoning_effort"),
        }
        raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _check_or_register_inflight(self, dedup_key: str) -> InflightRequest | None:
        with self.server._inflight_lock:
            if dedup_key in self.server._inflight:
                return self.server._inflight[dedup_key]
            req = InflightRequest(lock=threading.Event())
            self.server._inflight[dedup_key] = req
            return None

    def _notify_inflight(self, dedup_key: str | None, error: str | None = None) -> None:
        if not dedup_key:
            return
        with self.server._inflight_lock:
            req = self.server._inflight.get(dedup_key)
            if req is not None:
                req.error = error
                req.lock.set()

    def _unregister_inflight(self, dedup_key: str | None) -> None:
        if not dedup_key:
            return
        with self.server._inflight_lock:
            req = self.server._inflight.pop(dedup_key, None)
            if req is not None and not req.lock.is_set():
                req.lock.set()

    def _send_deduped_response(
        self,
        body: bytes,
        content_type: str,
        status: int,
        *,
        trace: TraceRequest | None = None,
    ) -> None:
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        }
        if trace is not None:
            trace.record_cursor_response(status=status, headers=headers, body=body)
        sent_headers = self._send_response_headers(
            status,
            [
                ("Content-Type", headers["Content-Type"]),
                ("Content-Length", headers["Content-Length"]),
            ],
            "sending deduped response headers",
        )
        if sent_headers:
            self._write_to_client(body, "sending deduped response body")

    def _start_trace(self, request_path: str) -> TraceRequest | None:
        writer = self.trace_writer
        if writer is None:
            return None
        try:
            return writer.start_request(
                method=self.command,
                path=request_path,
                client_address=self.client_address[0],
                headers={name: value for name, value in self.headers.items()},
            )
        except OSError as exc:
            LOG.warning("failed to start request trace: %s", exc)
            return None

    def _finish_trace(
        self,
        trace: TraceRequest | None,
        status: str,
        **extra: Any,
    ) -> None:
        if trace is None:
            return
        try:
            trace.finish(status, **extra)
        except OSError as exc:
            LOG.warning("failed to write request trace: %s", exc)

    def _cursor_authorization(self) -> str | None:
        auth_header = self.headers.get("Authorization", "")
        scheme, separator, token = auth_header.strip().partition(" ")
        if separator != " " or scheme.lower() != "bearer" or not token.strip():
            return None
        return f"Bearer {token.strip()}"

    def _send_cors_headers(self) -> None:
        if not self.config.cors:
            return
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Origin, Content-Type, Accept, Authorization",
        )
        self.send_header("Access-Control-Expose-Headers", "Content-Length")
        self.send_header("Access-Control-Allow-Credentials", "true")

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        *,
        trace: TraceRequest | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        if trace is not None:
            trace.record_cursor_response(
                status=status,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
                body=body,
            )
        sent_headers = self._send_response_headers(
            status,
            [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ],
            "sending JSON response headers",
        )
        if sent_headers:
            self._write_to_client(body, "sending JSON response body")

    def _send_response_headers(
        self,
        status: int,
        headers: list[tuple[str, str]],
        disconnect_context: str,
    ) -> bool:
        try:
            self.send_response(status)
            self._send_cors_headers()
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
        except (BrokenPipeError, ConnectionError) as exc:
            LOG.warning("client disconnected while %s: %s", disconnect_context, exc)
            return False
        return True

    def _write_to_client(
        self,
        body: bytes,
        disconnect_context: str,
        *,
        flush: bool = False,
    ) -> bool:
        try:
            self.wfile.write(body)
            if flush:
                self.wfile.flush()
        except (BrokenPipeError, ConnectionError) as exc:
            LOG.warning("client disconnected while %s: %s", disconnect_context, exc)
            return False
        return True

    def _send_models(self) -> None:
        created = int(time.time())
        model_ids = list(
            dict.fromkeys(
                [
                    self.config.upstream_model,
                    "deepseek-v4-pro",
                    "deepseek-v4-flash",
                ]
            )
        )
        models = [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "deepseek",
            }
            for model_id in model_ids
        ]
        self._send_json(200, {"object": "list", "data": models})

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length < 0:
            raise ValueError("Invalid Content-Length")
        if length > self.config.max_request_body_bytes:
            raise RequestBodyTooLarge(
                f"Request body is too large; limit is {self.config.max_request_body_bytes} bytes"
            )
        raw_body = self.rfile.read(length)
        if not raw_body:
            raise ValueError("Request body is empty")
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _upstream_headers(self, stream: bool, authorization: str) -> dict[str, str]:
        headers = {
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": self.server_version,
        }
        accept_language = self.headers.get("Accept-Language")
        if accept_language:
            headers["Accept-Language"] = accept_language
        return headers

    def _send_upstream_error(
        self,
        exc: HTTPError,
        *,
        trace: TraceRequest | None = None,
    ) -> None:
        body = read_response_body(exc)
        if self.config.verbose:
            log_bytes("upstream error body", body)
        headers = {
            "Content-Type": exc.headers.get("Content-Type", "application/json"),
            "Content-Length": str(len(body)),
        }
        if trace is not None:
            trace.record_upstream_response(
                status=exc.code,
                headers={name: value for name, value in exc.headers.items()},
                body=body,
            )
            trace.record_cursor_response(status=exc.code, headers=headers, body=body)
        sent_headers = self._send_response_headers(
            exc.code,
            [
                ("Content-Type", headers["Content-Type"]),
                ("Content-Length", headers["Content-Length"]),
            ],
            "sending upstream error headers",
        )
        if sent_headers:
            self._write_to_client(body, "sending upstream error body")

    def _proxy_regular_response(
        self,
        response: Any,
        original_model: str,
        request_messages: list[dict[str, Any]],
        cache_namespace: str,
        recovery_notice: str | None = None,
        trace: TraceRequest | None = None,
    ) -> ProxyResponseResult:
        body = read_response_body(response)
        upstream_body = body
        usage = usage_from_body(upstream_body)
        try:
            body = rewrite_response_body(
                body,
                original_model,
                self.reasoning_store,
                request_messages,
                cache_namespace,
                content_prefix=recovery_notice,
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            LOG.warning("failed to rewrite upstream JSON response: %s", exc)

        if self.config.verbose:
            log_bytes("cursor response body", body)

        headers = {
            "Content-Type": response.headers.get("Content-Type", "application/json"),
            "Content-Length": str(len(body)),
        }
        if trace is not None:
            trace.record_upstream_response(
                status=getattr(response, "status", 200),
                headers=response_headers(response),
                body=upstream_body,
                stream=False,
            )
            try:
                upstream_payload = json.loads(upstream_body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                upstream_payload = None
            if isinstance(upstream_payload, dict):
                trace.record_usage(upstream_payload.get("usage"))
            trace.record_cursor_response(
                status=getattr(response, "status", 200),
                headers=headers,
                body=body,
            )

        sent_headers = self._send_response_headers(
            getattr(response, "status", 200),
            [
                ("Content-Type", headers["Content-Type"]),
                ("Content-Length", headers["Content-Length"]),
            ],
            "sending upstream response headers",
        )
        if not sent_headers:
            return ProxyResponseResult(False, usage)
        sent = self._write_to_client(body, "sending upstream response body")
        return ProxyResponseResult(sent, usage, response_body=body if sent else None)

    def _proxy_streaming_response(
        self,
        response: Any,
        original_model: str,
        request_messages: list[dict[str, Any]],
        cache_namespace: str,
        recovery_notice: str | None = None,
        trace: TraceRequest | None = None,
    ) -> ProxyResponseResult:
        if trace is not None:
            trace.record_upstream_response(
                status=getattr(response, "status", 200),
                headers=response_headers(response),
                stream=True,
            )
            trace.record_cursor_response(
                status=getattr(response, "status", 200),
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                },
            )
        sent_headers = self._send_response_headers(
            getattr(response, "status", 200),
            [
                ("Content-Type", "text/event-stream"),
                ("Cache-Control", "no-cache"),
                ("Connection", "close"),
            ],
            "sending streaming response headers",
        )
        if not sent_headers:
            return ProxyResponseResult(False)
        self.close_connection = True

        scope = conversation_scope(request_messages, cache_namespace)
        response_contexts = [(scope, request_messages)]
        accumulator = StreamAccumulator()
        usage: dict[str, Any] | None = None
        display_adapter = (
            CursorReasoningDisplayAdapter()
            if self.config.cursor_display_reasoning
            else None
        )
        finalized = False
        pending_recovery_notice = recovery_notice
        while True:
            try:
                line = response.readline()
            except (HTTPException, OSError) as exc:
                LOG.warning("upstream streaming response read failed: %s", exc)
                return ProxyResponseResult(False, usage)
            if not line:
                break
            (
                rewritten,
                finalized,
                pending_recovery_notice,
                chunk_usage,
            ) = self._rewrite_sse_line(
                line,
                original_model,
                accumulator,
                cache_namespace,
                response_contexts,
                display_adapter,
                pending_recovery_notice,
                trace,
            )
            if chunk_usage is not None:
                usage = chunk_usage
            if trace is not None:
                trace.record_stream_chunk(line, rewritten)
            if not self._write_to_client(
                rewritten, "sending streaming response chunk", flush=True
            ):
                return ProxyResponseResult(False, usage)
            if finalized:
                break

        if not finalized:
            if self.config.verbose:
                log_json("model streaming assistant messages", accumulator.messages())
            stored = sum(
                accumulator.store_reasoning(
                    self.reasoning_store,
                    scope,
                    cache_namespace,
                    prior_messages,
                )
                for scope, prior_messages in response_contexts
            )
            if self.config.verbose and stored:
                LOG.info("stored %s streaming reasoning cache key(s)", stored)
        return ProxyResponseResult(True, usage)

    def _rewrite_sse_line(
        self,
        line: bytes,
        original_model: str,
        accumulator: StreamAccumulator,
        cache_namespace: str,
        response_contexts: list[tuple[str, list[dict[str, Any]]]],
        display_adapter: CursorReasoningDisplayAdapter | None,
        recovery_notice: str | None = None,
        trace: TraceRequest | None = None,
    ) -> tuple[bytes, bool, str | None, dict[str, Any] | None]:
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            return line, False, recovery_notice, None

        data = stripped[len(b"data:") :].strip()
        if data == b"[DONE]":
            if self.config.verbose:
                log_json("model streaming assistant messages", accumulator.messages())
            stored = sum(
                accumulator.store_reasoning(
                    self.reasoning_store,
                    scope,
                    cache_namespace,
                    prior_messages,
                )
                for scope, prior_messages in response_contexts
            )
            if self.config.verbose and stored:
                LOG.info("stored %s streaming reasoning cache key(s)", stored)
            prefix = b""
            if display_adapter is None:
                if recovery_notice:
                    prefix += sse_data(
                        recovery_notice_chunk(original_model, recovery_notice)
                    )
                return prefix + b"data: [DONE]\n\n", True, None, None
            closing_chunk = display_adapter.flush_chunk(original_model)
            if closing_chunk is not None:
                prefix += sse_data(closing_chunk)
            if recovery_notice:
                prefix += sse_data(
                    recovery_notice_chunk(original_model, recovery_notice)
                )
            return prefix + b"data: [DONE]\n\n", True, None, None

        try:
            chunk = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return line, False, recovery_notice, None

        if isinstance(chunk, dict):
            if recovery_notice and inject_recovery_notice(chunk, recovery_notice):
                recovery_notice = None
            accumulator.ingest_chunk(chunk)
            stored = sum(
                accumulator.store_ready_reasoning(
                    self.reasoning_store,
                    scope,
                    cache_namespace,
                    prior_messages,
                )
                for scope, prior_messages in response_contexts
            )
            if self.config.verbose and stored:
                LOG.info("stored %s streaming reasoning cache key(s)", stored)
            chunk_usage = chunk.get("usage")
            if trace is not None:
                trace.record_usage(chunk_usage)
            if display_adapter is not None:
                display_adapter.rewrite_chunk(chunk)
            if "model" in chunk:
                chunk["model"] = original_model
            ending = b"\r\n" if line.endswith(b"\r\n") else b"\n"
            return (
                (
                    b"data: "
                    + json.dumps(
                        chunk, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                    + ending
                ),
                False,
                recovery_notice,
                chunk_usage if isinstance(chunk_usage, dict) else None,
            )
        return line, False, recovery_notice, None

    def _log_cursor_request(
        self,
        payload: dict[str, Any],
        config: ProxyConfig,
    ) -> None:
        model = str(payload.get("model") or config.upstream_model)
        LOG.info(
            "┌ cursor  model=%s messages=%s tools=%s",
            model,
            format_count(message_count(payload)),
            format_count(tool_count(payload)),
        )

    def _log_context_summary(self, prepared: PreparedRequest) -> None:
        LOG.info(
            "├ context filled=%s missing=%s recovered=%s dropped=%s status=%s",
            format_count(prepared.patched_reasoning_messages),
            format_count(prepared.missing_reasoning_messages),
            format_count(prepared.recovered_reasoning_messages),
            format_count(prepared.recovery_dropped_messages),
            context_status(prepared),
        )

    def _log_send_summary(self, prepared: PreparedRequest) -> None:
        LOG.info(
            "├ send    user_msgs=%s messages=%s tools=%s reasoning_content=%s",
            format_count(user_message_count(prepared.payload)),
            format_count(message_count(prepared.payload)),
            format_count(tool_count(prepared.payload)),
            format_count(reasoning_content_count(prepared.payload)),
        )

    def _log_stats_summary(self, usage: dict[str, Any] | None) -> None:
        LOG.info(
            "└ stats   prompt=%s output=%s reasoning=%s cache_hit=%s",
            format_usage_count(usage, "prompt_tokens"),
            format_usage_count(usage, "completion_tokens"),
            format_count(reasoning_token_count(usage)),
            cache_hit_rate(usage),
        )


def context_status(prepared: PreparedRequest) -> str:
    if prepared.recovered_reasoning_messages:
        return "recovered"
    if prepared.missing_reasoning_messages:
        return "missing"
    return "ok"


def inject_recovery_notice(chunk: dict[str, Any], notice: str) -> bool:
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        if "content" not in delta and not delta.get("tool_calls"):
            continue
        existing_content = delta.get("content")
        delta["content"] = notice + (
            existing_content if isinstance(existing_content, str) else ""
        )
        return True
    return False


def recovery_notice_chunk(
    model: str,
    notice: str = RECOVERY_NOTICE_CONTENT,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-deepseek-cursor-proxy-recovery",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": notice},
                "finish_reason": None,
            }
        ],
    }
