import asyncio
from unittest.mock import patch

import app as module
import pytest
import starlette.formparsers
from app import AnalyzeRequestGuard


def scope():
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/analyze",
        "raw_path": b"/api/analyze",
        "query_string": b"",
        "headers": [(b"content-type", b"multipart/form-data; boundary=limit")],
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 1234),
    }


def test_chunked_overflow_returns_one_413_and_closes_spooled_file(monkeypatch):
    monkeypatch.setattr(module, "MAX_REQUEST", 2 * 1024 * 1024)
    opened = []
    original = starlette.formparsers.SpooledTemporaryFile

    def tracked(*args, **kwargs):
        result = original(*args, **kwargs)
        opened.append(result)
        return result

    chunks = [
        b'--limit\r\nContent-Disposition: form-data; name="file"; '
        b'filename="large.zip"\r\nContent-Type: application/zip\r\n\r\n',
        b"a" * (1536 * 1024),
        b"a" * (1024 * 1024),
        b"\r\n--limit--\r\n",
    ]

    async def exercise():
        messages = []
        iterator = iter(chunks)

        async def receive():
            part = next(iterator, None)
            if part is None:
                return {"type": "http.disconnect"}
            return {"type": "http.request", "body": part, "more_body": True}

        async def send(message):
            messages.append(message)

        await module.app(scope(), receive, send)
        starts = [m for m in messages if m["type"] == "http.response.start"]
        assert [m["status"] for m in starts] == [413]

    with (
        patch.object(starlette.formparsers, "SpooledTemporaryFile", tracked),
        patch.object(module, "_analyze", side_effect=AssertionError("endpoint ran")),
    ):
        asyncio.run(exercise())
    assert opened
    assert all(file.closed for file in opened)


def test_request_gate_covers_parse_and_releases_after_cancellation():
    async def exercise():
        release = asyncio.Event()
        entered = asyncio.Event()
        active = 0

        async def downstream(request_scope, receive, send):
            nonlocal active
            active += 1
            if active == 2:
                entered.set()
            await release.wait()

        guard = AnalyzeRequestGuard(downstream)

        async def receive():
            return {"type": "http.disconnect"}

        async def discard(message):
            pass

        first = asyncio.create_task(guard(scope(), receive, discard))
        second = asyncio.create_task(guard(scope(), receive, discard))
        await asyncio.wait_for(entered.wait(), timeout=2)
        responses = []

        async def collect(message):
            responses.append(message)

        await guard(scope(), receive, collect)
        assert responses[0]["status"] == 503
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        assert guard.gate._value == 2
        release.set()
        await guard(scope(), receive, discard)
        assert guard.gate._value == 2

    asyncio.run(exercise())


def test_request_gate_releases_after_application_error():
    async def exercise():
        async def fail(request_scope, receive, send):
            raise RuntimeError("test failure")

        guard = AnalyzeRequestGuard(fail)
        with pytest.raises(RuntimeError, match="test failure"):
            await guard(scope(), None, None)
        assert guard.gate._value == 2

    asyncio.run(exercise())
