import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")

from fastapi.testclient import TestClient

from src import qianWenProStream1 as cli
from src import qianWenProStream2 as svc


def make_chunk(content=None, usage=None, reasoning=None):
    choices = [SimpleNamespace(delta=SimpleNamespace(content=content, reasoning_content=reasoning))] if content is not None or reasoning is not None else []
    return SimpleNamespace(choices=choices, usage=usage)


class FakeAsyncStream:
    def __init__(self, chunks, error=None):
        self.chunks, self.error, self.closed = chunks, error, False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def __aiter__(self):
        for c in self.chunks:
            yield c
        if self.error:
            raise self.error


class FakeSyncStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.chunks)


def parse_events(body: str) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in body.split("\n\n") if line.startswith("data: ")]


class ChatStreamApiTest(unittest.TestCase):
    def setUp(self):
        self.http = TestClient(svc.app)

    def post(self, messages, headers=None):
        return self.http.post(svc.CHAT_STREAM_PATH, json={"messages": messages}, headers=headers or {})

    def mock_create(self, stream=None, side_effect=None):
        return mock.patch.object(svc.client.chat.completions, "create", new=mock.AsyncMock(return_value=stream, side_effect=side_effect))

    def test_stream_success(self):
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        stream = FakeAsyncStream([make_chunk("", reasoning=""), make_chunk(reasoning="想"), make_chunk("你"), make_chunk(""), make_chunk("好\n世界"), make_chunk(usage=usage)])
        with self.mock_create(stream) as create:
            resp = self.post([{"role": "user", "content": "hi"}], {"X-Request-ID": "r1"})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(parse_events(resp.text), [
            {"type": "reasoning", "content": "想"},
            {"type": "delta", "content": "你"},
            {"type": "delta", "content": "好\n世界"},
            {"type": "done", "usage": {"promptTokens": 5, "completionTokens": 3, "totalTokens": 8}},
        ])
        self.assertTrue(stream.closed)
        self.assertEqual(create.await_args.kwargs["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(create.await_args.kwargs["extra_body"], {"enable_thinking": svc.ENABLE_THINKING})

    def test_done_without_usage(self):
        with self.mock_create(FakeAsyncStream([make_chunk("a")])):
            events = parse_events(self.post([{"role": "user", "content": "hi"}]).text)
        self.assertEqual(events[-1], {"type": "done"})

    def test_upstream_create_error(self):
        with self.mock_create(side_effect=TimeoutError("boom")):
            events = parse_events(self.post([{"role": "user", "content": "hi"}]).text)
        self.assertEqual(events, [{"type": "error", "code": "TimeoutError", "message": svc.ERROR_MESSAGE}])

    def test_upstream_error_mid_stream(self):
        stream = FakeAsyncStream([make_chunk("半")], error=ConnectionError("reset"))
        with self.mock_create(stream):
            events = parse_events(self.post([{"role": "user", "content": "hi"}]).text)
        self.assertEqual(events, [{"type": "delta", "content": "半"}, {"type": "error", "code": "ConnectionError", "message": svc.ERROR_MESSAGE}])
        self.assertTrue(stream.closed)

    def test_semaphore_released_after_stream(self):
        with self.mock_create(FakeAsyncStream([make_chunk("a")])):
            for _ in range(svc.MAX_CONCURRENT_STREAMS + 1):
                self.assertEqual(self.post([{"role": "user", "content": "hi"}]).status_code, 200)

    def test_busy_returns_429(self):
        with mock.patch.object(svc, "semaphore", asyncio.Semaphore(0)):
            resp = self.post([{"role": "user", "content": "hi"}])
        self.assertEqual(resp.status_code, 429)

    def test_invalid_requests_return_422(self):
        cases = [
            [],
            [{"role": "user", "content": "   "}],
            [{"role": "tool", "content": "x"}],
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            [{"role": "user"}],
            [{"role": "user", "content": "x" * (svc.MAX_CONTENT_LENGTH + 1)}],
            [{"role": "user", "content": "x"}] * (svc.MAX_MESSAGES + 1),
        ]
        for messages in cases:
            with self.subTest(messages=str(messages)[:60]):
                self.assertEqual(self.post(messages).status_code, 422)
        self.assertEqual(self.http.post(svc.CHAT_STREAM_PATH, json={}).status_code, 422)


class CliTest(unittest.TestCase):
    def test_trim_history_keeps_recent_rounds(self):
        history = [{"role": r, "content": str(i)} for i in range(cli.MAX_HISTORY_ROUNDS + 5) for r in ("user", "assistant")]
        trimmed = cli.trim_history(history)
        self.assertEqual(len(trimmed), cli.MAX_HISTORY_ROUNDS * 2)
        self.assertEqual(trimmed[0]["role"], "user")
        self.assertEqual(trimmed[-1], history[-1])

    def test_trim_history_short_unchanged(self):
        history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        self.assertIs(cli.trim_history(history), history)
        self.assertEqual(cli.trim_history([]), [])

    def test_stream_answer_joins_content(self):
        fake = mock.Mock()
        fake.chat.completions.create.return_value = FakeSyncStream([make_chunk(reasoning="思考"), make_chunk("你"), make_chunk(None), make_chunk("好"), make_chunk(usage=object())])
        with mock.patch("builtins.print") as printed:
            self.assertEqual(cli.stream_answer(fake, [{"role": "user", "content": "hi"}]), "你好")
        self.assertIn(f"{cli.GRAY}思考{cli.RESET}", [c.args[0] for c in printed.call_args_list if c.args])

    def test_stream_answer_keyboard_interrupt_returns_partial(self):
        def chunks():
            yield make_chunk("部分")
            raise KeyboardInterrupt

        fake = mock.Mock()
        fake.chat.completions.create.return_value = FakeSyncStream(chunks())
        with mock.patch("builtins.print"):
            self.assertEqual(cli.stream_answer(fake, []), "部分")


if __name__ == "__main__":
    unittest.main()
