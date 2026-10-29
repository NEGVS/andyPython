import asyncio
import json
import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")

from fastapi.testclient import TestClient

from src import qianWenProStream1 as cli
from src import qianWenProStream2 as svc
from src import qianWenStreamUtil as util


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


class FakeCloseableSyncStream(FakeSyncStream):
    def __init__(self, chunks):
        super().__init__(chunks)
        self.closed = threading.Event()

    def __exit__(self, *args):
        self.closed.set()
        return False


def endless_chunks(text):
    while True:
        yield make_chunk(text)


class CliTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(util, "CHAR_INTERVAL_SECONDS", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

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

    def test_stream_answer_prints_char_by_char(self):
        fake = mock.Mock()
        fake.chat.completions.create.return_value = FakeSyncStream([make_chunk(reasoning="思考"), make_chunk("你好"), make_chunk(None), make_chunk("呀"), make_chunk(usage=object())])
        with mock.patch("builtins.print") as printed:
            self.assertEqual(cli.stream_answer(fake, [{"role": "user", "content": "hi"}]), "你好呀")
        outputs = [c.args[0] for c in printed.call_args_list if c.args]
        self.assertEqual(outputs[:2], [f"{cli.GRAY}思{cli.RESET}", f"{cli.GRAY}考{cli.RESET}"])
        self.assertEqual([o for o in outputs if o in ("你", "好", "呀")], ["你", "好", "呀"])

    def test_stream_answer_keyboard_interrupt_returns_partial_and_closes_upstream(self):
        stream = FakeCloseableSyncStream(endless_chunks("部分"))
        fake = mock.Mock()
        fake.chat.completions.create.return_value = stream

        def fake_print(*args, **kwargs):
            if args and args[0] == "分":
                raise KeyboardInterrupt

        with mock.patch("builtins.print", side_effect=fake_print):
            self.assertEqual(cli.stream_answer(fake, []), "部分")
        self.assertTrue(stream.closed.wait(1))

    def test_stream_answer_upstream_error_propagates(self):
        fake = mock.Mock()
        fake.chat.completions.create.side_effect = ConnectionError("down")
        with mock.patch("builtins.print"), self.assertRaises(ConnectionError):
            cli.stream_answer(fake, [])


class StreamUtilTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(util, "CHAR_INTERVAL_SECONDS", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_read_chat_stream_events(self):
        usage = SimpleNamespace(prompt_tokens=1)
        fake = mock.Mock()
        fake.chat.completions.create.return_value = FakeSyncStream([make_chunk("", reasoning=""), make_chunk(reasoning="想"), make_chunk("答"), make_chunk(usage=usage), SimpleNamespace(choices=[SimpleNamespace(delta=None)], usage=None)])
        self.assertEqual(list(util.read_chat_stream(fake, "m", [], False)), [(util.REASONING, "想"), (util.ANSWER, "答"), (util.USAGE, usage)])
        kwargs = fake.chat.completions.create.call_args.kwargs
        self.assertEqual((kwargs["model"], kwargs["extra_body"], kwargs["stream_options"]), ("m", {"enable_thinking": False}, {"include_usage": True}))

    def test_typewriter_splits_text_and_passes_through_others(self):
        usage = object()

        def pieces():
            yield util.REASONING, "想想"
            yield util.ANSWER, "你好"
            yield util.ANSWER, ""
            yield util.USAGE, usage

        self.assertEqual(list(util.typewriter(pieces())), [(util.REASONING, "想"), (util.REASONING, "想"), (util.ANSWER, "你"), (util.ANSWER, "好"), (util.USAGE, usage)])

    def test_typewriter_empty_source(self):
        self.assertEqual(list(util.typewriter(iter_empty())), [])

    def test_typewriter_reraises_source_error_after_output(self):
        def pieces():
            yield util.ANSWER, "半"
            raise TimeoutError("read timeout")

        out = []
        with self.assertRaises(TimeoutError):
            for item in util.typewriter(pieces()):
                out.append(item)
        self.assertEqual(out, [(util.ANSWER, "半")])

    def test_typewriter_close_stops_source(self):
        source_closed = threading.Event()

        def pieces():
            try:
                while True:
                    yield util.ANSWER, "字"
            finally:
                source_closed.set()

        chars = util.typewriter(pieces())
        self.assertEqual(next(chars), (util.ANSWER, "字"))
        chars.close()
        self.assertTrue(source_closed.wait(1))

    def test_char_interval_catch_up(self):
        threshold = util.CATCH_UP_BACKLOG_CHUNKS
        with mock.patch.object(util, "CHAR_INTERVAL_SECONDS", 0.03):
            self.assertEqual(util.char_interval(0), 0.03)
            self.assertEqual(util.char_interval(threshold), 0.03)
            self.assertAlmostEqual(util.char_interval(threshold * 4), 0.03 / 4)


def iter_empty():
    yield from ()


if __name__ == "__main__":
    unittest.main()
