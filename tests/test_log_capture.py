"""Проверяет захват stdout/stderr, дублирование вывода и восстановление потоков.

Охватываются потоки без дескрипторов, Loguru и Python logging, нативные и
подпроцессные записи, дочерние процессы с унаследованными каналами, повторные
изолированные захваты, очистка при исключениях и ошибки записи файла. Захват
должен сохранять вывод консоли и восстанавливать исходные потоки.
"""

import os
import signal
import subprocess
import sys
import textwrap
from io import StringIO
from pathlib import Path

from taei.logging import capture_output


def test_capture_falls_back_for_streams_without_file_descriptors(
    tmp_path: Path, monkeypatch
):
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    log_path = tmp_path / "fallback.log"

    with capture_output(log_path):
        print("fallback café")
        print("fallback error", file=sys.stderr)

    assert "fallback café" in stdout.getvalue()
    assert "fallback error" in stderr.getvalue()
    recorded = log_path.read_text(encoding="utf-8")
    assert "fallback café" in recorded
    assert "fallback error" in recorded


def test_fallback_cached_logger_stream_remains_writable_after_capture(
    tmp_path: Path, monkeypatch
):
    from loguru import logger

    from taei.logging import configure_logging

    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    log_path = tmp_path / "fallback.log"

    with capture_output(log_path):
        configure_logging(level="INFO")
        logger.info("during café")

    finished_log = log_path.read_text(encoding="utf-8")
    logger.info("after capture")

    assert "during café" in finished_log
    assert "after capture" not in log_path.read_text(encoding="utf-8")
    assert "during café" in stderr.getvalue()
    assert "after capture" in stderr.getvalue()
    assert not stdout.closed
    assert not stderr.closed
    assert "ValueError" not in stderr.getvalue()


def test_capture_returns_while_child_keeps_inherited_pipe_open(tmp_path: Path):
    log_path = tmp_path / "inherited.log"
    code = textwrap.dedent(
        f"""
        import subprocess
        import sys
        import time
        from taei.logging import capture_output

        started = time.monotonic()
        child = None
        try:
            with capture_output({str(log_path)!r}):
                print('before child outlives capture')
                child = subprocess.Popen([
                    sys.executable, '-c', 'import time; time.sleep(30)'
                ])
            elapsed = time.monotonic() - started
            assert elapsed < 2, f'capture teardown took {{elapsed:.2f}}s'
        finally:
            if child is not None:
                child.terminate()
                child.wait(timeout=2)
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise AssertionError(
            "capture teardown hung on inherited child descriptors"
        ) from None

    assert process.returncode == 0, stderr
    assert "before child outlives capture" in log_path.read_text(encoding="utf-8")


def test_capture_configures_nonblocking_readers_before_starting_relays(
    tmp_path: Path, monkeypatch
):
    import threading

    import taei.logging as logging_module

    configured: set[int] = set()
    configured_at_start: list[int] = []
    original_set_blocking = logging_module.os.set_blocking
    original_start = threading.Thread.start

    def record_nonblocking(fd, blocking):
        original_set_blocking(fd, blocking)
        if not blocking:
            configured.add(fd)

    def record_relay_start(thread):
        if thread.name == "taei-output-relay":
            configured_at_start.append(len(configured))
        return original_start(thread)

    monkeypatch.setattr(logging_module.os, "set_blocking", record_nonblocking)
    monkeypatch.setattr(threading.Thread, "start", record_relay_start)

    with capture_output(tmp_path / "ordering.log"):
        print("reader setup is race-free")

    assert configured_at_start == [2, 2]


def test_capture_preserves_console_and_records_python_logging(tmp_path: Path):
    log_path = tmp_path / "run.log"
    code = f"""
import logging
import sys
import warnings
from loguru import logger
from taei.logging import capture_output, configure_logging
configure_logging(level='INFO')
logging.basicConfig(level=logging.WARNING)
with capture_output({str(log_path)!r}):
    print('stdout café')
    sys.stderr.write('stderr naïve\\n')
    logging.basicConfig(level=logging.WARNING)
    logging.warning('stdlib warning')
    warnings.warn('captured warning', UserWarning)
    logger.info('loguru message')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )

    recorded = log_path.read_text(encoding="utf-8")
    assert "stdout café" in result.stdout
    assert "stderr naïve" in result.stderr
    assert "stdout café" in recorded
    assert "stderr naïve" in recorded
    assert "stdlib warning" in recorded
    assert "captured warning" in recorded
    assert "loguru message" in recorded


def test_capture_includes_os_and_inherited_subprocess_output(tmp_path: Path):
    log_path = tmp_path / "run.log"
    code = textwrap.dedent(
        f"""
        import os
        import subprocess
        import sys
        from taei.logging import capture_output
        with capture_output({str(log_path)!r}):
            os.write(1, 'native café\\n'.encode())
            os.write(2, b'native stderr\\n')
            child = (
                "import os; os.write(1, b'child\\\\n'); "
                "os.write(2, b'child stderr\\\\n')"
            )
            subprocess.run([sys.executable, '-c', child], check=True)
        """
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
    )

    recorded = log_path.read_text(encoding="utf-8")
    assert "native café" in recorded
    assert "native stderr" in recorded
    assert "child" in recorded
    assert "child stderr" in recorded


def test_capture_repeated_runs_are_isolated_and_exceptions_restore_streams(
    tmp_path: Path,
):
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    code = f"""
import sys
from taei.logging import capture_output
original_stdout = sys.stdout
with capture_output({str(first)!r}):
    print('first run')
try:
    with capture_output({str(second)!r}):
        print('second run')
        raise RuntimeError('expected')
except RuntimeError:
    pass
assert sys.stdout is original_stdout
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()

    first_text = first.read_text(encoding="utf-8")
    second_text = second.read_text(encoding="utf-8")
    assert "first run" in first_text
    assert "second run" in second_text
    assert "second run" not in first_text


def test_capture_propagates_log_write_failure_and_restores_streams(
    tmp_path: Path,
):
    code = textwrap.dedent(
        f"""
        import os
        import sys
        from pathlib import Path
        import taei.logging as logging_module
        from taei.logging import capture_output

        original_open = Path.open
        class FailingLog:
            def __init__(self, file): self.file = file
            def write(self, _text): raise OSError('simulated log disk failure')
            def flush(self): self.file.flush()
            def close(self): self.file.close()
        def failing_open(path, *args, **kwargs):
            return FailingLog(original_open(path, *args, **kwargs))
        Path.open = failing_open
        original_stdout, original_stderr = sys.stdout, sys.stderr
        try:
            with capture_output({str(tmp_path / "failure.log")!r}):
                os.write(1, b'trigger disk write\\n')
        except OSError as exc:
            assert str(exc) == 'simulated log disk failure'
        else:
            raise AssertionError('capture hid a logfile write failure')
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
