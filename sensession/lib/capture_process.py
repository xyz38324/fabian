"""
Context Manager class to maintain subprocesses of a single capture phase.

Takes care of starting potentially multiple subprocesses and cleaning them
up upon exiting or destruction of the context manager object.
"""

import time
import queue
import shlex
import traceback
import subprocess
from typing import (
    Any,  # type: ignore[W0611]   # Import is used in Task.process typehint
)
from dataclasses import dataclass

from loguru import logger

from sensession.lib.shell import shell_run


@dataclass
class Task:
    """
    Struct to connect a process and a potential cleanup command.
    Named task because upon destroying these are "worked off" in an effort to
    clean up properly.
    """

    process: "subprocess.Popen[Any] | None"  # Running background process
    cleanup_cmd: "str | None"  # Task specific cleanup shell command


class CaptureProcess:
    """
    CaptureProcess is a ContextManager wrapper that ensures that a capture process is
    properly cleaned up (i.e. the subprocess terminated).
    """

    def __init__(self):
        """
        Create a capture process for process lifetime management
        """
        # Context object can manage multiple subprocesses; store them in a LIFO queue
        self.processes = queue.LifoQueue()

    def __del__(self):
        """
        Destructor.
        Ensure that everything is stopped.
        """
        self.teardown()

    def start_process(
        self,
        shell_command: "str | None" = None,
        cleanup_command: "str | None" = None,
        suppress=True,
    ):
        """
        Start a process to be managed by this object.

        Args:
            shell_command   : Verbatim command (no cmd line expansion supported!).
            cleanup_command : Verbatim command to call for cleanup when closing process.
            suppress        : Whether to suppres terminal stdout output from processes.
        """

        process = None
        stdout = None

        if shell_command:
            if suppress:
                stdout = subprocess.DEVNULL

            fmt_cmd = shell_command.replace(" --", "\\ \n\t--")
            logger.info(f"Starting subprocess: \n{fmt_cmd}")

            process = subprocess.Popen(
                shlex.split(shell_command), shell=False, stdout=stdout
            )
        else:
            assert cleanup_command != "", "Neither shell nor cleanup command provided!!"

        # Remember started processes via queue
        self.processes.put(
            Task(
                process=process,
                cleanup_cmd=cleanup_command,
            )
        )

    def teardown(self):
        """
        Stop started capturing processes.
        """
        # Slight timeout to hope for empty buffers everywhere
        if self.processes.empty():
            return

        logger.trace(
            f"Stopping all registered {self.processes.qsize()} subprocesses ..."
        )
        time.sleep(0.5)

        while not self.processes.empty():
            task: Task = self.processes.get()

            if task.cleanup_cmd:
                shell_run(task.cleanup_cmd)

            if task.process:
                task.process.terminate()

            self.processes.task_done()

        logger.trace("Capture processes stopped!")

    def __enter__(self):
        """
        Context enter. This is trivial, the interesting part here is automatic cleanup
        on context exit.
        """
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """
        Context exit. Ensure that subprocesses are properly terminated.
        """
        logger.trace("Stopping CSI capture processes ...")
        self.teardown()

        if exc_type is not None:
            logger.error("Failed to stop capture processes gracefully!")
            traceback.print_exception(exc_type, exc_value, tb)
            return False

        logger.debug("CSI capture subprocesses should be stopped now.")
        return True
