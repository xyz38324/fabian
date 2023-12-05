"""
Little shell abstraction to centralize shell command calls
"""

import subprocess

from loguru import logger


def shell_run(
    command: str, capture=False, timeout_s: int = 600
) -> subprocess.CompletedProcess:
    """
    Run shell command

    Args:
        command   : command to run
        capture   : Whether to capture the output of the script
        timeout_s : Timeout (in seconds) for the subprocess

    Returns:
        The completed shell process info struct
    """

    logger.debug(f"Executing shell command as subprocess: {command}")

    result = subprocess.run(
        command,
        shell=True,
        text=True,
        check=True,
        capture_output=capture,
        timeout=timeout_s,
    )

    result.check_returncode()
    logger.trace(f"Executed `{command}` successfully")

    return result
