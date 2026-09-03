from __future__ import annotations

import logging
import sys
from pathlib import Path
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler


LOGGER_NAME = "insurance_paraplanner"


class AgentTerminalLogger(BaseCallbackHandler):
    """Expose observable model lifecycle events without logging hidden reasoning."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.logger.info("model_started provider=gemini")

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.logger.info("model_started provider=gemini")

    def on_llm_end(self, response, **kwargs):
        outputs = []
        for generation in getattr(response, "generations", []):
            for item in generation:
                text = getattr(item, "text", None)
                if text:
                    outputs.append(text)
                message = getattr(item, "message", None)
                content = getattr(message, "content", None)
                if content:
                    outputs.append(content)
        self.logger.info("model_intermediate_output=%s", outputs)

    def on_llm_error(self, error, **kwargs):
        self.logger.error("model_error=%s", error)


def configure_logging(log_directory: str | Path | None = None) -> tuple[logging.Logger, Path]:
    directory = Path(log_directory) if log_directory else Path(__file__).parents[1] / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"advisory_{uuid4().hex[:12]}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    terminal_handler = logging.StreamHandler(sys.stdout)
    terminal_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(terminal_handler)
    logger.propagate = False
    logger.info("run_started log_file=%s", log_path)
    return logger, log_path
