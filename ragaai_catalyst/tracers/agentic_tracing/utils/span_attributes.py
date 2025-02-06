import os
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)
logging_level = (
    logger.setLevel(logging.DEBUG)
    if os.getenv("DEBUG")
    else logger.setLevel(logging.INFO)
)


class SpanAttributes:
    def __init__(self, name):
        self.name = name
        self.tags = []
        self.metadata = {}
        self.metrics = []
        self.feedback = None
        self.trace_attributes = ["tags", "metadata", "metrics"]
        self.gt = None
        self.context = []

    def add_tags(self, tags: str | List[str]):
        if isinstance(tags, str):
            tags = [tags]
        self.tags.extend(tags)
        logger.debug(f"Added tags: {tags}")

    def add_metadata(self, metadata):
        self.metadata.update(metadata)
        logger.debug(f"Added metadata: {metadata}")

    def add_metrics(
        self,
        name: str,
        score: float | int,
        reasoning: str = "",
        cost: float = None,
        latency: float = None,
        metadata: Dict[str, Any] = {},
        config: Dict[str, Any] = {},
    ):
        self.metrics.append(
            {
                "name": name,
                "score": score,
                "reason": reasoning,
                "source": "user",
                "cost": cost,
                "latency": latency,
                "metadata": metadata,
                "mappings": [],
                "config": config,
            }
        )
        logger.debug(f"Added metrics: {self.metrics}")

    def add_feedback(self, feedback: Any):
        self.feedback = feedback
        logger.debug(f"Added feedback: {self.feedback}")

    def add_gt(self, gt: Any):
        if self.gt:
            logger.warning(f"GT already exists: {self.gt} \n Overwriting...")
        self.gt = gt
        logger.debug(f"Added gt: {self.gt}")

    def add_context(self, context: str|List[str]):
        if isinstance(context, str):
            context = [context]
        self.context = context
        logger.debug(f"Added context: {self.context}")