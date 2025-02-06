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
        if not isinstance(gt, (str, int, float, bool, list, dict)):
            raise TypeError(f"Unsupported type for gt: {type(gt)}")
        if self.gt:
            logger.warning(f"GT already exists: {self.gt} \n Overwriting...")
        self.gt = gt
        logger.debug(f"Added gt: {self.gt}")

    def add_context(self, context: str|List[str]):
        if isinstance(context, str):
            if not context.strip():
                logger.warning("Empty or whitespace-only context string provided")
            fin_context = [context]
        elif isinstance(context, list):
            fin_context = []
            for cntxt in context:
                if not isinstance(cntxt, str):
                    try:
                        cntxt = str(cntxt)
                    except Exception as e:
                        logger.warning('Cannot cast an element to string... Skipping')
                fin_context.append(cntxt)
            if not any(c for c in fin_context if c and c.strip()):
                logger.warning("No valid context strings provided")
        else:
            fin_context = []
            try:
                fin_context = [str(context)]
            except Exception as e:
                logger.warning('Cannot cast the context to string... Skipping')
        self.context = fin_context
        logger.debug(f"Added context: {self.context}")