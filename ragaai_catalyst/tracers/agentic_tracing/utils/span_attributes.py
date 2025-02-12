import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)
logging_level = (
    logger.setLevel(logging.DEBUG)
    if os.getenv("DEBUG")
    else logger.setLevel(logging.INFO)
)


class SpanAttributes:
    def __init__(self, name, project_id: Optional[int] = None):
        self.name = name
        self.tags = []
        self.metadata = {}
        self.metrics = []
        self.local_metrics = []
        self.feedback = None
        self.project_id = project_id
        self.trace_attributes = ["tags", "metadata", "metrics"]
        self.gt = None
        self.context = None

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

    # TODO: Add validation to check if all the required parameters are present
    def execute_metrics(self, **kwargs: Any):
        name = kwargs.get("name")
        model = kwargs.get("model")
        provider = kwargs.get("provider")
        display_name = kwargs.get("display_name", None)
        mapping = kwargs.get("mapping", None)

        if isinstance(name, str):
            metrics = [{
                "name": name
            }]
        else:
            metrics = name if isinstance(name, list) else [name] if isinstance(name, dict) else []

        for metric in metrics:
            if not isinstance(metric, dict):
                raise ValueError(f"Expected dict, got {type(metric)}")

            if "name" not in metric:
                raise ValueError("Metric must contain 'name'")

            metric_name = metric["name"]
            if metric_name in self.local_metrics:
                count = sum(1 for m in self.local_metrics if m.startswith(metric_name))
                metric_name = f"{metric_name}_{count + 1}"

            prompt =None
            context = None
            response = None
            # if mapping is not None:
            #     prompt = mapping['prompt']
            #     context = mapping['context']
            #     response = mapping['response']
            new_metric = {
                "name": metric_name,
                "model": model,
                "provider": provider,
                "project_id": self.project_id,
                # "prompt": prompt,
                # "context": context,
                # "response": response,
                "displayName": display_name,
                "mapping": mapping
            }
            self.local_metrics.append(new_metric)

    def add_gt(self, gt: Any):
        if not isinstance(gt, (str, int, float, bool, list, dict)):
            raise TypeError(f"Unsupported type for gt: {type(gt)}")
        if self.gt:
            logger.warning(f"GT already exists: {self.gt} \n Overwriting...")
        self.gt = gt
        logger.debug(f"Added gt: {self.gt}")

    def add_context(self, context: Any):
        if isinstance(context, str):
            if not context.strip():
                logger.warning("Empty or whitespace-only context string provided")
            self.context = str(context)
        else:
            try:
                self.context = str(context)
            except Exception as e:
                logger.warning('Cannot cast the context to string... Skipping')
        logger.debug(f"Added context: {self.context}")
