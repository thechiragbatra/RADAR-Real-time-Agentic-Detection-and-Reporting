"""Agentic investigation layer.

A flagged transaction is handed to an LLM agent that investigates it the way a fraud
analyst would: pull the customer's recent activity, check the device and the merchant,
look for similar past cases, then commit to a structured verdict through a
``submit_verdict`` tool whose schema is validated before anything downstream sees it.

The agent loop (``investigator.py``) is backend-agnostic: ``BedrockBackend`` talks to
Amazon Bedrock's Converse API; ``HeuristicBackend`` is a deterministic rule-based
investigator that drives the same tools and serves as the baseline the LLM must beat.
"""

from radar.agent.investigator import Investigator
from radar.agent.llm import BedrockBackend, HeuristicBackend
from radar.agent.tools import ToolBox

__all__ = ["BedrockBackend", "HeuristicBackend", "Investigator", "ToolBox"]
