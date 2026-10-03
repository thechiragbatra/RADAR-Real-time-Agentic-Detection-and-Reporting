"""RADAR — Real-time Agentic Detection and Reporting.

Real-time fraud detection with an agentic investigation layer, on AWS.

Package layout:
    simulator/  synthetic transaction world with injected fraud patterns
    features/   stateful feature engineering shared by training and serving
    model/      training, evaluation, drift detection, champion/challenger registry
    scoring/    FastAPI scoring service
    agent/      Bedrock-backed investigation agent with tools and guardrails
    lambdas/    AWS Lambda handlers (stream consumer, investigator, drift monitor)
"""

__version__ = "0.1.0"
