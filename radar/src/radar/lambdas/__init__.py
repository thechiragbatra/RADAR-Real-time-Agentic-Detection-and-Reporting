"""AWS Lambda handlers.

    stream_consumer.handler   Kinesis -> features -> score -> DynamoDB / SQS / Firehose
    investigate.handler       SQS (flagged txns) -> investigation agent -> DynamoDB cases / SNS
    drift_monitor.handler     scheduled: PSI on the last 24h of features -> CloudWatch / EventBridge

Each handler is a thin adapter over the same library code the tests exercise; AWS clients
are created lazily and cached across warm invocations.
"""
