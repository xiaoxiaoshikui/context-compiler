"""Network/VPC configuration helpers, unrelated to secrets handling."""

DEFAULT_SUBNETS = ["subnet-a", "subnet-b", "subnet-c"]


def resolve_subnet(stage: str) -> str:
    return DEFAULT_SUBNETS[hash(stage) % len(DEFAULT_SUBNETS)]
