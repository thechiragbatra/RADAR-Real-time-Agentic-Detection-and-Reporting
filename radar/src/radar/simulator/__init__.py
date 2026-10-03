"""Synthetic transaction world.

Generates users, merchants and devices with realistic spending behaviour, then injects
five fraud patterns (account takeover, card testing, velocity burst, geo jump, merchant
collusion) plus the legitimate "hard negative" behaviours that make fraud detection hard
in practice: travel, device upgrades and one-off big purchases.
"""
