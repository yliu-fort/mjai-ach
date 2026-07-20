"""Floor layer: leaf helpers with no internal imports (AGENTS.md §3 rule 5).

This package holds only small, dependency-free utilities: GPU assertion, seeding,
logging setup, checkpoint I/O. Anything that knows about policies or games does
NOT belong here.
"""
