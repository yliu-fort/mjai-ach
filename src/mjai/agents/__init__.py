"""Agents layer (sibling of :mod:`mjai.games`).

Policy + value function implementations sharing a common abstract interface
(``agents.base.Policy``): :class:`TabularPolicy`, :class:`MLPSharedActorCritic`.
Checkpoint I/O (``agents.ckpt_io``) defines the canonical manifest + weight
format shared by training and the Play CLI.

May import only :mod:`mjai.utils`.
"""
