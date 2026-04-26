"""
Spoiler guard — limits the campaign book text surfaced to the DM to only the
scenes the player has chronologically reached.

Public API
----------
revealed_text(scenes, progress) -> str
    Return the concatenated text of all scenes up to and including *progress*.
"""

from __future__ import annotations


def revealed_text(scenes: list[str], progress: int) -> str:
    """
    Return the campaign book text up to and including the scene at *progress*.

    Parameters
    ----------
    scenes:
        Ordered list of scene text blocks (each element is the full text of
        one scene, including its heading).  Produced by
        :func:`src.campaign.parser.parse_campaign`.
    progress:
        Zero-based index of the current (most-recently-reached) scene.
        Values below 0 are treated as 0.  Values beyond the last scene index
        are clamped so that all scenes are returned.

    Returns
    -------
    str
        Concatenation of ``scenes[0]`` through ``scenes[progress]`` separated
        by double newlines.  Returns an empty string when *scenes* is empty.

    Examples
    --------
    >>> revealed_text(["Scene A", "Scene B", "Scene C"], progress=1)
    'Scene A\\n\\nScene B'
    >>> revealed_text([], progress=5)
    ''
    """
    if not scenes:
        return ""
    idx = min(max(progress, 0), len(scenes) - 1)
    return "\n\n".join(scenes[: idx + 1])
