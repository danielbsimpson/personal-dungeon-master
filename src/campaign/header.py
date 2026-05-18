"""
Contextual chunk headers for campaign chunks.

Prepends a short metadata header to each CampaignChunk before embedding.
The header is baked into the embedding so retrieval naturally filters by
narrative location, not just semantic content.

The text injected into the LLM prompt remains the raw chunk only — avoiding
redundant metadata in the generated response.
"""

from __future__ import annotations

from src.campaign.chunker import CampaignChunk


HEADER_TEMPLATE = "[{act} | {scene}]\n"


def with_header(chunk: CampaignChunk, campaign_name: str = "") -> str:
    """
    Return the chunk text prefixed with a context header.

    This string is what gets embedded — NOT what gets injected into
    the LLM prompt.

    Args:
        chunk: A CampaignChunk produced by semantic_chunk.
        campaign_name: Optional campaign name for future multi-campaign use.

    Returns:
        Header string concatenated with the raw chunk text.
    """
    header = HEADER_TEMPLATE.format(act=chunk.act, scene=chunk.scene_header)
    return header + chunk.text


def header_string(chunk: CampaignChunk) -> str:
    """Return just the header, for display or debugging."""
    return HEADER_TEMPLATE.format(act=chunk.act, scene=chunk.scene_header)
