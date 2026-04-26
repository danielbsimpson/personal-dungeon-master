"""Tests for the campaign loader (Phase 3)."""

import pytest
from pathlib import Path

from src.campaign.loader import Campaign, load_campaigns


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_valid_campaign(base: Path, name: str) -> Path:
    """Create a minimal valid campaign folder under *base*."""
    folder = base / name
    folder.mkdir()
    (folder / "README.md").write_text(f"# {name}", encoding="utf-8")
    (folder / "character.md").write_text("## Hero\n", encoding="utf-8")
    (folder / "creature.md").write_text("## Beast\n", encoding="utf-8")
    (folder / f"{name}.txt").write_text("## SCENE 1 — Start\nScene text.\n", encoding="utf-8")
    return folder


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadCampaigns:
    def test_finds_example_campaign(self):
        """The real campaigns/ directory must contain the example campaign."""
        from src.config import settings

        campaigns = load_campaigns(settings.campaigns_dir)
        names = [c.name for c in campaigns]
        assert "example-campaign" in names

    def test_returns_campaign_dataclasses(self):
        from src.config import settings

        campaigns = load_campaigns(settings.campaigns_dir)
        assert all(isinstance(c, Campaign) for c in campaigns)

    def test_all_paths_exist(self):
        from src.config import settings

        campaigns = load_campaigns(settings.campaigns_dir)
        for c in campaigns:
            assert c.readme_path.exists(), f"{c.readme_path} missing"
            assert c.character_path.exists(), f"{c.character_path} missing"
            assert c.creature_path.exists(), f"{c.creature_path} missing"
            assert c.book_path.exists(), f"{c.book_path} missing"

    def test_campaign_name_matches_folder(self, tmp_path):
        _make_valid_campaign(tmp_path, "my-campaign")
        campaigns = load_campaigns(tmp_path)
        assert campaigns[0].name == "my-campaign"

    def test_invalid_campaign_excluded_when_valid_one_present(self, tmp_path):
        """A folder missing required files is silently excluded if others are valid."""
        _make_valid_campaign(tmp_path, "good-campaign")

        # Broken campaign — no character.md and no .txt file
        bad = tmp_path / "bad-campaign"
        bad.mkdir()
        (bad / "README.md").write_text("# Bad", encoding="utf-8")
        (bad / "creature.md").write_text("## Beast\n", encoding="utf-8")

        campaigns = load_campaigns(tmp_path)
        assert len(campaigns) == 1
        assert campaigns[0].name == "good-campaign"

    def test_empty_dir_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="No valid campaigns"):
            load_campaigns(tmp_path)

    def test_nonexistent_dir_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_campaigns(tmp_path / "does-not-exist")

    def test_all_broken_campaigns_error_includes_folder_name(self, tmp_path):
        broken = tmp_path / "broken-campaign"
        broken.mkdir()
        (broken / "README.md").write_text("# Broken", encoding="utf-8")
        # Missing character.md, creature.md, and .txt

        with pytest.raises(ValueError, match="broken-campaign"):
            load_campaigns(tmp_path)

    def test_multiple_campaigns_returned_sorted(self, tmp_path):
        _make_valid_campaign(tmp_path, "campaign-b")
        _make_valid_campaign(tmp_path, "campaign-a")
        campaigns = load_campaigns(tmp_path)
        assert [c.name for c in campaigns] == ["campaign-a", "campaign-b"]
