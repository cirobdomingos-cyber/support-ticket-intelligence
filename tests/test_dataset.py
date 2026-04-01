"""Tests for synthetic dataset generation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "4-support-ticket-api"))

import services


REQUIRED_KEYS = {
    "ticket_id", "product", "component", "failure_mode",
    "severity", "assigned_team", "status", "description",
    "creation_date", "creation_datetime",
}


def test_generate_ticket_has_required_keys():
    ticket = services._generate_ticket()
    missing = REQUIRED_KEYS - set(ticket.keys())
    assert not missing, f"Missing keys: {missing}"


def test_generate_ticket_assigned_team_not_empty():
    ticket = services._generate_ticket()
    assert ticket["assigned_team"], "assigned_team should not be empty"


def test_generate_ticket_severity_valid():
    valid = {"Low", "Medium", "High", "Critical"}
    for _ in range(20):
        ticket = services._generate_ticket()
        assert ticket["severity"] in valid


def test_generate_dataset_frame_shape():
    df = services._generate_dataset_frame(size=50)
    assert len(df) == 50
    assert "description" in df.columns
    assert "assigned_team" in df.columns


def test_normalize_dataset_columns_renames_aliases():
    import pandas as pd
    df = pd.DataFrame({"issue_description": ["test"], "route_team": ["Team A"]})
    result = services._normalize_dataset_columns(df)
    assert "description" in result.columns
    assert "assigned_team" in result.columns


def test_prepare_generated_dataset_for_training_maps_selected_columns():
    import pandas as pd

    df = pd.DataFrame(
        {
            "ticket_uuid": ["t-1"],
            "issue_description": ["test description"],
            "route_team": ["Team A"],
            "severity_level": ["High"],
        }
    )

    result = services._prepare_generated_dataset_for_training(
        df,
        services._get_internal_to_public_columns(),
        include_columns=["ticket_uuid", "issue_description", "route_team", "severity_level"],
        description_column="issue_description",
        assigned_team_column="route_team",
        ticket_id_column="ticket_uuid",
    )

    assert list(result.columns) == ["ticket_id", "description", "assigned_team", "severity_level"]


def test_prepare_generated_dataset_for_training_can_skip_team_and_autogenerate_ticket_id():
    import pandas as pd

    df = pd.DataFrame(
        {
            "issue_description": ["test description"],
            "severity_level": ["High"],
            "route_team": ["Team A"],
        }
    )

    result = services._prepare_generated_dataset_for_training(
        df,
        services._get_internal_to_public_columns(),
        include_columns=["issue_description", "severity_level"],
        description_column="issue_description",
        assigned_team_column="",
        ticket_id_column="",
    )

    assert "ticket_id" in result.columns
    assert "assigned_team" not in result.columns
    assert result.loc[0, "description"] == "test description"


def test_generate_ticket_closed_has_resolution_time():
    import random
    random.seed(42)
    found_closed = False
    for _ in range(100):
        ticket = services._generate_ticket()
        if ticket["status"] in {"Resolved", "Closed"}:
            found_closed = True
            assert ticket["time_to_close_seconds"] != "", "Closed ticket must have resolution time"
    assert found_closed, "Expected at least one closed ticket in 100 samples"
