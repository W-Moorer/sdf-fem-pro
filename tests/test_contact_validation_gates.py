from __future__ import annotations

from sfc.contact.validation_gates import contact_path_tracking_gate_metrics, contact_total_priority_gate_metrics


def _valid_total_row() -> dict[str, object]:
    return {
        "comparison_stage": "region_totals_before_nodal_cpress",
        "nodal_cpress_deferred": 1,
        "normal_force": 12.0,
        "contact_region_normal_force": 12.0,
        "contact_region_virtual_work": 0.2,
        "contact_region_energy": 0.1,
        "contact_active_area": 3.0,
        "active_contact_region_count": 4,
        "contact_path_cache_hit_fraction": 1.0,
        "contact_path_cache_match_fraction": 1.0,
        "contact_master_face_switch_fraction": 0.0,
        "contact_master_face_topological_continuity_fraction": 1.0,
        "contact_master_face_invalid_jump_fraction": 0.0,
        "contact_master_barycentric_drift_max": 0.0,
        "contact_secondary_pressure_recovery_source": "constraint_region",
        "contact_pressure_recovery_source": "constraint_region",
        "contact_legacy_pressure_alias_source": "secondary_constraint_region",
        "contact_legacy_pressure_alias_matches_secondary": 1,
    }


def test_contact_total_gate_accepts_region_totals_before_nodal_clouds() -> None:
    gate = contact_total_priority_gate_metrics([_valid_total_row()])

    assert int(gate["contact_total_gate_passed"]) == 1
    assert int(gate["contact_total_active_contact_present"]) == 1
    assert int(gate["contact_total_nodal_cpress_deferred"]) == 1
    assert int(gate["contact_total_no_nodal_priority_columns"]) == 1
    assert int(gate["contact_total_secondary_pressure_recovery_from_region"]) == 1
    assert int(gate["contact_total_legacy_pressure_alias_from_secondary"]) == 1
    assert int(gate["contact_total_force_consistency_passed"]) == 1


def test_contact_total_gate_rejects_nodal_pressure_priority_columns() -> None:
    row = _valid_total_row()
    row["max_contact_secondary_pressure_nodeavg"] = 99.0

    gate = contact_total_priority_gate_metrics([row])

    assert int(gate["contact_total_gate_passed"]) == 0
    assert int(gate["contact_total_no_nodal_priority_columns"]) == 0


def test_contact_total_gate_rejects_missing_path_tracking_or_sample_pressure_source() -> None:
    missing_path = _valid_total_row()
    missing_path.pop("contact_path_cache_hit_fraction")
    missing_gate = contact_total_priority_gate_metrics([missing_path])

    assert int(missing_gate["contact_total_gate_passed"]) == 0
    assert int(missing_gate["contact_total_path_columns_present"]) == 0

    wrong_source = _valid_total_row()
    wrong_source["contact_secondary_pressure_recovery_source"] = "slave_sample_alias"
    source_gate = contact_total_priority_gate_metrics([wrong_source])

    assert int(source_gate["contact_total_gate_passed"]) == 0
    assert int(source_gate["contact_total_secondary_pressure_recovery_from_region"]) == 0


def test_contact_total_gate_rejects_legacy_pressure_alias_not_from_secondary_region() -> None:
    wrong_alias = _valid_total_row()
    wrong_alias["contact_legacy_pressure_alias_source"] = "sample_scatter"

    alias_gate = contact_total_priority_gate_metrics([wrong_alias])

    assert int(alias_gate["contact_total_gate_passed"]) == 0
    assert int(alias_gate["contact_total_legacy_pressure_alias_from_secondary"]) == 0

    mismatch = _valid_total_row()
    mismatch["contact_legacy_pressure_alias_matches_secondary"] = 0
    mismatch_gate = contact_total_priority_gate_metrics([mismatch])

    assert int(mismatch_gate["contact_total_gate_passed"]) == 0
    assert int(mismatch_gate["contact_total_legacy_pressure_alias_from_secondary"]) == 0


def test_contact_total_gate_rejects_force_mismatch_between_response_and_region_totals() -> None:
    row = _valid_total_row()
    row["normal_force"] = 9.0

    gate = contact_total_priority_gate_metrics([row])

    assert int(gate["contact_total_gate_passed"]) == 0
    assert int(gate["contact_total_force_consistency_passed"]) == 0
    assert float(gate["contact_total_force_relative_mismatch_max"]) > 0.0


def _valid_path_row(**overrides: object) -> dict[str, object]:
    row = {
        "active_contact_region_count": 3,
        "contact_active_area": 2.0,
        "contact_region_normal_force": 5.0,
        "contact_path_cache_hit_fraction": 1.0,
        "contact_path_cache_match_fraction": 1.0,
        "contact_master_face_switch_fraction": 0.0,
        "contact_master_face_topological_continuity_fraction": 1.0,
        "contact_master_face_invalid_jump_fraction": 0.0,
        "contact_active_region_jaccard": 1.0,
        "contact_active_region_persistence_fraction": 1.0,
        "contact_master_barycentric_drift_max": 0.0,
    }
    row.update(overrides)
    return row


def test_path_tracking_gate_accepts_coherent_accepted_region_history() -> None:
    gate = contact_path_tracking_gate_metrics([_valid_path_row(), _valid_path_row()])

    assert int(gate["path_tracking_gate_passed"]) == 1
    assert int(gate["path_tracking_continuity_observable"]) == 1
    assert int(gate["path_tracking_required_columns_present"]) == 1
    assert int(gate["path_tracking_invalid_jump_gate_passed"]) == 1


def test_path_tracking_gate_rejects_single_frame_or_random_master_face_jumps() -> None:
    single = contact_path_tracking_gate_metrics([_valid_path_row()])

    assert int(single["path_tracking_gate_passed"]) == 0
    assert int(single["path_tracking_continuity_observable"]) == 0

    jump = contact_path_tracking_gate_metrics(
        [
            _valid_path_row(),
            _valid_path_row(
                contact_master_face_switch_fraction=1.0,
                contact_master_face_topological_continuity_fraction=0.0,
                contact_master_face_invalid_jump_fraction=1.0,
            ),
        ]
    )

    assert int(jump["path_tracking_gate_passed"]) == 0
    assert int(jump["path_tracking_topological_continuity_gate_passed"]) == 0
    assert int(jump["path_tracking_invalid_jump_gate_passed"]) == 0


def test_path_tracking_gate_rejects_missing_active_region_continuity_columns() -> None:
    row = _valid_path_row()
    row.pop("contact_active_region_jaccard")

    gate = contact_path_tracking_gate_metrics([_valid_path_row(), row])

    assert int(gate["path_tracking_gate_passed"]) == 0
    assert int(gate["path_tracking_required_columns_present"]) == 0
