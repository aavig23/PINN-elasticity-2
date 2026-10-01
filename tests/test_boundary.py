from pinn_elasticity.physics.boundary import (
    BOUNDARY_CONDITIONS,
    INITIAL_CONDITIONS,
    LOAD,
    ZERO,
    conditions_for_edge,
)


def test_boundary_table_matches_spec_section_2_4():
    expected = {
        ("left", "u", ZERO),
        ("left", "w", ZERO),
        ("bottom", "szz", ZERO),
        ("bottom", "sxz", ZERO),
        ("top", "szz", ZERO),
        ("top", "sxz", ZERO),
        ("right", "sxx", LOAD),
        ("right", "sxz", ZERO),
    }
    assert {(c.edge, c.quantity, c.target) for c in BOUNDARY_CONDITIONS} == expected
    assert len(BOUNDARY_CONDITIONS) == len(expected)


def test_initial_table_matches_spec_section_2_6():
    assert {(c.quantity, c.target) for c in INITIAL_CONDITIONS} == {
        ("u", ZERO),
        ("w", ZERO),
        ("u_t", ZERO),
        ("w_t", ZERO),
    }


def test_loss_term_mapping_matches_spec_section_5_5():
    terms = {c.term for c in BOUNDARY_CONDITIONS}
    assert terms == {"clamp_u", "clamp_w", "free_zz", "free_xz", "load_xx", "load_xz"}
    # Top and bottom feed the same traction-free terms.
    assert {c.term for c in conditions_for_edge("top")} == {c.term for c in conditions_for_edge("bottom")}
