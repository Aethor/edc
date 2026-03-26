import pytest
from dataset2edc import format_quad


@pytest.mark.parametrize(
    "slot,clean_slot",
    [
        (
            "yago:V_U002E_J_U002E__Holmes_Q100804978",
            "V.J. Holmes",
        ),
        (
            "yago:Ncaa_Division_I_Men_U0027_S_Basketball_Q94861615",
            "Ncaa Division I Men\\'S Basketball",
        ),
        ("yago:BBM__u0028_software_u0029_", "BBM (software)"),
        ("yago:Richard_J_U002E__Malak_Q107365243", "Richard J. Malak"),
        (
            "yago:Centre_D_U0027_Études_Et_De_Recherche_En_Droit_De_L_U0027_Immatériel_Q51785320",
            "Centre D\\'Études Et De Recherche En Droit De L\\'Immatériel",
        ),
    ],
)
def test_format_quad_matrix(slot: str, clean_slot: str):
    assert format_quad((slot, "rel", "obj", "ts")) == (clean_slot, "rel", "obj", "ts")
