# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from assume.scenario.oeds.infrastructure import InfrastructureInterface
from assume.scenario.oeds.static import fuel_translation, mastr_fuel_type


def test_mastr_fuel_type_mappings():
    assert mastr_fuel_type["lignite"] == "Braunkohle"
    assert mastr_fuel_type["hard coal"] == "Steinkohle"
    assert mastr_fuel_type["gas"] == "Erdgas"
    assert mastr_fuel_type["oil"] == "Mineralölprodukte"
    assert mastr_fuel_type["nuclear"] == "Kernenergie"

    for ger, eng in fuel_translation.items():
        if eng in mastr_fuel_type:
            assert mastr_fuel_type[eng] in fuel_translation


@pytest.fixture
def mock_infrastructure():
    with patch.object(InfrastructureInterface, "__init__", return_value=None):
        interface = InfrastructureInterface("test", "dummy_uri")
        interface.plz_nuts = pd.DataFrame(
            {"latitude": [50.0], "longitude": [6.0], "nuts3": ["DEA2D"]},
            index=[52353],
        )
        interface.databases = {"mastr": MagicMock(), "nuts": MagicMock()}
        interface.energietraeger_translated = mastr_fuel_type
        interface.mastr_generation_codes = {}
        return interface


def test_set_default_params(mock_infrastructure):
    df = pd.DataFrame(
        {
            "maxPower": [10000.0],
            "turbineTyp": ["Dampfturbine"],
            "startDate": [pd.to_datetime("2010-01-01")],
            "endDate": [pd.to_datetime("2040-01-01")],
            "generatorID": [1],
        }
    )
    result = mock_infrastructure.set_default_params(df)

    assert result["minPower"].iloc[0] == 5000.0
    assert result["ramp_up"].iloc[0] == 1000.0
    assert result["turbineTyp"].iloc[0] == "Dampfturbine"
    assert result["type"].iloc[0] == 2000


def test_get_power_plant_in_area_queries(mock_infrastructure):
    from datetime import datetime

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=pd.DataFrame()) as mock_read_sql:
        # Gas with created_before and stopped_after date filters
        mock_infrastructure.get_power_plant_in_area(
            area=52353,
            fuel_type="gas",
            created_before=datetime(2020, 1, 1),
            stopped_after=datetime(2023, 1, 1),
        )
        query_gas = mock_read_sql.call_args[0][0]
        assert "ev.\"Energietraeger\" = 'Erdgas'" in query_gas
        assert 'FROM "combustion_extended" ev' in query_gas
        assert "AND ev.\"Inbetriebnahmedatum\" < '2020-01-01T00:00:00'" in query_gas
        assert (
            'AND (ev."DatumEndgueltigeStilllegung" IS NULL OR ev."DatumEndgueltigeStilllegung"  > \'2023-01-01T00:00:00\')'
            in query_gas
        )

        # Lignite with string postal code
        mock_read_sql.reset_mock()
        mock_infrastructure.get_power_plant_in_area(area="52353", fuel_type="lignite")
        query_lignite = mock_read_sql.call_args[0][0]
        assert "ev.\"Energietraeger\" = 'Braunkohle'" in query_lignite

        # Hard Coal
        mock_read_sql.reset_mock()
        mock_infrastructure.get_power_plant_in_area(area=52353, fuel_type="hard coal")
        query_hard_coal = mock_read_sql.call_args[0][0]
        assert "ev.\"Energietraeger\" = 'Steinkohle'" in query_hard_coal

        # Oil
        mock_read_sql.reset_mock()
        mock_infrastructure.get_power_plant_in_area(area=52353, fuel_type="oil")
        query_oil = mock_read_sql.call_args[0][0]
        assert "ev.\"Energietraeger\" = 'Mineralölprodukte'" in query_oil

        # Nuclear
        mock_read_sql.reset_mock()
        mock_infrastructure.get_power_plant_in_area(area=52353, fuel_type="nuclear")
        query_nuclear = mock_read_sql.call_args[0][0]
        assert 'FROM "nuclear_extended" ev' in query_nuclear


def test_asset_queries_with_string_plz(mock_infrastructure):
    # Verify string postal codes work across all asset query methods without 'invalid plz code' exception
    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=pd.DataFrame()):
        assert mock_infrastructure.get_solar_systems_in_area(area="52353").empty
        assert mock_infrastructure.get_wind_turbines_in_area(area="52353").empty
        assert mock_infrastructure.get_biomass_systems_in_area(area="52353").empty
        assert mock_infrastructure.get_run_river_systems_in_area(area="52353").empty
        assert mock_infrastructure.get_water_storage_systems(area="52353") == []
        assert mock_infrastructure.get_solar_storage_systems_in_area(area="52353").empty


def test_operational_status_filter_in_queries(mock_infrastructure):
    # the removed "EinheitBetriebsstatus >= 35" filter must be replaced by a
    # string status filter so that "In Planung" units are not counted as
    # built capacity by callers that are not time-bounded
    import re

    from assume.scenario.oeds.infrastructure import MASTR_OPERATIONAL_STATUS

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    OPERATIONAL = f"IN {MASTR_OPERATIONAL_STATUS}"
    expected_clauses = {
        "solar": f'AND "EinheitBetriebsstatus" {OPERATIONAL}',
        "wind": f'AND "EinheitBetriebsstatus" {OPERATIONAL}',
        "biomass": f'AND "EinheitBetriebsstatus" {OPERATIONAL}',
        "hydro": f'AND "EinheitBetriebsstatus" {OPERATIONAL}',
        "solar_storage": f'AND so."EinheitBetriebsstatus" {OPERATIONAL}',
        "storage": "AND spe.\"EinheitBetriebsstatus\" = 'In Betrieb'",
        "power_plant": f'AND ev."EinheitBetriebsstatus" {OPERATIONAL}',
    }
    calls = [
        ("solar", lambda: mock_infrastructure.get_solar_systems_in_area(area=52353)),
        ("wind", lambda: mock_infrastructure.get_wind_turbines_in_area(area=52353)),
        (
            "biomass",
            lambda: mock_infrastructure.get_biomass_systems_in_area(area=52353),
        ),
        (
            "hydro",
            lambda: mock_infrastructure.get_run_river_systems_in_area(area=52353),
        ),
        (
            "solar_storage",
            lambda: mock_infrastructure.get_solar_storage_systems_in_area(area=52353),
        ),
        ("storage", lambda: mock_infrastructure.get_water_storage_systems(area=52353)),
        (
            "power_plant",
            lambda: mock_infrastructure.get_power_plant_in_area(
                area=52353, fuel_type="gas"
            ),
        ),
    ]
    for label, call in calls:
        with (
            patch("pandas.read_sql", return_value=pd.DataFrame()) as mr1,
            patch("pandas.read_sql_query", return_value=pd.DataFrame()) as mr2,
        ):
            call()
        query = (mr1.call_args or mr2.call_args)[0][0]
        assert expected_clauses[label] in query, (
            label,
            re.sub(r"\s+", " ", query)[-200:],
        )


def test_get_lat_lon_area(mock_infrastructure):
    # NUTS3 string area
    lat, lon = mock_infrastructure.get_lat_lon_area("DEA2D")
    assert (lat, lon) == (50.0, 6.0)

    # Integer postal code
    lat, lon = mock_infrastructure.get_lat_lon_area(52353)
    assert (lat, lon) == (50.0, 6.0)

    # String postal code
    lat, lon = mock_infrastructure.get_lat_lon_area("52353")
    assert (lat, lon) == (50.0, 6.0)
