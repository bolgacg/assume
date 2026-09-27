# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from unittest.mock import MagicMock, patch

import numpy as np
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
        "solar": f'"EinheitBetriebsstatus" {OPERATIONAL}',
        "wind": f'"EinheitBetriebsstatus" {OPERATIONAL}',
        "biomass": f'"EinheitBetriebsstatus" {OPERATIONAL}',
        "hydro": f'"EinheitBetriebsstatus" {OPERATIONAL}',
        "solar_storage": f'so."EinheitBetriebsstatus" {OPERATIONAL}',
        "storage": "spe.\"EinheitBetriebsstatus\" = 'In Betrieb'",
        "power_plant": f'ev."EinheitBetriebsstatus" {OPERATIONAL}',
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


def test_area_none_queries_without_postal_code_filter(mock_infrastructure):
    # area=None selects all units, so no postal code filter may be applied;
    # a postal code area keeps the filter
    postal_code, lat, lon = mock_infrastructure.resolve_area(None)
    assert postal_code is None
    assert (lat, lon) == (50.0, 6.0)

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    getters = [
        mock_infrastructure.get_solar_systems_in_area,
        mock_infrastructure.get_wind_turbines_in_area,
        mock_infrastructure.get_biomass_systems_in_area,
        mock_infrastructure.get_run_river_systems_in_area,
        mock_infrastructure.get_solar_storage_systems_in_area,
        mock_infrastructure.get_water_storage_systems,
        mock_infrastructure.get_power_plant_in_area,
    ]
    for getter in getters:
        for area, filtered in [(None, False), (52353, True)]:
            with (
                patch("pandas.read_sql", return_value=pd.DataFrame()) as mr1,
                patch("pandas.read_sql_query", return_value=pd.DataFrame()) as mr2,
            ):
                getter(area=area)
            query = (mr1.call_args or mr2.call_args)[0][0]
            assert ("Postleitzahl\" in ('52353')" in query) == filtered, (
                getter.__name__,
                area,
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


def test_get_solar_systems_in_area_power_cap_threshold(mock_infrastructure):
    # A system with 120 kWp installed capacity and a 70% feed-in limit
    # keeps its installed capacity in maxPower (the limit is applied as a
    # peak clip in get_solar_series) and must still qualify for direct
    # marketing (installed 120 kWp > 100 kWp threshold).
    raw_df = pd.DataFrame(
        {
            "unitID": ["SEE12345"],
            "maxPower": [120.0],
            "lon": [6.0],
            "lat": [50.0],
            "plzCode": ["52353"],
            "azimuthCode": ["Süd"],
            "limited": ["Ja, auf 70%"],
            "ownConsumption": [0],
            "tiltCode": ["21 - 40 Grad"],
            "startDate": [pd.to_datetime("2018-01-01")],
            "eeg": [None],
        }
    )

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=raw_df):
        result = mock_infrastructure.get_solar_systems_in_area(area=52353)

        # maxPower keeps the installed (pre-limit) capacity
        assert result["maxPower"].iloc[0] == pytest.approx(120.0)
        # demandP reflects the clipped peak: 120 * 0.7 * 1000 = 84000.0 W
        assert result["demandP"].iloc[0] == pytest.approx(84000.0)
        # limit_factor is carried for get_solar_series
        assert result["limit_factor"].iloc[0] == 0.7
        # eeg should be set to 0 (direct marketing) because installed capacity was 120 > 100
        assert result["eeg"].iloc[0] == 0


def test_get_solar_systems_in_area_ost_west_split(mock_infrastructure):
    # An Ost-West double-row unit is split into a south-independent east
    # (90) and west (270) orientation at half capacity each, so total
    # capacity is preserved and the groupby in get_solar_series produces
    # two separate orientations.
    raw_df = pd.DataFrame(
        {
            "unitID": ["SEE10001", "SEE10002"],
            "maxPower": [30.0, 30.0],
            "lon": [6.0, 6.0],
            "lat": [50.0, 50.0],
            "plzCode": ["52353", "52353"],
            "azimuthCode": ["Ost-West", "Süd"],
            "limited": ["Nein", "Nein"],
            "ownConsumption": [0, 0],
            "tiltCode": ["21 - 40 Grad", "21 - 40 Grad"],
            "startDate": [pd.to_datetime("2019-01-01"), pd.to_datetime("2019-01-01")],
            "eeg": [None, 0],
        }
    )

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=raw_df):
        result = mock_infrastructure.get_solar_systems_in_area(area=52353)

        # total capacity unchanged
        assert result["maxPower"].sum() == pytest.approx(60.0)
        # Ost-West unit produced one east and one west row at half capacity
        ew = result[(result["azimuth"] == "90") | (result["azimuth"] == "270")]
        assert ew["maxPower"].tolist() == [15.0, 15.0]
        assert result[result["azimuth"] == "180"]["maxPower"].iloc[0] == 30.0


def test_map_mastr_orientation_codes(caplog):
    from assume.scenario.oeds.infrastructure import map_mastr_codes
    from assume.scenario.oeds.static import mastr_solar_azimuth, mastr_solar_tilt

    tilt_codes = pd.Series(
        [
            "Nachgeführt",
            "5 - 20 Grad",
            "unter 5 Grad (horizontal)",
            # MaStR carries a trailing space on this code
            "90 Grad (vertikal) ",
            None,
            "unbekannt",
        ]
    )
    with caplog.at_level("WARNING"):
        tilt = map_mastr_codes(tilt_codes, mastr_solar_tilt, "30")
    # tracked units keep the default tilt instead of the 180° azimuth value
    assert tilt.tolist() == ["30", "13", "3", "90", "30", "30"]
    # missing codes are silent, unknown codes are reported
    assert "unbekannt" in caplog.text

    azimuth_codes = pd.Series(["nachgeführt", "Süd-West", None])
    azimuth = map_mastr_codes(azimuth_codes, mastr_solar_azimuth, "180")
    assert azimuth.tolist() == ["180", "225", "180"]


def test_get_solar_systems_in_area_ost_west_split_after_size_rules(
    mock_infrastructure,
):
    # a 50 kWp Ost-West unit must get the 70% feed-in limit for units
    # above 30 kWp before it is split into two 25 kWp rows
    raw_df = pd.DataFrame(
        {
            "unitID": ["SEE10001"],
            "maxPower": [50.0],
            "lon": [6.0],
            "lat": [50.0],
            "plzCode": ["52353"],
            "azimuthCode": ["Ost-West"],
            "limited": [None],
            "ownConsumption": [None],
            "tiltCode": [None],
            "startDate": [pd.to_datetime("2019-01-01")],
            "eeg": [None],
        }
    )

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=raw_df):
        result = mock_infrastructure.get_solar_systems_in_area(area=52353)

    assert result["maxPower"].tolist() == [25.0, 25.0]
    assert result["limit_factor"].tolist() == [0.7, 0.7]
    assert result["tilt"].tolist() == ["30", "30"]


def test_get_solar_storage_ost_west_split_keeps_battery_totals(mock_infrastructure):
    # the Ost-West split must not double the battery of a PV+battery unit,
    # get_solar_series sums batPower per orientation group
    raw_df = pd.DataFrame(
        {
            "unitID": ["SEL10001"],
            "unitMastrID": ["SSE10001"],
            "maxPower": [10.0],
            "batPower": [5.0],
            "lon": [6.0],
            "lat": [50.0],
            "plzCode": ["52353"],
            "azimuthCode": ["Ost-West"],
            "limited": ["Nein"],
            "ownConsumption": ["Teileinspeisung (einschließlich Eigenverbrauch)"],
            "tiltCode": ["21 - 40 Grad"],
            "startDate": ["2020-01-01"],
            "VMax": [8.0],
        }
    )

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=raw_df):
        result = mock_infrastructure.get_solar_storage_systems_in_area(area=52353)

    assert sorted(result["azimuth"]) == ["270", "90"]
    assert result["maxPower"].sum() == pytest.approx(10.0)
    assert result["batPower"].sum() == pytest.approx(5.0)
    assert result["VMax"].sum() == pytest.approx(8.0)


def test_get_solar_series_applies_feed_in_limit_as_peak_clip():
    # A limited system must keep its rated capacity in maxPower but its
    # series must be clipped at limit_factor * rated, not scaled by it.
    from assume.scenario.oeds.infrastructure import get_solar_series

    index = pd.date_range("2020-06-21", periods=3, freq="h", tz="Europe/Berlin")
    weather_df = pd.DataFrame(
        {
            "zenith": [93.0, 10.0, 93.0],
            "azimuth": [180.0, 180.0, 180.0],
            "dni": [0.0, 900.0, 0.0],
            "ghi": [50.0, 450.0, 50.0],
            "dhi": np.zeros(3),
        },
        index=index,
    )

    rated = 100.0
    unlimited = pd.DataFrame({"azimuth": ["180"], "tilt": ["30"], "maxPower": [rated]})
    solar_open, _ = get_solar_series(unlimited, weather_df, inverter_efficiency=1.0)
    open_kw = solar_open.to_numpy()
    assert open_kw.max() > 0

    # MaStR semantics: output is clipped at limit_factor * rated power
    factor = 0.8
    expected = np.minimum(open_kw, factor * rated)

    systems = pd.DataFrame(
        {
            "azimuth": ["180"],
            "tilt": ["30"],
            "maxPower": [rated],
            "limit_factor": [factor],
        }
    )
    solar, _ = get_solar_series(systems, weather_df, inverter_efficiency=1.0)
    clipped_kw = solar.to_numpy()
    # clipped, never scaled: the shape below the clip level is unchanged,
    # the peak sits exactly at the clip level instead of the natural peak
    assert not np.allclose(clipped_kw, open_kw * factor)
    assert np.allclose(clipped_kw, expected)
    assert np.allclose(clipped_kw.max(), factor * rated)


def test_clipped_feed_in_matches_per_unit_clip():
    from assume.scenario.oeds.infrastructure import clipped_feed_in

    rng = np.random.default_rng(0)
    installed = rng.uniform(1, 100, 50)
    installed[3] = 0.0  # units without capacity are ignored
    cap = installed * rng.uniform(0.4, 1.0, 50)
    share = np.array([0.0, 0.3, 0.6, 0.85, 1.0, 1.1])

    expected = np.minimum(np.outer(share, installed), cap).sum(axis=1)
    assert np.allclose(clipped_feed_in(share, installed, cap), expected)


def test_get_solar_series_clips_units_at_ac_capacity():
    # maxPower is the installed DC capacity; units with a smaller inverter
    # (acPower) or a feed-in limit are clipped individually, even if they
    # share an orientation group
    from assume.scenario.oeds.infrastructure import get_solar_series

    index = pd.date_range("2020-06-21", periods=3, freq="h", tz="Europe/Berlin")
    weather_df = pd.DataFrame(
        {
            "zenith": [93.0, 10.0, 93.0],
            "azimuth": [180.0, 180.0, 180.0],
            "dni": [0.0, 900.0, 0.0],
            "ghi": [50.0, 450.0, 50.0],
            "dhi": np.zeros(3),
        },
        index=index,
    )
    unit = pd.DataFrame({"azimuth": ["180"], "tilt": ["30"], "maxPower": [100.0]})
    open_kw = get_solar_series(unit, weather_df)[0].to_numpy()

    systems = pd.DataFrame(
        {
            "azimuth": ["180", "180"],
            "tilt": ["30", "30"],
            "maxPower": [100.0, 100.0],
            "acPower": [60.0, 100.0],
            "limit_factor": [1.0, 0.7],
        }
    )
    solar, _ = get_solar_series(systems, weather_df)
    expected = np.minimum(open_kw, 60.0) + np.minimum(open_kw, 70.0)
    assert open_kw.max() > 70.0
    assert np.allclose(solar.to_numpy(), expected)


def test_get_solar_series_applies_losses_and_aging_before_clip():
    # DC output is reduced by the inverter efficiency and by linear aging
    # at the middle of the simulated period, the AC capacity clips afterwards
    from assume.scenario.oeds.infrastructure import get_solar_series

    index = pd.date_range("2020-06-21 11:00", periods=3, freq="h")
    weather_df = pd.DataFrame(
        {
            "zenith": [93.0, 10.0, 93.0],
            "azimuth": [180.0, 180.0, 180.0],
            "dni": [0.0, 900.0, 0.0],
            "ghi": [50.0, 450.0, 50.0],
            "dhi": np.zeros(3),
        },
        index=index,
    )
    unit = pd.DataFrame({"azimuth": ["180"], "tilt": ["30"], "maxPower": [100.0]})
    lossless = get_solar_series(unit, weather_df, inverter_efficiency=1.0)[0]

    # commissioned ten years before the middle of the period
    aged = unit.assign(startDate=pd.Timestamp("2010-06-21 12:00"))
    solar = get_solar_series(aged, weather_df, inverter_efficiency=0.9)[0]
    expected_factor = 0.9 * (1 - 0.005 * 3653 / 365.25)
    assert np.allclose(solar, lossless * expected_factor)

    # units commissioned after the period do not gain from negative aging
    new = unit.assign(startDate=pd.Timestamp("2021-01-01"))
    solar = get_solar_series(new, weather_df, inverter_efficiency=0.9)[0]
    assert np.allclose(solar, lossless * 0.9)

    # the AC capacity clips the output after losses
    capped = unit.assign(acPower=60.0)
    solar = get_solar_series(capped, weather_df, inverter_efficiency=0.9)[0]
    assert np.allclose(solar, np.minimum(lossless * 0.9, 60.0))
    assert (lossless * 0.9).max() > 60.0


def test_wind_hub_height_and_diameter_are_filled_in_series(mock_infrastructure):
    # the query returns the raw MaStR values; missing hub heights and rotor
    # diameters are only filled for the simulation
    from assume.scenario.oeds.infrastructure import get_wind_series

    raw_df = pd.DataFrame(
        {
            "unitID": ["SEE1", "SEE2"],
            "maxPower": [2000.0, 2000.0],
            "lon": [6.0, 6.0],
            "lat": [50.0, 50.0],
            "plzCode": ["52353", "52353"],
            "height": [100.0, None],
            "diameter": [None, 90.0],
            "generatorID": [None, None],
        }
    )

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql_query", return_value=raw_df):
        turbines = mock_infrastructure.get_wind_turbines_in_area(area=52353)
    assert turbines["height"].isna().tolist() == [False, True]
    assert turbines["diameter"].isna().tolist() == [True, False]

    index = pd.date_range("2020-01-01", periods=3, freq="h")
    weather_df = pd.DataFrame(
        {"temp_air": [280.0] * 3, "wind_speed": [3.0, 8.0, 12.0]}, index=index
    )
    filled = turbines.fillna({"height": 100.0, "diameter": 90.0})
    assert np.allclose(
        get_wind_series(turbines, weather_df), get_wind_series(filled, weather_df)
    )
    assert get_wind_series(turbines, weather_df).max() > 0
