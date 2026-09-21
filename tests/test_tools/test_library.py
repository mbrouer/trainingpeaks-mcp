"""Tests for workout library tools."""

from unittest.mock import AsyncMock, patch

import pytest

from tp_mcp.client.context import athlete_override
from tp_mcp.client.http import APIResponse
from tp_mcp.tools.library import (
    tp_create_library,
    tp_create_library_item,
    tp_delete_library,
    tp_get_libraries,
    tp_get_library_items,
    tp_schedule_library_workout,
    tp_update_library_item,
)


class TestGetLibraries:
    @pytest.mark.asyncio
    async def test_list_libraries(self):
        data = [
            {"exerciseLibraryId": 1, "libraryName": "My Workouts", "isDefaultContent": False,
             "ownerName": "Athlete", "ownerId": 7, "itemCount": 5},
            {"exerciseLibraryId": 2, "libraryName": "Default", "isDefaultContent": True,
             "ownerName": "Joe Friel", "ownerId": 7},
        ]
        response = APIResponse(success=True, data=data)
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_get_libraries()

        assert result["count"] == 2
        assert result["libraries"][0]["name"] == "My Workouts"
        assert result["libraries"][0]["owner_name"] == "Athlete"
        assert result["libraries"][0]["owner_id"] == 7
        assert result["libraries"][0]["item_count"] == 5
        assert result["libraries"][1]["is_default"] is True


class TestGetLibraryItems:
    @pytest.mark.asyncio
    async def test_list_items(self):
        data = [
            {
                "exerciseLibraryItemId": 10,
                "itemName": "Sweet Spot",
                "workoutTypeId": 2,
                "totalTimePlanned": 1.5,
                "tssPlanned": 80,
            },
        ]
        response = APIResponse(success=True, data=data)
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_get_library_items("1")

        assert result["count"] == 1
        assert result["items"][0]["name"] == "Sweet Spot"
        assert result["items"][0]["sport"] == 2


class TestCreateLibrary:
    @pytest.mark.asyncio
    async def test_create_sends_name(self):
        response = APIResponse(success=True, data={"exerciseLibraryId": 3})
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance._get_user_data = AsyncMock(return_value={"personId": 999})
            mock_instance.post = AsyncMock(return_value=response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_create_library("Race Prep")

        assert result["success"] is True
        assert result["library_id"] == 3
        payload = mock_instance.post.call_args[1]["json"]
        assert payload["libraryName"] == "Race Prep"
        assert payload["ownerId"] == 999


class TestDeleteLibrary:
    @pytest.mark.asyncio
    async def test_delete(self):
        response = APIResponse(success=True, data=None)
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.delete = AsyncMock(return_value=response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_delete_library("1")

        assert result["success"] is True


class TestCreateLibraryItem:
    @pytest.mark.asyncio
    async def test_create_with_structure_nested_object(self):
        """Library item structure should be nested object, not string."""
        structure = {"structure": [{"type": "step"}]}
        response = APIResponse(success=True, data={"exerciseLibraryItemId": 20})
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.post = AsyncMock(return_value=response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_create_library_item(
                library_id="1", name="Tempo",
                sport_family_id=2, sport_type_id=3,
                structure=structure,
            )

        assert result["success"] is True
        payload = mock_instance.post.call_args[1]["json"]
        # Structure should be nested object, NOT JSON string
        assert isinstance(payload["structure"], dict)
        # Library API expects workoutTypeId/workoutSubTypeId; the fitness-API
        # field names silently create items with sport 0 (no power targets)
        assert payload["workoutTypeId"] == 2
        assert payload["workoutSubTypeId"] == 3
        assert "workoutTypeFamilyId" not in payload
        assert "workoutTypeValueId" not in payload

    @pytest.mark.asyncio
    async def test_create_backfills_polyline_and_range(self):
        """A native structure without preview fields gets polyline +
        primaryIntensityTargetOrRange so TP renders the thumbnail."""
        def _block(begin, end, dur, lo, hi, cls):
            return {
                "type": "step", "length": {"value": 1, "unit": "repetition"},
                "begin": begin, "end": end,
                "steps": [{
                    "name": cls, "length": {"value": dur, "unit": "second"},
                    "targets": [{"minValue": lo, "maxValue": hi}],
                    "intensityClass": cls,
                }],
            }
        structure = {
            "primaryIntensityMetric": "percentOfFtp",
            "primaryLengthMetric": "duration",
            "structure": [
                _block(0, 300, 300, 50, 60, "warmUp"),
                _block(300, 3300, 3000, 65, 72, "active"),
                _block(3300, 3600, 300, 50, 55, "coolDown"),
            ],
        }
        response = APIResponse(success=True, data={"exerciseLibraryItemId": 21})
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.post = AsyncMock(return_value=response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_create_library_item(
                library_id="1", name="Endurance",
                sport_family_id=2, sport_type_id=2, structure=structure,
            )

        assert result["success"] is True
        st = mock_instance.post.call_args[1]["json"]["structure"]
        assert st["primaryIntensityTargetOrRange"] == "range"
        # 3 single-step blocks → 3 bars × 4 points
        assert len(st["polyline"]) == 12
        # normalised so the structure's peak target (active 72) = 1.0;
        # warm-up 60 → 60/72 = 0.8333. Nothing exceeds 1.0.
        assert [0.0833, 1.0] in st["polyline"]
        assert [0.0, 0.8333] in st["polyline"]
        assert all(p[1] <= 1.0 for p in st["polyline"])


class TestStructurePreviewHelper:
    def test_polyline_expands_repetition_and_normalises(self):
        from tp_mcp.tools.library import _compute_native_polyline
        blocks = [
            {"type": "step", "length": {"value": 1, "unit": "repetition"},
             "steps": [{"length": {"value": 2000, "unit": "meter"},
                        "targets": [{"minValue": 70, "maxValue": 80}]}]},
            {"type": "repetition", "length": {"value": 6, "unit": "repetition"},
             "steps": [
                 {"length": {"value": 800, "unit": "meter"},
                  "targets": [{"minValue": 102, "maxValue": 104}]},
                 {"length": {"value": 400, "unit": "meter"},
                  "targets": [{"minValue": 70, "maxValue": 75}]},
             ]},
        ]
        poly = _compute_native_polyline(blocks)
        # warmup bar + 6×(work+rest) bars = 13 bars × 4 points
        assert len(poly) == 13 * 4
        # normalised: the peak target (work 104) = 1.0 and nothing exceeds it
        assert any(pt[1] == 1.0 for pt in poly)
        assert all(pt[1] <= 1.0 for pt in poly)

    def test_polyline_absolute_targets_stay_in_unit_range(self):
        """Absolute watts must normalise to [0,1], not the old /100 (300 W → 3.0)."""
        from tp_mcp.tools.library import _compute_native_polyline
        blocks = [
            {"type": "step", "length": {"value": 1, "unit": "repetition"},
             "steps": [{"length": {"value": 600, "unit": "second"},
                        "targets": [{"minValue": 150, "maxValue": 150}]}]},
            {"type": "step", "length": {"value": 1, "unit": "repetition"},
             "steps": [{"length": {"value": 300, "unit": "second"},
                        "targets": [{"minValue": 300, "maxValue": 300}]}]},
        ]
        poly = _compute_native_polyline(blocks)
        assert max(p[1] for p in poly) == 1.0    # 300 W peak → 1.0, not 3.0
        assert any(p[1] == 0.5 for p in poly)     # 150 W → 150/300

    def test_polyline_uses_minvalue_when_no_max(self):
        """A floor-only target (`{"minValue": 55}`) is a real bar, not height 0."""
        from tp_mcp.tools.library import _compute_native_polyline
        blocks = [
            {"type": "step", "length": {"value": 1, "unit": "repetition"},
             "steps": [{"length": {"value": 300, "unit": "second"},
                        "targets": [{"minValue": 55}]}]},
            {"type": "step", "length": {"value": 1, "unit": "repetition"},
             "steps": [{"length": {"value": 300, "unit": "second"},
                        "targets": [{"minValue": 100, "maxValue": 100}]}]},
        ]
        poly = _compute_native_polyline(blocks)
        assert any(p[1] == 0.55 for p in poly)

    def test_ensure_preview_noop_on_non_native(self):
        from tp_mcp.tools.library import _ensure_structure_preview
        assert _ensure_structure_preview(None) is None
        assert _ensure_structure_preview({"steps": []}) == {"steps": []}

    def test_ensure_preview_does_not_mutate_caller(self):
        """The helper returns a copy — the caller's structure dict is untouched."""
        from tp_mcp.tools.library import _ensure_structure_preview
        src = {"primaryIntensityMetric": "percentOfFtp",
               "structure": [{"type": "step", "length": {"value": 1, "unit": "repetition"},
                              "steps": [{"length": {"value": 300, "unit": "second"},
                                         "targets": [{"minValue": 90, "maxValue": 90}]}]}]}
        out = _ensure_structure_preview(src)
        assert "polyline" in out
        assert "polyline" not in src            # caller NOT mutated
        assert out is not src


class TestUpdateLibraryItem:
    @pytest.mark.asyncio
    async def test_sets_workout_type_on_sportless_template(self):
        """A template saved with workoutTypeId=0 gets its sport set, and the
        merged value reaches the PUT payload."""
        existing = {"exerciseLibraryItemId": 20, "itemName": "Tempo", "workoutTypeId": 0}
        get_resp = APIResponse(success=True, data=[existing])
        put_resp = APIResponse(success=True, data=None)
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=get_resp)
            mock_instance.put = AsyncMock(return_value=put_resp)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_update_library_item(
                library_id="1", item_id="20",
                workout_type_id=2, workout_sub_type_id=6,
            )

        assert result["success"] is True
        payload = mock_instance.put.call_args[1]["json"]
        assert payload["workoutTypeId"] == 2
        assert payload["workoutSubTypeId"] == 6

    @pytest.mark.asyncio
    async def test_workout_type_untouched_when_not_passed(self):
        """Omitting the sport params leaves the existing workoutTypeId alone and
        adds no workoutSubTypeId key."""
        existing = {"exerciseLibraryItemId": 21, "itemName": "Endurance", "workoutTypeId": 2}
        get_resp = APIResponse(success=True, data=[existing])
        put_resp = APIResponse(success=True, data=None)
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=get_resp)
            mock_instance.put = AsyncMock(return_value=put_resp)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_update_library_item(
                library_id="1", item_id="21", name="Endurance v2",
            )

        assert result["success"] is True
        payload = mock_instance.put.call_args[1]["json"]
        assert payload["workoutTypeId"] == 2          # unchanged
        assert "workoutSubTypeId" not in payload
        assert payload["itemName"] == "Endurance v2"  # the field we did change


class TestScheduleLibraryWorkout:
    TEMPLATE = {
        "exerciseLibraryItemId": 10,
        "itemName": "Sweet Spot",
        "workoutTypeId": 2,
        "workoutSubTypeId": 3,
        "totalTimePlanned": 1.5,
        "tssPlanned": 80.0,
        "ifPlanned": 0.85,
        "description": "2x15 @ 88-93%",
        "structure": {"structure": [{"type": "step"}]},
    }

    @pytest.mark.asyncio
    async def test_schedule_copies_template_to_workout(self):
        items_response = APIResponse(success=True, data=[self.TEMPLATE])
        create_response = APIResponse(success=True, data={"workoutId": 999})
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=items_response)
            mock_instance.post = AsyncMock(return_value=create_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_schedule_library_workout("1", "10", "2026-04-01")

        assert result["success"] is True
        assert result["workout_id"] == 999
        # Copies the template into a planned workout (the native
        # addworkoutfromlibraryitem command endpoint returns HTTP 500)
        endpoint = mock_instance.post.call_args[0][0]
        assert endpoint == "/fitness/v6/athletes/123/workouts"
        payload = mock_instance.post.call_args[1]["json"]
        assert payload["workoutDay"] == "2026-04-01T00:00:00"
        assert payload["title"] == "Sweet Spot"
        assert payload["workoutTypeFamilyId"] == 2
        assert payload["workoutTypeValueId"] == 2
        assert payload["workoutSubTypeId"] == 3
        assert payload["tssPlanned"] == 80.0
        # Calendar workouts carry structure as a JSON string
        assert isinstance(payload["structure"], str)

    @pytest.mark.asyncio
    async def test_schedule_without_workout_id_is_error(self):
        """A 200 with no workout id means the create did not persist — the tool
        must report failure, not a false success."""
        items_response = APIResponse(success=True, data=[self.TEMPLATE])
        # TP answered OK but returned no workoutId (silently dropped create).
        create_response = APIResponse(success=True, data={})
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=items_response)
            mock_instance.post = AsyncMock(return_value=create_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_schedule_library_workout("1", "10", "2026-04-01")

        assert result.get("isError") is True
        assert result.get("success") is not True
        assert "not confirmed" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_schedule_resolves_library_and_item_by_name(self):
        """Passing the library NAME and item NAME (not numeric ids) resolves to
        the right ids and schedules the workout."""
        libraries_response = APIResponse(
            success=True,
            data=[{"exerciseLibraryId": 3820613, "libraryName": "Adam"}],
        )
        items_response = APIResponse(success=True, data=[self.TEMPLATE])
        create_response = APIResponse(success=True, data={"workoutId": 999})
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            # 1st GET = libraries (name lookup), 2nd GET = that library's items.
            mock_instance.get = AsyncMock(
                side_effect=[libraries_response, items_response]
            )
            mock_instance.post = AsyncMock(return_value=create_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_schedule_library_workout(
                "Adam", "Sweet Spot", "2026-04-01"
            )

        assert result["success"] is True
        assert result["workout_id"] == 999
        # The items were fetched from the RESOLVED numeric library id.
        items_endpoint = mock_instance.get.call_args_list[1][0][0]
        assert items_endpoint == "/exerciselibrary/v2/libraries/3820613/items"

    @pytest.mark.asyncio
    async def test_schedule_unknown_library_name_errors(self):
        """An unknown library name returns NOT_FOUND, not a crash."""
        libraries_response = APIResponse(
            success=True,
            data=[{"exerciseLibraryId": 3820613, "libraryName": "Adam"}],
        )
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=libraries_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_schedule_library_workout(
                "Nonexistent", "Sweet Spot", "2026-04-01"
            )

        assert result.get("isError") is True
        assert result["error_code"] == "NOT_FOUND"
        mock_instance.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_athlete_ignores_bulk_shape(self):
        """Omitting athletes keeps the original single-athlete result shape."""
        items_response = APIResponse(success=True, data=[self.TEMPLATE])
        create_response = APIResponse(success=True, data={"workoutId": 999})
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=items_response)
            mock_instance.post = AsyncMock(return_value=create_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_schedule_library_workout("1", "10", "2026-04-01")

        assert result["success"] is True
        assert "scheduled" not in result
        assert "errors" not in result

    @pytest.mark.asyncio
    async def test_schedule_unknown_item_returns_not_found(self):
        items_response = APIResponse(success=True, data=[self.TEMPLATE])
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=items_response)
            mock_instance.post = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_schedule_library_workout("1", "404", "2026-04-01")

        assert result.get("isError") is True
        assert result["error_code"] == "NOT_FOUND"
        mock_instance.post.assert_not_called()


class TestScheduleLibraryWorkoutBulk:
    TEMPLATE = TestScheduleLibraryWorkout.TEMPLATE

    def _mock(self, mock_client, athlete_ids, post_responses):
        items_response = APIResponse(success=True, data=[self.TEMPLATE])
        mock_instance = AsyncMock()
        mock_instance.ensure_athlete_id = AsyncMock(side_effect=athlete_ids)
        mock_instance.get = AsyncMock(return_value=items_response)
        mock_instance.post = AsyncMock(side_effect=post_responses)
        mock_client.return_value.__aenter__.return_value = mock_instance
        return mock_instance

    @pytest.mark.asyncio
    async def test_bulk_schedules_each_athlete(self):
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = self._mock(
                mock_client,
                athlete_ids=[111, 222],
                post_responses=[
                    APIResponse(success=True, data={"workoutId": 1001}),
                    APIResponse(success=True, data={"workoutId": 1002}),
                ],
            )

            result = await tp_schedule_library_workout(
                "1", "10", "2026-04-01", athletes=["Alice", "222"],
            )

        assert result.get("isError") is not True
        assert [s["athlete_id"] for s in result["scheduled"]] == [111, 222]
        assert [s["workout_id"] for s in result["scheduled"]] == [1001, 1002]
        assert result["errors"] == []
        endpoints = [c[0][0] for c in mock_instance.post.call_args_list]
        assert endpoints == [
            "/fitness/v6/athletes/111/workouts",
            "/fitness/v6/athletes/222/workouts",
        ]
        # Each payload targets its own athlete
        payloads = [c[1]["json"] for c in mock_instance.post.call_args_list]
        assert [p["athleteId"] for p in payloads] == [111, 222]
        assert all(p["title"] == "Sweet Spot" for p in payloads)

    @pytest.mark.asyncio
    async def test_bulk_partial_failure_is_not_error(self):
        """One athlete failing is reported in errors, without isError."""
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            self._mock(
                mock_client,
                athlete_ids=[111, 222],
                post_responses=[
                    APIResponse(success=True, data={"workoutId": 1001}),
                    APIResponse(success=False, message="boom"),
                ],
            )

            result = await tp_schedule_library_workout(
                "1", "10", "2026-04-01", athletes=["Alice", "Bob"],
            )

        assert result.get("isError") is not True
        assert len(result["scheduled"]) == 1
        assert result["errors"] == [
            {"athlete": "Bob", "athlete_id": 222, "message": "boom"},
        ]

    @pytest.mark.asyncio
    async def test_bulk_unresolvable_athlete_reported(self):
        """An athlete not in the roster (ensure_athlete_id -> None) is a
        per-athlete error; the rest still get scheduled."""
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = self._mock(
                mock_client,
                athlete_ids=[None, 222],
                post_responses=[
                    APIResponse(success=True, data={"workoutId": 1002}),
                ],
            )

            result = await tp_schedule_library_workout(
                "1", "10", "2026-04-01", athletes=["Nobody", "222"],
            )

        assert result.get("isError") is not True
        assert [s["athlete_id"] for s in result["scheduled"]] == [222]
        assert result["errors"][0]["athlete"] == "Nobody"
        assert mock_instance.post.call_count == 1

    @pytest.mark.asyncio
    async def test_bulk_ambiguous_name_reported(self):
        """ensure_athlete_id raising ValueError (ambiguous name) becomes a
        per-athlete error rather than blowing up the whole call."""
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            self._mock(
                mock_client,
                athlete_ids=[ValueError("Ambiguous athlete name 'Alex'"), 222],
                post_responses=[
                    APIResponse(success=True, data={"workoutId": 1002}),
                ],
            )

            result = await tp_schedule_library_workout(
                "1", "10", "2026-04-01", athletes=["Alex", "222"],
            )

        assert result.get("isError") is not True
        assert "Ambiguous" in result["errors"][0]["message"]
        assert [s["athlete_id"] for s in result["scheduled"]] == [222]

    @pytest.mark.asyncio
    async def test_bulk_total_failure_sets_is_error(self):
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            self._mock(
                mock_client,
                athlete_ids=[111, 222],
                post_responses=[
                    APIResponse(success=False, message="boom"),
                    APIResponse(success=False, message="boom"),
                ],
            )

            result = await tp_schedule_library_workout(
                "1", "10", "2026-04-01", athletes=["Alice", "Bob"],
            )

        assert result["isError"] is True
        assert result["error_code"] == "API_ERROR"
        assert result["scheduled"] == []
        assert len(result["errors"]) == 2

    @pytest.mark.asyncio
    async def test_both_athlete_and_athletes_rejected(self):
        """Passing the single 'athlete' target alongside 'athletes' is a
        validation error before any API call."""
        token = athlete_override.set("Alice")
        try:
            with patch("tp_mcp.tools.library.TPClient") as mock_client:
                result = await tp_schedule_library_workout(
                    "1", "10", "2026-04-01", athletes=["Bob"],
                )
        finally:
            athlete_override.reset(token)

        assert result["isError"] is True
        assert result["error_code"] == "VALIDATION_ERROR"
        assert "not both" in result["message"]
        mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_athletes_list_rejected(self):
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            result = await tp_schedule_library_workout(
                "1", "10", "2026-04-01", athletes=[],
            )

        assert result["isError"] is True
        assert result["error_code"] == "VALIDATION_ERROR"
        mock_client.assert_not_called()


def _step(name, seconds, lo, hi, cls):
    return {
        "name": name,
        "type": "step",
        "length": {"value": seconds, "unit": "second"},
        "targets": [{"minValue": lo, "maxValue": hi}],
        "intensityClass": cls,
    }


def _single(begin, end, step):
    return {
        "type": "step",
        "length": {"value": 1, "unit": "repetition"},
        "begin": begin,
        "end": end,
        "steps": [step],
    }


def _repetition(begin, end, reps, steps):
    return {
        "type": "repetition",
        "length": {"value": reps, "unit": "repetition"},
        "begin": begin,
        "end": end,
        "steps": steps,
    }


class TestApplyStructureOverrides:
    """Unit tests for the structure-adjustment helper."""

    # 10min warm-up, 5×(3min work + 2min rest), 5min cool-down = 2400s.
    INTERVAL_STRUCTURE = {
        "primaryIntensityMetric": "percentOfFtp",
        "primaryLengthMetric": "duration",
        "structure": [
            _single(0, 600, _step("Warm up", 600, 45, 55, "warmUp")),
            _repetition(600, 2100, 5, [
                _step("Work", 180, 88, 93, "active"),
                _step("Rest", 120, 40, 50, "rest"),
            ]),
            _single(2100, 2400, _step("Cool down", 300, 45, 55, "coolDown")),
        ],
    }

    # 10min warm-up, 5×(3min work + 2min in-set rest), 20min endurance filler,
    # 5min cool-down = 3600s.
    INTERVAL_WITH_FILLER = {
        "primaryIntensityMetric": "percentOfFtp",
        "primaryLengthMetric": "duration",
        "structure": [
            _single(0, 600, _step("Warm up", 600, 45, 55, "warmUp")),
            _repetition(600, 2100, 5, [
                _step("Work", 180, 88, 93, "active"),
                _step("Rest", 120, 40, 50, "rest"),
            ]),
            _single(2100, 3300, _step("Endurance", 1200, 60, 70, "active")),
            _single(3300, 3600, _step("Cool down", 300, 45, 55, "coolDown")),
        ],
    }

    def test_no_override_returns_untouched(self):
        from tp_mcp.tools.library import _apply_structure_overrides
        out, total = _apply_structure_overrides(self.INTERVAL_STRUCTURE)
        assert out is self.INTERVAL_STRUCTURE
        assert total is None

    def test_endurance_minutes_override_scales_work_keeps_anchors(self):
        from tp_mcp.tools.library import _apply_structure_overrides
        structure = {
            "primaryIntensityMetric": "percentOfFtp",
            "primaryLengthMetric": "duration",
            "structure": [
                _single(0, 300, _step("Warm up", 300, 45, 55, "warmUp")),
                _single(300, 3300, _step("Endurance", 3000, 65, 75, "active")),
                _single(3300, 3600, _step("Cool down", 300, 45, 55, "coolDown")),
            ],
        }
        out, total = _apply_structure_overrides(structure, endurance_minutes=90)
        blocks = out["structure"]
        # Warm-up + cool-down fixed (600s); work scaled 3000 → 4800 to hit 5400s.
        assert blocks[0]["steps"][0]["length"]["value"] == 300
        assert blocks[1]["steps"][0]["length"]["value"] == 4800
        assert blocks[2]["steps"][0]["length"]["value"] == 300
        assert total == 5400

    def test_endurance_override_uniform_when_no_work_step(self):
        from tp_mcp.tools.library import _apply_structure_overrides
        structure = {
            "structure": [
                _single(0, 3600, _step("Ride", 3600, 65, 75, "warmUp")),
            ],
        }
        out, total = _apply_structure_overrides(structure, endurance_minutes=30)
        assert out["structure"][0]["steps"][0]["length"]["value"] == 1800
        assert total == 1800

    def test_longer_duration_keeps_work_fixed_stretches_rest(self):
        # 60min → 70min: the interval set (work 180s AND in-set rest 120s) and
        # the warm-up/cool-down stay fixed; the standalone endurance filler grows
        # to absorb the extra 10 minutes.
        from tp_mcp.tools.library import _apply_structure_overrides
        out, total = _apply_structure_overrides(
            self.INTERVAL_WITH_FILLER, endurance_minutes=70
        )
        blocks = out["structure"]
        assert blocks[0]["steps"][0]["length"]["value"] == 600  # warm-up fixed
        assert blocks[1]["steps"][0]["length"]["value"] == 180  # work UNCHANGED
        assert blocks[1]["steps"][1]["length"]["value"] == 120  # in-set rest UNCHANGED
        assert blocks[2]["steps"][0]["length"]["value"] == 1800  # filler 1200 → 1800
        assert blocks[3]["steps"][0]["length"]["value"] == 300  # cool-down fixed
        assert total == 4200

    def test_shorter_duration_keeps_work_fixed_shrinks_rest(self):
        # 60min → 50min: interval work + in-set rest stay fixed; the endurance
        # filler shrinks from 1200s to 600s.
        from tp_mcp.tools.library import _apply_structure_overrides
        out, total = _apply_structure_overrides(
            self.INTERVAL_WITH_FILLER, endurance_minutes=50
        )
        blocks = out["structure"]
        assert blocks[1]["steps"][0]["length"]["value"] == 180  # work UNCHANGED
        assert blocks[1]["steps"][1]["length"]["value"] == 120  # in-set rest UNCHANGED
        assert blocks[2]["steps"][0]["length"]["value"] == 600  # filler 1200 → 600
        assert total == 3000

    def test_very_short_target_keeps_intervals_fixed_collapses_rest(self):
        # Target shorter than intervals + warm-up/cool-down: the interval efforts
        # and their in-set rest stay put and the filler collapses toward its
        # minimum, never compressing the efforts.
        from tp_mcp.tools.library import _apply_structure_overrides
        out, _ = _apply_structure_overrides(
            self.INTERVAL_WITH_FILLER, endurance_minutes=30
        )
        blocks = out["structure"]
        assert blocks[1]["steps"][0]["length"]["value"] == 180  # work UNCHANGED
        assert blocks[1]["steps"][1]["length"]["value"] == 120  # in-set rest UNCHANGED
        assert blocks[2]["steps"][0]["length"]["value"] < 1200  # filler collapsed

    def test_repetition_intervals_fixed_endurance_filler_absorbs_change(self):
        # Mirrors the reported case: 5×(3:40 VT-2 + 0:20 VO₂) with NO rest inside
        # the block. A separate endurance step flexes; the efforts stay fixed.
        from tp_mcp.tools.library import _apply_structure_overrides
        structure = {
            "structure": [
                _single(0, 600, _step("Warm up", 600, 45, 55, "warmUp")),
                _repetition(600, 1800, 5, [
                    _step("VT-2", 220, 95, 100, "active"),
                    _step("VO2", 20, 118, 125, "active"),
                ]),
                _single(1800, 4800, _step("Endurance", 3000, 60, 70, "active")),
                _single(4800, 5100, _step("Cool down", 300, 45, 55, "coolDown")),
            ],
        }
        out, total = _apply_structure_overrides(structure, endurance_minutes=70)
        blocks = out["structure"]
        assert blocks[1]["steps"][0]["length"]["value"] == 220  # VT-2 UNCHANGED
        assert blocks[1]["steps"][1]["length"]["value"] == 20   # VO2 UNCHANGED
        assert blocks[1]["end"] == 600 + 5 * (220 + 20)         # block 2 = 20:00
        assert blocks[2]["steps"][0]["length"]["value"] == 2100  # 3000 → 2100
        assert blocks[3]["steps"][0]["length"]["value"] == 300   # cool-down fixed
        assert total == 4200  # total 70 min

    def test_in_set_recovery_and_edge_floors_reported_case(self):
        # Exact reported bug: VO₂ (a `rest` step INSIDE the VT-2 repetition set)
        # and the sprint's in-set easy recovery must NOT be reduced, while the
        # standalone aerobic/easy blocks absorb the change and the opening
        # warm-up / closing cool-down keep a 10-minute floor.
        from tp_mcp.tools.library import _apply_structure_overrides
        structure = {
            "structure": [
                _single(0, 1800, _step("Aerobic", 1800, 60, 70, "active")),
                _repetition(1800, 3000, 5, [
                    _step("VT-2", 220, 90, 92, "active"),
                    _step("Vo2", 20, 110, 120, "rest"),
                ]),
                _single(3000, 4200, _step("Easy", 1200, 40, 50, "warmUp")),
                _repetition(4200, 5400, 5, [
                    _step("VT-2", 220, 90, 92, "active"),
                    _step("Vo2", 20, 110, 120, "rest"),
                ]),
                _single(5400, 6600, _step("Easy", 1200, 40, 50, "warmUp")),
                _single(6600, 10425, _step("Aerobic", 3825, 60, 70, "active")),
                _repetition(10425, 11375, 5, [
                    _step("Sprint", 10, 200, 400, "active"),
                    _step("Easy", 180, 50, 60, "rest"),
                ]),
                _single(11375, 12000, _step("Aerobic tail", 625, 60, 70, "active")),
            ],
        }
        out, _ = _apply_structure_overrides(structure, endurance_minutes=180)
        blocks = out["structure"]
        # In-set recovery steps stay fixed (the reported 20s → 18s regression).
        assert blocks[1]["steps"][1]["length"]["value"] == 20   # VO₂ UNCHANGED
        assert blocks[6]["steps"][1]["length"]["value"] == 180  # sprint recovery UNCHANGED
        # Interval efforts untouched.
        assert blocks[1]["steps"][0]["length"]["value"] == 220
        assert blocks[6]["steps"][0]["length"]["value"] == 10
        # Opening warm-up and closing cool-down keep their 10-minute floor.
        assert blocks[0]["steps"][0]["length"]["value"] >= 600
        assert blocks[7]["steps"][0]["length"]["value"] >= 600

    def test_reps_not_adjustable_endurance_only(self):
        # Interval reps are fixed by the template; a plain endurance ride only
        # scales its duration.
        from tp_mcp.tools.library import _apply_structure_overrides
        structure = {
            "structure": [
                _single(0, 3600, _step("Ride", 3600, 65, 75, "active")),
            ],
        }
        out, total = _apply_structure_overrides(structure, endurance_minutes=30)
        assert out["structure"][0]["length"]["value"] == 1  # unchanged
        assert total == 1800

    def test_non_native_structure_ignored(self):
        from tp_mcp.tools.library import _apply_structure_overrides
        assert _apply_structure_overrides(None, endurance_minutes=90) == (None, None)
        assert _apply_structure_overrides({"steps": []}, endurance_minutes=90) == (
            {"steps": []}, None,
        )

    def test_interval_reps_override_rewrites_block(self):
        from tp_mcp.tools.library import _apply_structure_overrides
        out, total = _apply_structure_overrides(
            self.INTERVAL_STRUCTURE, interval_reps={0: 3}
        )
        blocks = out["structure"]
        assert blocks[1]["length"]["value"] == 3  # reps 5 → 3
        # total = warm-up 600 + 3×(180+120) + cool-down 300 = 1800
        assert total == 600 + 3 * (180 + 120) + 300
        # Source untouched (deep copy)
        assert self.INTERVAL_STRUCTURE["structure"][1]["length"]["value"] == 5

    def test_interval_reps_zero_removes_set(self):
        from tp_mcp.tools.library import _apply_structure_overrides
        out, total = _apply_structure_overrides(
            self.INTERVAL_STRUCTURE, interval_reps={0: 0}
        )
        blocks = out["structure"]
        # The repetition block is dropped, leaving warm-up + cool-down only.
        assert len(blocks) == 2
        assert all(b.get("type") != "repetition" for b in blocks)
        # total = warm-up 600 + cool-down 300 = 900
        assert total == 900
        # Source untouched (deep copy)
        assert len(self.INTERVAL_STRUCTURE["structure"]) == 3

    def test_interval_reps_zero_removes_trailing_rest_block(self):
        from tp_mcp.tools.library import _apply_structure_overrides
        structure = {
            "structure": [
                _single(0, 600, _step("Warm up", 600, 45, 55, "warmUp")),
                _repetition(600, 2100, 5, [
                    _step("Work", 180, 88, 93, "active"),
                    _step("Rest", 120, 40, 50, "rest"),
                ]),
                _single(2100, 2400, _step("Recovery", 300, 40, 50, "rest")),
                _repetition(2400, 3900, 5, [
                    _step("Work", 180, 88, 93, "active"),
                    _step("Rest", 120, 40, 50, "rest"),
                ]),
                _single(3900, 4200, _step("Cool down", 300, 45, 55, "coolDown")),
            ],
        }
        out, _ = _apply_structure_overrides(structure, interval_reps={0: 0})
        blocks = out["structure"]
        # First interval set AND the recovery block right after it are removed.
        assert [b["steps"][0]["name"] for b in blocks] == [
            "Warm up",
            "Work",
            "Cool down",
        ]
        assert blocks[1]["type"] == "repetition"  # second set survives

    def test_interval_reps_zero_removes_easy_warmup_recovery(self):
        from tp_mcp.tools.library import _apply_structure_overrides
        # Real-world shape: sets separated by an "Easy" recovery that TP tags
        # warmUp. Dropping the middle set must also drop its trailing recovery.
        structure = {
            "structure": [
                _single(0, 1800, _step("Aerobic", 1800, 60, 70, "active")),
                _repetition(1800, 3000, 5, [
                    _step("VT-2", 220, 90, 92, "active"),
                    _step("Vo2", 20, 110, 120, "rest"),
                ]),
                _single(3000, 4200, _step("Easy", 1200, 40, 50, "warmUp")),
                _repetition(4200, 5400, 5, [
                    _step("VT-2", 220, 90, 92, "active"),
                    _step("Vo2", 20, 110, 120, "rest"),
                ]),
                _single(5400, 6600, _step("Easy", 1200, 40, 50, "warmUp")),
                _repetition(6600, 7800, 5, [
                    _step("VT-2", 220, 90, 92, "active"),
                    _step("Vo2", 20, 110, 120, "rest"),
                ]),
                _single(7800, 9000, _step("Aerobic tail", 1200, 60, 70, "active")),
            ],
        }
        out, _ = _apply_structure_overrides(structure, interval_reps={1: 0})
        names = [b["steps"][0]["name"] for b in out["structure"]]
        assert names == ["Aerobic", "VT-2", "Easy", "VT-2", "Aerobic tail"]

    def test_interval_reps_zero_preserves_trailing_cooldown(self):
        from tp_mcp.tools.library import _apply_structure_overrides
        out, _ = _apply_structure_overrides(
            self.INTERVAL_STRUCTURE, interval_reps={0: 0}
        )
        # INTERVAL_STRUCTURE ends warm-up, set, cool-down: the cool-down is the
        # final block and must survive when the set is dropped.
        names = [b["steps"][0]["name"] for b in out["structure"]]
        assert names == ["Warm up", "Cool down"]

    def test_interval_reps_and_duration_applies_reps_first(self):
        from tp_mcp.tools.library import _apply_structure_overrides
        out, total = _apply_structure_overrides(
            self.INTERVAL_STRUCTURE, interval_reps={0: 3}, endurance_minutes=30
        )
        blocks = out["structure"]
        assert blocks[1]["length"]["value"] == 3  # reps applied
        assert blocks[0]["steps"][0]["length"]["value"] == 600  # warm-up anchor
        assert blocks[2]["steps"][0]["length"]["value"] == 300  # cool-down anchor
        assert total == 1800  # scaled to 30 min

    def test_interval_reps_unknown_ordinal_noop(self):
        from tp_mcp.tools.library import _apply_structure_overrides
        out, total = _apply_structure_overrides(
            self.INTERVAL_STRUCTURE, interval_reps={5: 9}
        )
        blocks = out["structure"]
        assert blocks[1]["length"]["value"] == 5  # unchanged
        assert total == 2400  # unchanged total

    def test_interval_reps_leaves_non_repetition_blocks_untouched(self):
        from tp_mcp.tools.library import _apply_structure_overrides
        out, _ = _apply_structure_overrides(
            self.INTERVAL_STRUCTURE, interval_reps={0: 3}
        )
        blocks = out["structure"]
        assert blocks[0]["length"]["value"] == 1  # single warm-up block
        assert blocks[2]["length"]["value"] == 1  # single cool-down block



class TestScheduleWithOverrides:
    STRUCTURED_TEMPLATE = {
        "exerciseLibraryItemId": 10,
        "itemName": "5x3 VO2",
        "workoutTypeId": 2,
        "workoutSubTypeId": 3,
        "totalTimePlanned": 2400 / 3600,   # 0.6667h, matches the structure
        "tssPlanned": 60.0,
        "ifPlanned": 0.9,
        "description": "5x3min @ 110%",
        "structure": TestApplyStructureOverrides.INTERVAL_STRUCTURE,
    }

    async def _run(self, **kwargs):
        import json

        items_response = APIResponse(success=True, data=[self.STRUCTURED_TEMPLATE])
        create_response = APIResponse(success=True, data={"workoutId": 999})
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=items_response)
            mock_instance.post = AsyncMock(return_value=create_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_schedule_library_workout("1", "10", "2026-04-01", **kwargs)
        payload = mock_instance.post.call_args[1]["json"]
        payload_structure = json.loads(payload["structure"])
        return result, payload, payload_structure

    @pytest.mark.asyncio
    async def test_endurance_override_flows_into_payload(self):
        # endurance_minutes_override=45 → 2700s → 0.75h; TSS scaled from 60.
        result, payload, structure = await self._run(endurance_minutes_override=45)
        assert result["success"] is True
        # Interval reps are untouched — only the duration scales.
        assert structure["structure"][1]["length"]["value"] == 5
        assert payload["totalTimePlanned"] == 0.75
        assert payload["tssPlanned"] == 67.5

    @pytest.mark.asyncio
    async def test_description_override_replaces_text(self):
        result, payload, _ = await self._run(description_override="6x3min @ 110%")
        assert result["success"] is True
        assert payload["description"] == "6x3min @ 110%"

    @pytest.mark.asyncio
    async def test_no_override_copies_template_verbatim(self):
        _, payload, structure = await self._run()
        assert structure["structure"][1]["length"]["value"] == 5
        assert payload["description"] == "5x3min @ 110%"
        assert payload["tssPlanned"] == 60.0

    @pytest.mark.asyncio
    async def test_bulk_applies_override_to_every_athlete(self):
        import json

        items_response = APIResponse(success=True, data=[self.STRUCTURED_TEMPLATE])
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(side_effect=[111, 222])
            mock_instance.get = AsyncMock(return_value=items_response)
            mock_instance.post = AsyncMock(side_effect=[
                APIResponse(success=True, data={"workoutId": 1001}),
                APIResponse(success=True, data={"workoutId": 1002}),
            ])
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_schedule_library_workout(
                "1", "10", "2026-04-01", athletes=["Alice", "222"],
                endurance_minutes_override=45,
            )

        assert result.get("isError") is not True
        payloads = [c[1]["json"] for c in mock_instance.post.call_args_list]
        for p in payloads:
            # Reps stay at the template's 5; only the duration is scaled.
            assert json.loads(p["structure"])["structure"][1]["length"]["value"] == 5
            assert p["totalTimePlanned"] == 0.75

    @pytest.mark.asyncio
    async def test_invalid_endurance_minutes_rejected(self):
        with patch("tp_mcp.tools.library.TPClient") as mock_client:
            result = await tp_schedule_library_workout(
                "1", "10", "2026-04-01", endurance_minutes_override=-5,
            )
        assert result["isError"] is True
        assert result["error_code"] == "VALIDATION_ERROR"
        mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_interval_reps_override_flows_into_payload(self):
        # interval_reps_override={"0": 3}: reps 5 → 3, total 2400 → 1800s (0.5h);
        # TSS scaled proportionally from 60 (0.6667h) → 45.
        result, payload, structure = await self._run(
            interval_reps_override={"0": 3}
        )
        assert result["success"] is True
        assert structure["structure"][1]["length"]["value"] == 3
        assert payload["totalTimePlanned"] == 0.5
        assert payload["tssPlanned"] == 45.0

    @pytest.mark.asyncio
    async def test_interval_reps_override_tolerates_quoted_keys(self):
        # Models sometimes over-escape the map keys (e.g. '"0"' instead of '0').
        # The tool must strip the extra quotes and apply the reps just the same
        # as a clean {"0": 3}, not reject the whole override.
        result, payload, structure = await self._run(
            interval_reps_override={'"0"': 3}
        )
        assert result["success"] is True
        assert structure["structure"][1]["length"]["value"] == 3
        assert payload["totalTimePlanned"] == 0.5
        assert payload["tssPlanned"] == 45.0

    @pytest.mark.asyncio
    async def test_interval_reps_and_duration_combined(self):
        result, payload, structure = await self._run(
            interval_reps_override={"0": 3}, endurance_minutes_override=30,
        )
        assert result["success"] is True
        assert structure["structure"][1]["length"]["value"] == 3
        assert payload["totalTimePlanned"] == 0.5  # 30 min

    @pytest.mark.asyncio
    async def test_invalid_interval_reps_rejected(self):
        for bad in (
            [1, 2, 3],          # non-dict
            {"0": True},        # bool value
            {"0": -1},          # below 0
            {"0": 101},         # above 100
            {"0": 3.5},         # non-integer
            {"foo": 3},         # non-integer key
        ):
            with patch("tp_mcp.tools.library.TPClient") as mock_client:
                result = await tp_schedule_library_workout(
                    "1", "10", "2026-04-01", interval_reps_override=bad,
                )
            assert result["isError"] is True, bad
            assert result["error_code"] == "VALIDATION_ERROR", bad
            mock_client.assert_not_called()



