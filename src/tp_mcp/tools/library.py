"""Workout library tools: templates, scheduling."""

import copy
import json
import logging
from typing import Any

from pydantic import ValidationError

from tp_mcp.client import TPClient
from tp_mcp.client.context import athlete_override
from tp_mcp.tools._validation import WorkoutIdInput, format_validation_error

logger = logging.getLogger("tp-mcp")


def _step_intensity(step: dict[str, Any]) -> float | None:
    """The bar height value for a step: the target's maxValue, falling back to
    minValue when only a floor is set (e.g. `{"minValue": 55}`). None when the
    step has no numeric target (→ flat bar)."""
    targets = step.get("targets") or []
    if not targets:
        return None
    t = targets[0]
    v = t.get("maxValue")
    if v is None:
        v = t.get("minValue")
    return v if isinstance(v, (int, float)) else None


def _compute_native_polyline(blocks: list[dict[str, Any]]) -> list[list[float]]:
    """Rectangular-bar preview polyline from a native structure block list.

    y is NORMALISED so the structure's highest target = 1.0 (HAR-verified
    against TP's own web-UI preview: e.g. a percentOfFtp item with targets up to
    102% renders 65% as 0.637 and 90% as 0.882). This relative scaling is
    intensity-metric AGNOSTIC — it is correct for percentOf* targets AND for
    absolute watts/pace, whereas a fixed `maxValue / 100` produced 2.5-tall bars
    for an absolute-watt target and could exceed 1.0 for any >100% step.
    t is normalised to total length (duration or distance — units cancel);
    repetition blocks expand into per-rep bars."""
    # Expand to a flat list of (length, intensity) per step instance.
    spans: list[tuple[float, float | None]] = []
    total = 0.0
    for b in blocks:
        reps = int(b["length"]["value"]) if b.get("type") == "repetition" else 1
        for _ in range(reps):
            for s in b.get("steps", []):
                ln = s.get("length", {}).get("value", 0) or 0
                spans.append((ln, _step_intensity(s)))
                total += ln
    ymax = max((y for _, y in spans if isinstance(y, (int, float)) and y > 0),
               default=0.0)
    if total <= 0 or ymax <= 0:
        return []
    poly: list[list[float]] = []
    pos = 0.0
    for ln, y in spans:
        yn = round(y / ymax, 4) if isinstance(y, (int, float)) and y > 0 else 0
        t0 = pos / total
        pos += ln
        t1 = pos / total
        poly.append([round(t0, 4), 0])
        poly.append([round(t0, 4), yn])
        poly.append([round(t1, 4), yn])
        poly.append([round(t1, 4), 0])
    return poly


def _ensure_structure_preview(structure: Any) -> Any:
    """Library create/update store the structure as-is and do NOT build the
    preview fields, so templates saved this way render without a structure
    thumbnail in TP (HAR-verified: a raw API create without `polyline` reads
    back with no polyline — TP never backfills it server-side). Backfill the two
    missing pieces: `primaryIntensityTargetOrRange` and `polyline` (the preview
    graph). Only touches a native structure (dict with a "structure" block
    list); anything else is returned untouched. Returns a shallow COPY — the
    caller's dict is never mutated in place."""
    if not isinstance(structure, dict):
        return structure
    blocks = structure.get("structure")
    if not isinstance(blocks, list) or not blocks:
        return structure
    out = dict(structure)               # copy — do not mutate the caller's dict
    out.setdefault("primaryIntensityTargetOrRange", "range")
    if not out.get("polyline"):
        poly = _compute_native_polyline(blocks)
        if poly:
            out["polyline"] = poly
    return out


def _block_reps(block: dict[str, Any]) -> int:
    """Repetition count for a block (1 for a single-step block)."""
    if block.get("type") == "repetition":
        return int((block.get("length") or {}).get("value", 1) or 1)
    return 1


def _step_seconds(block: dict[str, Any]) -> int:
    """Total seconds of one pass through a block's steps (before reps)."""
    total = 0
    for s in block.get("steps", []):
        total += int((s.get("length") or {}).get("value", 0) or 0)
    return total


def _recompute_begin_end(blocks: list[dict[str, Any]]) -> int:
    """Rewrite each block's cumulative ``begin``/``end`` from its own step
    durations and rep count. Returns the structure's total seconds."""
    cursor = 0
    for b in blocks:
        duration = _step_seconds(b) * _block_reps(b)
        b["begin"] = cursor
        cursor += duration
        b["end"] = cursor
    return cursor


def _primary_repetition_block(
    blocks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """The template's main interval set: the ``repetition`` block with the most
    reps (ties broken by longest per-pass duration). ``None`` when the structure
    has no repetition block (e.g. a plain endurance ride)."""
    reps = [b for b in blocks if b.get("type") == "repetition"]
    if not reps:
        return None
    return max(
        reps,
        key=lambda b: ((b.get("length") or {}).get("value", 0) or 0, _step_seconds(b)),
    )


def _set_total_duration(blocks: list[dict[str, Any]], target_seconds: int) -> None:
    """Scale the workout so its total planned time equals ``target_seconds``.

    Warm-up, cool-down and rest steps are treated as fixed anchors; the
    remaining ``active``/``other`` work steps are scaled proportionally to
    absorb the change. When nothing is marked as work (e.g. a single untyped
    endurance step), every step is scaled uniformly instead."""
    def _classify() -> tuple[list[tuple[dict[str, Any], int]], int, int]:
        work: list[tuple[dict[str, Any], int]] = []
        work_secs = 0
        fixed_secs = 0
        for b in blocks:
            reps = _block_reps(b)
            for s in b.get("steps", []):
                dur = int((s.get("length") or {}).get("value", 0) or 0)
                if s.get("intensityClass") in ("warmUp", "coolDown", "rest"):
                    fixed_secs += dur * reps
                else:
                    work.append((s, reps))
                    work_secs += dur * reps
        return work, work_secs, fixed_secs

    work, work_secs, fixed_secs = _classify()
    if not work or work_secs <= 0:
        # No dedicated work steps — scale everything uniformly.
        work = [(s, _block_reps(b)) for b in blocks for s in b.get("steps", [])]
        work_secs = sum(
            int((s.get("length") or {}).get("value", 0) or 0) * reps
            for s, reps in work
        )
        fixed_secs = 0
    if work_secs <= 0:
        return
    budget = max(target_seconds - fixed_secs, len(work))
    factor = budget / work_secs
    for step, _reps in work:
        length = step.setdefault("length", {})
        length["value"] = max(int(round((length.get("value", 0) or 0) * factor)), 1)
        length.setdefault("unit", "second")


def _apply_structure_overrides(
    structure: Any,
    *,
    interval_reps: int | None = None,
    endurance_minutes: float | None = None,
) -> tuple[Any, int | None]:
    """Return a COPY of a native template structure with its main interval rep
    count and/or total duration adjusted, plus the resulting total seconds.

    ``begin``/``end`` offsets and the preview ``polyline`` are recomputed so the
    scheduled workout renders correctly. Non-native or empty structures (and the
    no-override case) are returned untouched with ``total_seconds=None``."""
    if interval_reps is None and endurance_minutes is None:
        return structure, None
    if not isinstance(structure, dict):
        return structure, None
    blocks = structure.get("structure")
    if not isinstance(blocks, list) or not blocks:
        return structure, None

    out = dict(structure)
    blocks = copy.deepcopy(blocks)
    out["structure"] = blocks

    if interval_reps is not None:
        rep_block = _primary_repetition_block(blocks)
        if rep_block is not None:
            length = rep_block.setdefault("length", {})
            length["value"] = int(interval_reps)
            length.setdefault("unit", "repetition")

    if endurance_minutes is not None:
        _set_total_duration(blocks, int(round(endurance_minutes * 60)))

    total_seconds = _recompute_begin_end(blocks)
    poly = _compute_native_polyline(blocks)
    if poly:
        out["polyline"] = poly
    return out, total_seconds


async def tp_get_libraries() -> dict[str, Any]:
    """List all workout library folders.

    Returns:
        Dict with libraries list.
    """
    async with TPClient() as client:
        athlete_id = await client.ensure_athlete_id()
        if not athlete_id:
            return {
                "isError": True,
                "error_code": "AUTH_INVALID",
                "message": "Could not get athlete ID. Re-authenticate.",
            }

        endpoint = "/exerciselibrary/v2/libraries"
        response = await client.get(endpoint)

        if response.is_error:
            return {
                "isError": True,
                "error_code": response.error_code.value if response.error_code else "API_ERROR",
                "message": response.message,
            }

        data = response.data if isinstance(response.data, list) else []
        libraries = [
            {
                "id": lib.get("exerciseLibraryId", lib.get("id")),
                # v2 libraries endpoint returns "libraryName"/"isDefaultContent";
                # keep the old keys as fallbacks for safety.
                "name": lib.get("libraryName", lib.get("name", "")),
                "is_default": lib.get("isDefaultContent", lib.get("isDefault", False)),
                "owner_name": lib.get("ownerName"),
                # The v2 libraries endpoint usually omits an item count; read it
                # if present, otherwise fall back to 0.
                "item_count": lib.get("itemCount", 0),
                "owner_id": lib.get("ownerId"),
            }
            for lib in data
        ]

        return {"libraries": libraries, "count": len(libraries)}


async def tp_get_library_items(library_id: str) -> dict[str, Any]:
    """List templates in a workout library.

    Args:
        library_id: Library ID.

    Returns:
        Dict with library items list.
    """
    try:
        validated = WorkoutIdInput(workout_id=library_id)
    except (ValidationError, ValueError) as e:
        msg = format_validation_error(e) if isinstance(e, ValidationError) else str(e)
        return {
            "isError": True,
            "error_code": "VALIDATION_ERROR",
            "message": msg,
        }

    async with TPClient() as client:
        athlete_id = await client.ensure_athlete_id()
        if not athlete_id:
            return {
                "isError": True,
                "error_code": "AUTH_INVALID",
                "message": "Could not get athlete ID. Re-authenticate.",
            }

        endpoint = f"/exerciselibrary/v2/libraries/{validated.workout_id}/items"
        response = await client.get(endpoint)

        if response.is_error:
            return {
                "isError": True,
                "error_code": response.error_code.value if response.error_code else "API_ERROR",
                "message": response.message,
            }

        data = response.data if isinstance(response.data, list) else []
        items = [
            {
                "id": item.get("exerciseLibraryItemId", item.get("id")),
                "name": item.get("itemName", item.get("name", "")),
                "sport": item.get("workoutTypeId"),
                "duration": item.get("totalTimePlanned"),
                "tss": item.get("tssPlanned"),
            }
            for item in data
        ]

        return {
            "items": items,
            "count": len(items),
            "library_id": validated.workout_id,
        }


async def tp_get_library_item(library_id: str, item_id: str) -> dict[str, Any]:
    """Get full template details including structure.

    Args:
        library_id: Library ID.
        item_id: Library item ID.

    Returns:
        Dict with item details.
    """
    try:
        lib_validated = WorkoutIdInput(workout_id=library_id)
        item_validated = WorkoutIdInput(workout_id=item_id)
    except (ValidationError, ValueError) as e:
        msg = format_validation_error(e) if isinstance(e, ValidationError) else str(e)
        return {
            "isError": True,
            "error_code": "VALIDATION_ERROR",
            "message": msg,
        }

    async with TPClient() as client:
        athlete_id = await client.ensure_athlete_id()
        if not athlete_id:
            return {
                "isError": True,
                "error_code": "AUTH_INVALID",
                "message": "Could not get athlete ID. Re-authenticate.",
            }

        # Get all items and find the specific one
        endpoint = f"/exerciselibrary/v2/libraries/{lib_validated.workout_id}/items"
        response = await client.get(endpoint)

        if response.is_error:
            return {
                "isError": True,
                "error_code": response.error_code.value if response.error_code else "API_ERROR",
                "message": response.message,
            }

        data = response.data if isinstance(response.data, list) else []

        for item in data:
            iid = item.get("exerciseLibraryItemId", item.get("id"))
            if iid == item_validated.workout_id:
                return {"item": item}

        return {
            "isError": True,
            "error_code": "NOT_FOUND",
            "message": f"Item {item_validated.workout_id} not found in library {lib_validated.workout_id}.",
        }


async def tp_create_library(name: str) -> dict[str, Any]:
    """Create a workout library folder.

    Args:
        name: Library name.

    Returns:
        Dict with confirmation or error.
    """
    if not name or not name.strip():
        return {
            "isError": True,
            "error_code": "VALIDATION_ERROR",
            "message": "Library name must not be empty.",
        }

    async with TPClient() as client:
        athlete_id = await client.ensure_athlete_id()
        if not athlete_id:
            return {
                "isError": True,
                "error_code": "AUTH_INVALID",
                "message": "Could not get athlete ID. Re-authenticate.",
            }

        # TP expects "libraryName" and the owner's personId, not "name".
        owner_id = None
        user_data = await client._get_user_data()
        if user_data:
            owner_id = user_data.get("personId")

        endpoint = "/exerciselibrary/v1/libraries"
        payload: dict[str, Any] = {"libraryName": name.strip()}
        if owner_id is not None:
            payload["ownerId"] = owner_id
        response = await client.post(endpoint, json=payload)

        if response.is_error:
            return {
                "isError": True,
                "error_code": response.error_code.value if response.error_code else "API_ERROR",
                "message": response.message,
            }

        lib_id = None
        if isinstance(response.data, dict):
            lib_id = response.data.get("exerciseLibraryId", response.data.get("id"))

        return {
            "success": True,
            "library_id": lib_id,
            "name": name.strip(),
        }


async def tp_delete_library(library_id: str) -> dict[str, Any]:
    """Delete a workout library folder and all its templates.

    Args:
        library_id: Library ID.

    Returns:
        Dict with confirmation or error.
    """
    try:
        validated = WorkoutIdInput(workout_id=library_id)
    except (ValidationError, ValueError) as e:
        msg = format_validation_error(e) if isinstance(e, ValidationError) else str(e)
        return {
            "isError": True,
            "error_code": "VALIDATION_ERROR",
            "message": msg,
        }

    async with TPClient() as client:
        athlete_id = await client.ensure_athlete_id()
        if not athlete_id:
            return {
                "isError": True,
                "error_code": "AUTH_INVALID",
                "message": "Could not get athlete ID. Re-authenticate.",
            }

        endpoint = f"/exerciselibrary/v1/libraries/{validated.workout_id}"
        response = await client.delete(endpoint)

        if response.is_error:
            return {
                "isError": True,
                "error_code": response.error_code.value if response.error_code else "API_ERROR",
                "message": response.message,
            }

        return {
            "success": True,
            "message": f"Library {validated.workout_id} deleted.",
        }


async def tp_create_library_item(
    library_id: str,
    name: str,
    sport_family_id: int,
    sport_type_id: int,
    duration_hours: float | None = None,
    tss: float | None = None,
    description: str | None = None,
    structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a workout template to a library.

    Args:
        library_id: Library ID.
        name: Template name.
        sport_family_id: Sport ID (e.g. 2 = Bike; see tp_get_workout_types).
        sport_type_id: Sport subtype ID (e.g. 3 = Road Bike).
        duration_hours: Optional duration in hours.
        tss: Optional planned TSS.
        description: Optional description.
        structure: Optional interval structure (nested object, NOT string).

    Returns:
        Dict with confirmation or error.
    """
    try:
        lib_validated = WorkoutIdInput(workout_id=library_id)
    except (ValidationError, ValueError) as e:
        msg = format_validation_error(e) if isinstance(e, ValidationError) else str(e)
        return {
            "isError": True,
            "error_code": "VALIDATION_ERROR",
            "message": msg,
        }

    if not name or not name.strip():
        return {
            "isError": True,
            "error_code": "VALIDATION_ERROR",
            "message": "Template name must not be empty.",
        }

    async with TPClient() as client:
        athlete_id = await client.ensure_athlete_id()
        if not athlete_id:
            return {
                "isError": True,
                "error_code": "AUTH_INVALID",
                "message": "Could not get athlete ID. Re-authenticate.",
            }

        # Library items use workoutTypeId/workoutSubTypeId (not the
        # workoutTypeFamilyId/workoutTypeValueId pair of the fitness API).
        # Sending the wrong field names silently creates items with sport 0
        # ("unknown"), which render without power targets in the TP UI.
        payload: dict[str, Any] = {
            "exerciseLibraryId": lib_validated.workout_id,
            "itemName": name.strip(),
            "workoutTypeId": sport_family_id,
            "workoutSubTypeId": sport_type_id,
        }
        if duration_hours is not None:
            payload["totalTimePlanned"] = duration_hours
        if tss is not None:
            payload["tssPlanned"] = tss
        if description:
            payload["description"] = description
        if structure is not None:
            # Library items use nested object, NOT double-serialised string.
            # Backfill polyline/range so TP renders the structure preview.
            payload["structure"] = _ensure_structure_preview(structure)

        endpoint = f"/exerciselibrary/v1/libraries/{lib_validated.workout_id}/items"
        response = await client.post(endpoint, json=payload)

        if response.is_error:
            return {
                "isError": True,
                "error_code": response.error_code.value if response.error_code else "API_ERROR",
                "message": response.message,
            }

        item_id = None
        if isinstance(response.data, dict):
            item_id = response.data.get("exerciseLibraryItemId", response.data.get("id"))

        return {
            "success": True,
            "item_id": item_id,
            "name": name.strip(),
            "library_id": lib_validated.workout_id,
        }


async def tp_update_library_item(
    library_id: str,
    item_id: str,
    name: str | None = None,
    duration_hours: float | None = None,
    tss: float | None = None,
    description: str | None = None,
    structure: dict[str, Any] | None = None,
    workout_type_id: int | None = None,
    workout_sub_type_id: int | None = None,
) -> dict[str, Any]:
    """Edit a workout template.

    Args:
        library_id: Library ID.
        item_id: Item ID.
        name: Optional new name.
        duration_hours: Optional duration in hours.
        tss: Optional planned TSS.
        description: Optional description.
        structure: Optional structure (nested object).
        workout_type_id: Optional sport/workout type (1=swim, 2=bike, 3=run, ...).
            Use to set the sport on templates that were saved without one.
        workout_sub_type_id: Optional workout subtype id (e.g. 6=Indoor Bike).

    Returns:
        Dict with confirmation or error.
    """
    try:
        lib_validated = WorkoutIdInput(workout_id=library_id)
        item_validated = WorkoutIdInput(workout_id=item_id)
    except (ValidationError, ValueError) as e:
        msg = format_validation_error(e) if isinstance(e, ValidationError) else str(e)
        return {
            "isError": True,
            "error_code": "VALIDATION_ERROR",
            "message": msg,
        }

    async with TPClient() as client:
        athlete_id = await client.ensure_athlete_id()
        if not athlete_id:
            return {
                "isError": True,
                "error_code": "AUTH_INVALID",
                "message": "Could not get athlete ID. Re-authenticate.",
            }

        # GET existing items to find and merge
        get_endpoint = f"/exerciselibrary/v2/libraries/{lib_validated.workout_id}/items"
        get_response = await client.get(get_endpoint)

        if get_response.is_error:
            return {
                "isError": True,
                "error_code": get_response.error_code.value if get_response.error_code else "API_ERROR",
                "message": get_response.message,
            }

        data = get_response.data if isinstance(get_response.data, list) else []

        existing = None
        for item in data:
            iid = item.get("exerciseLibraryItemId", item.get("id"))
            if iid == item_validated.workout_id:
                existing = item
                break

        if existing is None:
            return {
                "isError": True,
                "error_code": "NOT_FOUND",
                "message": f"Item {item_validated.workout_id} not found.",
            }

        # Merge updates
        if name is not None:
            existing["itemName"] = name
        if duration_hours is not None:
            existing["totalTimePlanned"] = duration_hours
        if tss is not None:
            existing["tssPlanned"] = tss
        if description is not None:
            existing["description"] = description
        if structure is not None:
            existing["structure"] = _ensure_structure_preview(structure)
        if workout_type_id is not None:
            existing["workoutTypeId"] = workout_type_id
        if workout_sub_type_id is not None:
            existing["workoutSubTypeId"] = workout_sub_type_id

        put_endpoint = (
            f"/exerciselibrary/v1/libraries/{lib_validated.workout_id}"
            f"/items/{item_validated.workout_id}"
        )
        put_response = await client.put(put_endpoint, json=existing)

        if put_response.is_error:
            return {
                "isError": True,
                "error_code": put_response.error_code.value if put_response.error_code else "API_ERROR",
                "message": put_response.message,
            }

        return {
            "success": True,
            "message": f"Library item {item_validated.workout_id} updated.",
        }


def _template_workout_payload(
    item: dict[str, Any],
    date: str,
    athlete_id: int,
    *,
    description_override: str | None = None,
    interval_reps_override: int | None = None,
    endurance_minutes_override: float | None = None,
) -> dict[str, Any]:
    """Build the planned-workout payload that copies a library template.

    Optional adjustments let a caller schedule a variant of the template
    without editing the library item itself:

    * ``interval_reps_override`` — set the number of reps in the template's main
      interval set (e.g. schedule 6×3min from a 5×3min template).
    * ``endurance_minutes_override`` — set the total planned duration in minutes,
      scaling the work portion while keeping warm-up/cool-down fixed.
    * ``description_override`` — replace the template's description text.

    When an interval/duration adjustment changes the structure, ``begin``/
    ``end``, the preview polyline, ``totalTimePlanned`` and (proportionally,
    since IF is unchanged) ``tssPlanned`` are recomputed to match.
    """
    sport_id = item.get("workoutTypeId")
    structure = item.get("structure")
    total_planned = item.get("totalTimePlanned")
    tss_planned = item.get("tssPlanned")

    if structure and (
        interval_reps_override is not None or endurance_minutes_override is not None
    ):
        structure, total_seconds = _apply_structure_overrides(
            structure,
            interval_reps=interval_reps_override,
            endurance_minutes=endurance_minutes_override,
        )
        if total_seconds:
            new_hours = round(total_seconds / 3600, 4)
            # Keep planned TSS consistent: at unchanged intensity, TSS scales
            # linearly with duration.
            if (
                isinstance(tss_planned, (int, float))
                and isinstance(total_planned, (int, float))
                and total_planned > 0
            ):
                tss_planned = round(tss_planned * (new_hours / total_planned), 1)
            total_planned = new_hours

    description = item.get("description")
    if description_override is not None and description_override.strip():
        description = description_override

    payload: dict[str, Any] = {
        "athleteId": athlete_id,
        "workoutDay": f"{date}T00:00:00",
        "workoutTypeFamilyId": sport_id,
        "workoutTypeValueId": sport_id,
        "title": item.get("itemName"),
        "totalTimePlanned": total_planned,
        "tssPlanned": tss_planned,
        "ifPlanned": item.get("ifPlanned"),
        "distancePlanned": item.get("distancePlanned"),
        "elevationGainPlanned": item.get("elevationGainPlanned"),
        "caloriesPlanned": item.get("caloriesPlanned"),
        "description": description,
        "coachComments": item.get("coachComments"),
    }
    if item.get("workoutSubTypeId") is not None:
        payload["workoutSubTypeId"] = item["workoutSubTypeId"]
    if structure:
        # Calendar workouts carry structure as a JSON string
        payload["structure"] = json.dumps(structure)
    return payload


async def tp_schedule_library_workout(
    library_id: str,
    item_id: str,
    date: str,
    athletes: list[str] | None = None,
    description_override: str | None = None,
    interval_reps_override: int | None = None,
    endurance_minutes_override: float | None = None,
) -> dict[str, Any]:
    """Schedule a library template to a calendar date.

    Copies the template into a planned workout (title, structure, planned
    metrics, description). The native ``addworkoutfromlibraryitem`` command
    endpoint returns HTTP 500 for every payload shape, so this mirrors what
    the TP web app effectively does when a template is dragged onto the
    calendar.

    Args:
        library_id: Library ID.
        item_id: Library item ID.
        date: Target date (YYYY-MM-DD).
        athletes: Optional list of athlete names or IDs (coach accounts) to
            schedule the same template to several athletes in one call.
            Mutually exclusive with the ``athlete`` targeting parameter.
        description_override: Optional text that replaces the template's
            description on the scheduled workout (the template itself is left
            unchanged).
        interval_reps_override: Optional number of reps for the template's main
            interval set, e.g. schedule 6 reps from a 5-rep template. Ignored
            for templates with no repetition structure.
        endurance_minutes_override: Optional total planned duration in minutes.
            The work portion is scaled to hit it while warm-up/cool-down stay
            fixed; ``totalTimePlanned``/``tssPlanned`` are recomputed to match.

    Returns:
        Dict with confirmation (including new workout_id) or error. In bulk
        mode, a ``scheduled`` list plus per-athlete ``errors``; ``isError``
        is set only when EVERY athlete failed.
    """
    try:
        lib_validated = WorkoutIdInput(workout_id=library_id)
        item_validated = WorkoutIdInput(workout_id=item_id)
    except (ValidationError, ValueError) as e:
        msg = format_validation_error(e) if isinstance(e, ValidationError) else str(e)
        return {
            "isError": True,
            "error_code": "VALIDATION_ERROR",
            "message": msg,
        }

    try:
        from datetime import date as date_type

        date_type.fromisoformat(date)
    except ValueError:
        return {
            "isError": True,
            "error_code": "VALIDATION_ERROR",
            "message": f"Invalid date: {date}",
        }

    if interval_reps_override is not None and (
        not isinstance(interval_reps_override, int)
        or isinstance(interval_reps_override, bool)
        or interval_reps_override <= 0
    ):
        return {
            "isError": True,
            "error_code": "VALIDATION_ERROR",
            "message": "interval_reps_override must be a positive integer.",
        }

    if endurance_minutes_override is not None and (
        not isinstance(endurance_minutes_override, (int, float))
        or isinstance(endurance_minutes_override, bool)
        or endurance_minutes_override <= 0
    ):
        return {
            "isError": True,
            "error_code": "VALIDATION_ERROR",
            "message": "endurance_minutes_override must be a positive number of minutes.",
        }

    overrides: dict[str, Any] = {
        "description_override": description_override,
        "interval_reps_override": interval_reps_override,
        "endurance_minutes_override": endurance_minutes_override,
    }

    if athletes is not None:
        if athlete_override.get() is not None:
            return {
                "isError": True,
                "error_code": "VALIDATION_ERROR",
                "message": (
                    "Pass either 'athlete' (single target) or 'athletes' "
                    "(bulk), not both."
                ),
            }
        if (
            not isinstance(athletes, (list, tuple))
            or not athletes
            or not all(isinstance(a, (str, int)) and str(a).strip() for a in athletes)
        ):
            return {
                "isError": True,
                "error_code": "VALIDATION_ERROR",
                "message": "athletes must be a non-empty list of athlete names or IDs.",
            }

    async with TPClient() as client:
        athlete_id: int | None = None
        if athletes is None:
            athlete_id = await client.ensure_athlete_id()
            if not athlete_id:
                return {
                    "isError": True,
                    "error_code": "AUTH_INVALID",
                    "message": "Could not get athlete ID. Re-authenticate.",
                }

        # Fetch the template to copy
        items_endpoint = f"/exerciselibrary/v2/libraries/{lib_validated.workout_id}/items"
        items_response = await client.get(items_endpoint)

        if items_response.is_error:
            return {
                "isError": True,
                "error_code": items_response.error_code.value
                if items_response.error_code
                else "API_ERROR",
                "message": items_response.message,
            }

        items = items_response.data if isinstance(items_response.data, list) else []
        item = next(
            (
                i
                for i in items
                if i.get("exerciseLibraryItemId", i.get("id")) == item_validated.workout_id
            ),
            None,
        )
        if item is None:
            return {
                "isError": True,
                "error_code": "NOT_FOUND",
                "message": (
                    f"Item {item_validated.workout_id} not found in "
                    f"library {lib_validated.workout_id}."
                ),
            }

        if athletes is not None:
            return await _schedule_item_bulk(client, item, date, athletes, overrides)

        assert athlete_id is not None  # resolved above in the single-athlete path
        endpoint = f"/fitness/v6/athletes/{athlete_id}/workouts"
        response = await client.post(
            endpoint,
            json=_template_workout_payload(item, date, athlete_id, **overrides),
        )

        if response.is_error:
            return {
                "isError": True,
                "error_code": response.error_code.value if response.error_code else "API_ERROR",
                "message": response.message,
            }

        workout_id = None
        if isinstance(response.data, dict):
            workout_id = response.data.get("workoutId")

        return {
            "success": True,
            "message": f"Library workout scheduled for {date}.",
            "date": date,
            "workout_id": workout_id,
            "title": item.get("itemName"),
        }


async def _schedule_item_bulk(
    client: TPClient,
    item: dict[str, Any],
    date: str,
    athletes: list[str],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Schedule one library template to several athletes, sequentially.

    Each entry is resolved exactly as the single ``athlete`` targeting
    parameter would be (name or ID, via the athlete_override context var).
    Follows the groups-tools partial-failure pattern: per-athlete ``errors``,
    ``isError`` only when EVERY athlete failed. Any ``overrides`` (description /
    interval reps / duration) are applied identically to every athlete.
    """
    overrides = overrides or {}
    scheduled: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for entry in athletes:
        target = str(entry).strip()
        token = athlete_override.set(target)
        try:
            athlete_id = await client.ensure_athlete_id()
        except ValueError as e:
            # Ambiguous athlete name — ensure_athlete_id lists the matches.
            errors.append({"athlete": target, "message": str(e)})
            continue
        finally:
            athlete_override.reset(token)

        if not athlete_id:
            errors.append({
                "athlete": target,
                "message": f"Could not resolve athlete {target!r} in your roster.",
            })
            continue

        endpoint = f"/fitness/v6/athletes/{athlete_id}/workouts"
        response = await client.post(
            endpoint,
            json=_template_workout_payload(item, date, athlete_id, **overrides),
        )
        if response.is_error:
            errors.append({
                "athlete": target,
                "athlete_id": athlete_id,
                "message": response.message,
            })
        else:
            workout_id = None
            if isinstance(response.data, dict):
                workout_id = response.data.get("workoutId")
            scheduled.append({
                "athlete": target,
                "athlete_id": athlete_id,
                "workout_id": workout_id,
            })

    result: dict[str, Any] = {
        "date": date,
        "title": item.get("itemName"),
        "scheduled": scheduled,
        "errors": errors,
        "message": (
            f"Scheduled for {len(scheduled)} of {len(athletes)} athlete(s) on {date}."
        ),
    }
    if errors and not scheduled:
        result["isError"] = True
        result["error_code"] = "API_ERROR"
        result["message"] = (
            f"None of the {len(errors)} athlete(s) could be scheduled; "
            "see errors for per-athlete detail."
        )
    return result
