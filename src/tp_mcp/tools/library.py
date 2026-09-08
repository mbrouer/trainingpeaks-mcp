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


def _scale_work_steps(blocks: list[dict[str, Any]], target_seconds: int) -> None:
    """Scale the ``active``/``other`` work steps to hit ``target_seconds``,
    treating warm-up, cool-down and rest steps as fixed anchors. When nothing is
    marked as work (e.g. a single untyped endurance step), every step is scaled
    uniformly instead. Used as the fallback when a workout has no rest steps to
    flex (see ``_set_total_duration``)."""
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
        length["value"] = _round_to_half_minute(
            (length.get("value", 0) or 0) * factor
        )
        length.setdefault("unit", "second")


# Minimum duration (seconds) protected for the opening warm-up AND the closing
# cool-down block (the two edge blocks).
_EDGE_BLOCK_MIN_SECONDS = 600  # 10 min


def _round_to_half_minute(seconds: float) -> int:
    """Snap an adjusted duration to a clean half/whole minute so the plan never
    shows odd times like 10:27. A minute or more rounds to the nearest 30s
    (e.g. 627 → 630 = 10:30); sub-minute leftovers keep their whole-second value
    so short recovery bits are not inflated up to 30s. Only the flexed/scaled
    durations are rounded — untouched interval efforts keep their exact length."""
    if seconds < 60:
        return max(int(round(seconds)), 1)
    return int(round(seconds / 30)) * 30


def _scale_flex_to_budget(
    flex: list[tuple[dict[str, Any], int, int]], budget: float
) -> None:
    """Distribute ``budget`` seconds across the flexible steps, scaling them by a
    single factor BUT never letting any step drop below its own floor. Uses
    water-filling: on each pass, steps whose scaled length would fall under their
    floor are pinned at the floor and removed from the pool, and the remaining
    budget is re-shared across the rest. This keeps the reduction proportional
    while honouring per-step minimums (e.g. the 10-min opening warm-up / closing
    cool-down), instead of one big pool crushing every flexible step toward 1s."""
    entries: list[dict[str, Any]] = []
    for step, reps, floor in flex:
        dur = int((step.get("length") or {}).get("value", 0) or 0)
        entries.append(
            {
                "step": step,
                "reps": reps,
                "floor": max(floor, 1),
                "dur": dur,
                "value": None,
            }
        )

    remaining = budget
    while True:
        active = [x for x in entries if x["value"] is None]
        scalable = sum(x["dur"] * x["reps"] for x in active)
        if not active or scalable <= 0:
            break
        factor = remaining / scalable
        # Pin every step that would fall below its floor at this factor, then
        # retry with the leftover budget spread across the still-flexible steps.
        pinned = [x for x in active if x["dur"] * factor < x["floor"]]
        if not pinned:
            for x in active:
                x["value"] = max(int(round(x["dur"] * factor)), 1)
            break
        for x in pinned:
            x["value"] = x["floor"]
            remaining -= x["floor"] * x["reps"]

    for x in entries:
        length = x["step"].setdefault("length", {})
        length["value"] = _round_to_half_minute(
            x["value"] if x["value"] is not None else x["floor"]
        )
        length.setdefault("unit", "second")


def _set_total_duration(blocks: list[dict[str, Any]], target_seconds: int) -> None:
    """Scale the workout so its total planned time equals ``target_seconds``.

    The MAIN INTERVALS stay fixed. Concretely, EVERY step inside a ``repetition``
    block — both the hard effort AND the short in-set recovery between reps
    (e.g. ``5×(3:40 VT-2 + 0:20 VO₂)``) — is treated as an untouchable interval,
    so interval timing is never stretched or squashed (a 20s VO₂ stays 20s, not
    18s). The duration change is absorbed only by the flexible pool: standalone
    ``rest`` recovery blocks between sets, standalone steady/endurance steps, and
    mid-workout "easy" recovery blocks. This matches the coach's intent: a "make
    it longer/shorter" tweak stretches the recovery/endurance, never the efforts.

    Only a LEADING ``warmUp`` block and a TRAILING ``coolDown`` block are treated
    as fixed anchors (the true warm-up / cool-down). A low-intensity block in the
    middle of the workout — even if TrainingPeaks tags it ``warmUp``/``coolDown``
    — is really between-set recovery, so it flexes with the rest of the pool.

    The flexible pool is shrunk/grown by water-filling (``_scale_flex_to_budget``)
    so the reduction spreads proportionally while honouring per-step minimums. In
    particular the FIRST and LAST blocks keep a 10-minute floor so the opening
    warm-up and closing cool-down are never squashed to a few seconds. Scaled
    durations snap to clean half/whole minutes.

    Falls back to scaling the work steps (``_scale_work_steps``) only when there
    is no flexible pool at all — a workout made purely of repetition intervals
    and/or warm-up/cool-down, where nothing else can move."""
    flex: list[tuple[dict[str, Any], int, int]] = []  # (step, reps, floor)
    interval_secs = 0  # every step inside repetition blocks — fixed
    anchor_secs = 0  # leading warm-up / trailing cool-down — fixed
    last_idx = len(blocks) - 1
    for idx, b in enumerate(blocks):
        reps = _block_reps(b)
        is_repetition = b.get("type") == "repetition"
        is_edge_block = idx == 0 or idx == last_idx
        for s in b.get("steps", []):
            dur = int((s.get("length") or {}).get("value", 0) or 0)
            cls = s.get("intensityClass")
            is_warm_cool = cls in ("warmUp", "coolDown")
            if is_repetition:
                # Main interval set — EVERY step (hard effort AND the short in-set
                # recovery between reps, e.g. the 20s VO₂) stays fixed.
                interval_secs += dur * reps
            elif is_warm_cool and is_edge_block:
                # True warm-up / cool-down at the very start/end — never scaled.
                anchor_secs += dur * reps
            elif cls == "rest":
                # Standalone recovery block between sets — flexible.
                flex.append((s, reps, 1))
            else:
                # Standalone steady/endurance/easy step — flexible. Protect the
                # opening warm-up AND closing cool-down (the edge blocks) with a
                # 10-minute floor (capped at the step's original length so we
                # never invent time for an already-short block).
                floor = min(dur, _EDGE_BLOCK_MIN_SECONDS) if is_edge_block else 1
                flex.append((s, reps, floor))

    # Preferred: keep intervals + true warm-up/cool-down fixed, absorb the delta
    # across the flexible pool via water-filling (proportional, floor-aware).
    if flex:
        budget = target_seconds - interval_secs - anchor_secs
        _scale_flex_to_budget(flex, budget)
        return

    # Fallback: nothing flexible (pure intervals and/or warm-up/cool-down) —
    # scale the work steps as a last resort so a target can still be approached.
    _scale_work_steps(blocks, target_seconds)


def _is_recovery_block(block: dict[str, Any]) -> bool:
    """True when ``block`` is a standalone recovery block: a non-``repetition``
    block whose every step is low-intensity recovery — a ``rest`` step, or a
    ``warmUp``/``coolDown`` step (mid-workout, these are really between-set
    "easy" recovery, matching how ``_set_total_duration`` classifies non-edge
    warm-up/cool-down blocks). A block containing any main-effort
    (``active``/``other``) step is NOT recovery. Used to also drop the orphaned
    recovery that follows an interval set removed via a 0 rep count."""
    if block.get("type") == "repetition":
        return False
    steps = block.get("steps") or []
    if not steps:
        return False
    return all(
        s.get("intensityClass") in ("rest", "warmUp", "coolDown") for s in steps
    )


def _set_block_reps(
    blocks: list[dict[str, Any]], reps_by_ordinal: dict[int, int]
) -> None:
    """Rewrite the rep count (``length.value``) of each targeted repetition
    block. ``reps_by_ordinal`` is keyed by repetition-block ordinal (0-based,
    counting only ``type:"repetition"`` blocks in structure order). A rep count
    of 0 REMOVES the whole interval set (that repetition block) from the
    structure, along with the standalone recovery block immediately following it
    (unless that recovery is the final block, i.e. a true trailing cool-down,
    which is preserved). Non-repetition blocks and ordinals not present in the
    map are left untouched. Mirrors the TS ``setBlockReps`` helper."""
    ord_ = 0
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.get("type") == "repetition":
            v = reps_by_ordinal.get(ord_)
            if isinstance(v, int) and not isinstance(v, bool):
                if v <= 0:
                    # Rep count 0 → drop the entire interval set...
                    blocks.pop(i)
                    # ...and the orphaned recovery block after it, as long as it
                    # is not the final block (a trailing cool-down stays put).
                    if i < len(blocks) - 1 and _is_recovery_block(blocks[i]):
                        blocks.pop(i)
                    ord_ += 1
                    continue  # list shifted; don't advance ``i``
                length = b.setdefault("length", {})
                length["value"] = int(v)
                length.setdefault("unit", "repetition")
            ord_ += 1
        i += 1


def _apply_structure_overrides(
    structure: Any,
    *,
    endurance_minutes: float | None = None,
    interval_reps: dict[int, int] | None = None,
) -> tuple[Any, int | None]:
    """Return a COPY of a native template structure with its total duration
    and/or per-block interval reps adjusted, plus the resulting total seconds.

    ``begin``/``end`` offsets and the preview ``polyline`` are recomputed so the
    scheduled workout renders correctly. Non-native or empty structures (and the
    no-override case) are returned untouched with ``total_seconds=None``.

    Per-block interval reps are adjustable via ``interval_reps`` (a map of
    repetition-block ordinal → new rep count). Rep overrides are applied BEFORE
    duration scaling so the duration budget accounts for the new rep count. The
    template's stored structure supplies the defaults; overrides rewrite the
    scheduled copy only."""
    if endurance_minutes is None and not interval_reps:
        return structure, None
    if not isinstance(structure, dict):
        return structure, None
    blocks = structure.get("structure")
    if not isinstance(blocks, list) or not blocks:
        return structure, None

    out = dict(structure)
    blocks = copy.deepcopy(blocks)
    out["structure"] = blocks

    if interval_reps:
        _set_block_reps(blocks, interval_reps)
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
    endurance_minutes_override: float | None = None,
    interval_reps_override: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Build the planned-workout payload that copies a library template.

    Optional adjustments let a caller schedule a variant of the template
    without editing the library item itself:

    * ``endurance_minutes_override`` — set the total planned duration in minutes,
      absorbing the change in the rest periods while keeping the main interval
      work and warm-up/cool-down fixed (falls back to scaling the work when the
      workout has no rest to flex).
    * ``interval_reps_override`` — set new rep counts per repetition block (a map
      of block ordinal → rep count). Applied before duration scaling.
    * ``description_override`` — replace the template's description text.

    When a duration and/or interval-rep adjustment changes the structure,
    ``begin``/``end``, the preview polyline, ``totalTimePlanned`` and
    (proportionally, since IF is unchanged) ``tssPlanned`` are recomputed to
    match.
    """
    sport_id = item.get("workoutTypeId")
    structure = item.get("structure")
    total_planned = item.get("totalTimePlanned")
    tss_planned = item.get("tssPlanned")

    if structure and (
        endurance_minutes_override is not None or interval_reps_override
    ):
        structure, total_seconds = _apply_structure_overrides(
            structure,
            endurance_minutes=endurance_minutes_override,
            interval_reps=interval_reps_override,
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
    endurance_minutes_override: float | None = None,
    interval_reps_override: dict[str, int] | None = None,
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
        endurance_minutes_override: Optional total planned duration in minutes.
            The rest periods are stretched/shrunk to hit it while the main
            interval work and warm-up/cool-down stay fixed (falling back to
            scaling the work when there is no rest to flex);
            ``totalTimePlanned``/``tssPlanned`` are recomputed to match.
        interval_reps_override: Optional map of repetition-block ordinal
            (0-based, counting only repetition blocks in structure order) to a
            new rep count (0..100, where 0 removes that interval set). Only
            listed blocks change; unlisted blocks
            keep the template's reps. Combinable with
            ``endurance_minutes_override`` (reps are applied first);
            ``totalTimePlanned``/``tssPlanned`` are recomputed to match. The
            template itself is left unchanged.

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

    normalized_reps: dict[int, int] | None = None
    if interval_reps_override is not None:
        reps_err = {
            "isError": True,
            "error_code": "VALIDATION_ERROR",
            "message": (
                "interval_reps_override must map repetition-block indices to "
                "integer rep counts between 0 and 100 (0 removes the set)."
            ),
        }
        if not isinstance(interval_reps_override, dict):
            return reps_err
        normalized_reps = {}
        for key, value in interval_reps_override.items():
            key_str = str(key)
            if not key_str.isdigit():
                return reps_err
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value > 100
            ):
                return reps_err
            normalized_reps[int(key_str)] = int(value)

    overrides: dict[str, Any] = {
        "description_override": description_override,
        "endurance_minutes_override": endurance_minutes_override,
        "interval_reps_override": normalized_reps,
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
