"""Tests for HTTP client, including throttling and athlete ID caching."""

import asyncio
import time
from unittest.mock import AsyncMock

import httpx
import pytest

from tp_mcp.client import http as http_mod
from tp_mcp.client.http import MIN_REQUEST_INTERVAL, APIResponse, TPClient


class TestThrottling:
    """Tests for request throttling."""

    @pytest.mark.asyncio
    async def test_throttle_enforces_minimum_interval(self):
        """Throttle should enforce minimum interval between requests."""
        client = TPClient()

        # First call should not block
        start = time.monotonic()
        await client._throttle()
        first_duration = time.monotonic() - start
        assert first_duration < 0.05  # Should be nearly instant

        # Immediate second call should be delayed
        start = time.monotonic()
        await client._throttle()
        second_duration = time.monotonic() - start
        assert second_duration >= MIN_REQUEST_INTERVAL * 0.9  # Allow 10% tolerance

    @pytest.mark.asyncio
    async def test_throttle_no_delay_when_spaced(self):
        """Throttle should not delay when requests are naturally spaced."""
        client = TPClient()

        await client._throttle()

        # Wait longer than the interval
        import asyncio

        await asyncio.sleep(MIN_REQUEST_INTERVAL + 0.05)

        # Next call should not block
        start = time.monotonic()
        await client._throttle()
        duration = time.monotonic() - start
        assert duration < 0.05  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_throttle_multiple_rapid_calls(self):
        """Multiple rapid calls should each be throttled."""
        client = TPClient()

        start = time.monotonic()

        # Make 4 rapid throttle calls
        for _ in range(4):
            await client._throttle()

        total_duration = time.monotonic() - start

        # Should take at least 3 * MIN_REQUEST_INTERVAL (first is instant, next 3 are throttled)
        expected_min = MIN_REQUEST_INTERVAL * 3 * 0.9  # 10% tolerance
        assert total_duration >= expected_min

    @pytest.mark.asyncio
    async def test_client_init_sets_last_request_time(self):
        """Client should initialize last request time to 0."""
        client = TPClient()
        assert client._last_request_time == 0.0


class TestEnsureAthleteId:
    """Tests for athlete ID caching via ensure_athlete_id."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Reset class-level caches between tests."""
        TPClient._cached_athlete_id = None
        TPClient._cached_user_data = None
        TPClient._shared_token_cache = None
        yield
        TPClient._cached_athlete_id = None
        TPClient._cached_user_data = None
        TPClient._shared_token_cache = None

    @pytest.mark.asyncio
    async def test_returns_cached_class_level_value(self):
        """Should return class-level cached athlete ID without API call."""
        TPClient._cached_athlete_id = 999
        client = TPClient()
        client.get = AsyncMock()  # should not be called

        result = await client.ensure_athlete_id()

        assert result == 999
        client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetches_from_api_and_caches(self):
        """Should fetch athlete ID from API and cache at class level."""
        client = TPClient()
        client.get = AsyncMock(return_value=APIResponse(success=True, data={"user": {"personId": 42}}))

        result = await client.ensure_athlete_id()

        assert result == 42
        assert TPClient._cached_athlete_id == 42
        assert client.athlete_id == 42

    @pytest.mark.asyncio
    async def test_falls_back_to_athletes_array(self):
        """Should use athletes[0].athleteId when personId is missing."""
        client = TPClient()
        client.get = AsyncMock(
            return_value=APIResponse(
                success=True,
                data={"user": {"athletes": [{"athleteId": 77}]}},
            )
        )

        result = await client.ensure_athlete_id()

        assert result == 77
        assert TPClient._cached_athlete_id == 77

    @pytest.mark.asyncio
    async def test_returns_none_on_api_failure(self):
        """Should return None when API call fails (no caching)."""
        client = TPClient()
        client.get = AsyncMock(return_value=APIResponse(success=False, message="Auth failed"))

        result = await client.ensure_athlete_id()

        assert result is None
        assert TPClient._cached_athlete_id is None

    @pytest.mark.asyncio
    async def test_class_cache_persists_across_instances(self):
        """Class-level cache should persist across TPClient instances."""
        client1 = TPClient()
        client1.get = AsyncMock(return_value=APIResponse(success=True, data={"user": {"personId": 123}}))
        await client1.ensure_athlete_id()

        # Second instance should use cached value without API call
        client2 = TPClient()
        client2.get = AsyncMock()

        result = await client2.ensure_athlete_id()

        assert result == 123
        client2.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_instance_athlete_id_if_set(self):
        """Should return instance-level athlete_id if already set."""
        client = TPClient()
        client.athlete_id = 555
        client.get = AsyncMock()

        result = await client.ensure_athlete_id()

        assert result == 555
        client.get.assert_not_called()


class TestSharedTokenCache:
    """Tests for shared TokenCache across TPClient instances."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        """Reset shared token cache between tests."""
        TPClient._shared_token_cache = None
        yield
        TPClient._shared_token_cache = None

    def test_token_cache_shared_across_instances(self):
        """Multiple TPClient instances should share the same TokenCache."""
        client1 = TPClient()
        client2 = TPClient()
        assert client1._token_cache is client2._token_cache

    def test_token_cache_lazily_created(self):
        """Shared cache should be None until first TPClient is created."""
        assert TPClient._shared_token_cache is None
        TPClient()
        assert TPClient._shared_token_cache is not None


class TestHandleResponse:
    """Tests for HTTP response handling."""

    def test_204_is_success(self):
        """204 No Content responses should be treated as successful writes."""
        client = TPClient()

        response = httpx.Response(status_code=204)

        result = client._handle_response(response)

        assert result.success is True
        assert result.data is None


class TestForbiddenEndpoints:
    """Safeguard: destructive plan commands are hard-blocked at the client (the
    native applyplan + unapply emptied a published plan's template, 2026-06-17)."""

    def test_helper_matches_applyplan_only(self):
        from tp_mcp.client.http import _is_forbidden
        assert _is_forbidden("/plans/v1/commands/applyplan")
        assert _is_forbidden("/plans/v1/commands/applyplan?x=1")
        # the synthetic-apply read paths + status poll stay allowed
        assert not _is_forbidden("/plans/v1/plans/163992/workouts/2018-12-17/2019-04-10")
        assert not _is_forbidden("/plans/v1/appliedplans/applyPlanStatus")
        assert not _is_forbidden("/fitness/v6/athletes/123/workouts")

    @pytest.mark.asyncio
    async def test_applyplan_blocked_with_no_network_call(self):
        from tp_mcp.client.http import ErrorCode
        client = TPClient()
        # Sent via post() → _request(): blocked before any auth/HTTP happens.
        r = await client.post("/plans/v1/commands/applyplan", json=[{"planId": 1}])
        assert r.is_error and r.error_code == ErrorCode.FORBIDDEN_ENDPOINT
        # get_raw() is guarded too.
        rr = await client.get_raw("/plans/v1/commands/applyplan")
        assert rr.is_error and rr.error_code == ErrorCode.FORBIDDEN_ENDPOINT


class TestResponseCache:
    """Tests for the GET TTL cache + in-flight coalescing (options 1 & 2)."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self, monkeypatch):
        """Fresh response cache and caching enabled for each test."""
        monkeypatch.setattr(http_mod, "CACHE_DISABLED", False)
        TPClient._shared_response_cache = None
        yield
        TPClient._shared_response_cache = None

    @pytest.mark.asyncio
    async def test_repeat_get_served_from_cache(self):
        """A second identical GET is a HIT and does not re-hit the API."""
        client = TPClient()
        client._request = AsyncMock(
            return_value=APIResponse(success=True, data={"v": 1})
        )

        first = await client.get("/exerciselibrary/v2/libraries/3820613/items")
        second = await client.get("/exerciselibrary/v2/libraries/3820613/items")

        assert first.data == {"v": 1}
        assert second.data == {"v": 1}
        client._request.assert_awaited_once()  # only the MISS hit the API

    @pytest.mark.asyncio
    async def test_distinct_params_are_separate_keys(self):
        """Different query params are cached independently."""
        client = TPClient()
        client._request = AsyncMock(
            return_value=APIResponse(success=True, data={"ok": True})
        )

        await client.get("/x", params={"a": 1})
        await client.get("/x", params={"a": 2})

        assert client._request.await_count == 2

    @pytest.mark.asyncio
    async def test_expired_entry_refetches(self):
        """An entry past its TTL is a MISS and refetches."""
        client = TPClient()
        client._request = AsyncMock(
            return_value=APIResponse(success=True, data={"v": 1})
        )

        await client.get("/workouts/2026-09-09/2026-11-15")
        # Force the stored entry to have already expired.
        cache = TPClient._get_response_cache()
        for entry in cache._entries.values():
            entry.expires_at = time.monotonic() - 1

        await client.get("/workouts/2026-09-09/2026-11-15")
        assert client._request.await_count == 2

    @pytest.mark.asyncio
    async def test_failed_response_not_cached(self):
        """Errors are never cached; the next call retries."""
        client = TPClient()
        client._request = AsyncMock(
            return_value=APIResponse(success=False, message="boom")
        )

        await client.get("/settings")
        await client.get("/settings")
        assert client._request.await_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_gets_are_coalesced(self):
        """Concurrent identical GETs share a single in-flight request."""
        client = TPClient()
        release = asyncio.Event()
        calls = 0

        async def slow_request(method, endpoint, **kwargs):
            nonlocal calls
            calls += 1
            await release.wait()
            return APIResponse(success=True, data={"calls": calls})

        client._request = slow_request

        t1 = asyncio.create_task(client.get("/users/v3/user"))
        t2 = asyncio.create_task(client.get("/users/v3/user"))
        await asyncio.sleep(0.01)  # let both reach the in-flight point
        release.set()
        r1, r2 = await asyncio.gather(t1, t2)

        assert calls == 1  # coalesced into one API call
        assert r1.data == r2.data == {"calls": 1}

    @pytest.mark.asyncio
    async def test_write_invalidates_same_athlete_cache(self):
        """A write to an athlete's path drops that athlete's cached GETs."""
        client = TPClient()
        client._request = AsyncMock(
            return_value=APIResponse(success=True, data={"v": 1})
        )

        endpoint = "/fitness/v6/athletes/1402240/workouts/2026-09-09/2026-11-15"
        await client.get(endpoint)  # MISS → cached
        await client.get(endpoint)  # HIT
        assert client._request.await_count == 1

        # Writing a workout for the same athlete invalidates the cached read.
        await client.post("/fitness/v6/athletes/1402240/workouts", json={})
        await client.get(endpoint)  # MISS again
        assert client._request.await_count == 3  # 1 read + 1 write + 1 refetch

    @pytest.mark.asyncio
    async def test_write_leaves_other_athlete_cache_intact(self):
        """Invalidation is scoped to the written athlete only."""
        client = TPClient()
        client._request = AsyncMock(
            return_value=APIResponse(success=True, data={"v": 1})
        )

        other = "/fitness/v6/athletes/999/workouts/2026-09-09/2026-11-15"
        await client.get(other)  # cache athlete 999
        await client.post("/fitness/v6/athletes/1402240/workouts", json={})
        await client.get(other)  # still a HIT — untouched
        assert client._request.await_count == 2  # 1 read + 1 write, no refetch

    @pytest.mark.asyncio
    async def test_cache_disabled_bypasses(self, monkeypatch):
        """With caching disabled every GET hits the API."""
        monkeypatch.setattr(http_mod, "CACHE_DISABLED", True)
        client = TPClient()
        client._request = AsyncMock(
            return_value=APIResponse(success=True, data={"v": 1})
        )

        await client.get("/settings")
        await client.get("/settings")
        assert client._request.await_count == 2

    @pytest.mark.asyncio
    async def test_inflight_read_not_cached_when_write_lands_midflight(self):
        """A GET in flight when a write invalidates it must not cache its stale
        result, and a reader arriving after the write must refetch."""
        client = TPClient()
        endpoint = "/fitness/v6/athletes/1402240/workouts/2026-09-09/2026-11-15"
        release = asyncio.Event()
        reads = 0

        async def gated_read(method, ep, **kwargs):
            nonlocal reads
            reads += 1
            if reads == 1:
                await release.wait()  # hold the first read in flight
            return APIResponse(success=True, data={"read": reads})

        client._request = gated_read

        leader = asyncio.create_task(client.get(endpoint))
        await asyncio.sleep(0.01)  # ensure the leader is in flight

        # A write for the same athlete lands while the read is in flight.
        client._invalidate_cache_after_write(
            "/fitness/v6/athletes/1402240/workouts"
        )

        release.set()
        await leader

        # The pre-write read must not have been cached: a fresh GET refetches.
        result = await client.get(endpoint)
        assert reads == 2
        assert result.data == {"read": 2}

    @pytest.mark.asyncio
    async def test_write_bumps_generation(self):
        """Each write advances the invalidation generation counter."""
        client = TPClient()
        cache = TPClient._get_response_cache()
        start = cache.generation
        client._invalidate_cache_after_write("/fitness/v6/athletes/1402240/workouts")
        assert cache.generation == start + 1


