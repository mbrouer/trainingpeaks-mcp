"""HTTP client wrapper for TrainingPeaks API."""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

from tp_mcp.auth import get_credential

logger = logging.getLogger("tp-mcp")

TP_API_BASE = "https://tpapi.trainingpeaks.com"
DEFAULT_TIMEOUT = 30.0
MIN_REQUEST_INTERVAL = 0.15  # 150ms between requests to avoid rate limiting
TOKEN_ENDPOINT = "/users/v3/token"
TOKEN_REFRESH_BUFFER = 60  # Refresh token 60s before expiry

# --- Read-through response cache -------------------------------------------
# Set TP_MCP_CACHE_DISABLED=1 (or true/yes) to bypass GET caching entirely.
CACHE_DISABLED = os.getenv("TP_MCP_CACHE_DISABLED", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Per-endpoint freshness windows. First matching substring wins; the list is
# ordered most-specific first. Read-only reference data (libraries) is held far
# longer than calendar data (workouts/events) that a coach actively edits.
_CACHE_TTL_RULES: tuple[tuple[str, float], ...] = (
    ("/exerciselibrary/", 3600.0),  # workout template libraries — rarely change
    ("/settings", 1800.0),          # athlete settings/zones
    ("/workouts/", 180.0),          # planned/completed workouts
    ("/events", 180.0),             # events / next / focus event
    ("/users/v3/user", 3600.0),     # coach + roster identity
)
_DEFAULT_CACHE_TTL = 120.0


def _cache_ttl_for(endpoint: str) -> float:
    """TTL (seconds) for a GET endpoint, chosen by path substring."""
    for needle, ttl in _CACHE_TTL_RULES:
        if needle in endpoint:
            return ttl
    return _DEFAULT_CACHE_TTL


class APIError(Exception):
    """Base exception for API errors."""

    pass


class AuthenticationError(APIError):
    """Authentication failed or expired."""

    pass


class NotFoundError(APIError):
    """Resource not found."""

    pass


class RateLimitError(APIError):
    """Rate limit exceeded."""

    pass


class ErrorCode(Enum):
    """Error codes for API responses."""

    AUTH_EXPIRED = "AUTH_EXPIRED"
    AUTH_INVALID = "AUTH_INVALID"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    PREMIUM_REQUIRED = "PREMIUM_REQUIRED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    API_ERROR = "API_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    FORBIDDEN_ENDPOINT = "FORBIDDEN_ENDPOINT"


# Hard-blocked endpoints — destructive operations the connector must NEVER issue,
# regardless of caller (tool, probe, or future code). DEFENSE-IN-DEPTH safeguard.
#
# /plans/v1/commands/applyplan — the NATIVE training-plan apply command. Live
# incident 2026-06-17: applying a published plan via this command produced a
# degenerate application that associated the SOURCE plan-template's own workouts
# with the application; a subsequent Unapply ("remove all associated workouts")
# then DELETED the template's 139 workouts, emptying a published Plan-Store plan.
# The connector applies plans with a SYNTHETIC copy (tp_apply_training_plan), which
# never calls this command, so blocking it loses no functionality.
_FORBIDDEN_ENDPOINTS: tuple[str, ...] = ("/plans/v1/commands/applyplan",)


def _is_forbidden(endpoint: str) -> bool:
    return any(bad in endpoint for bad in _FORBIDDEN_ENDPOINTS)


@dataclass
class APIResponse:
    """Wrapper for API responses."""

    success: bool
    data: dict[str, Any] | list[Any] | None = None
    error_code: ErrorCode | None = None
    message: str = ""

    @property
    def is_error(self) -> bool:
        """Check if response is an error."""
        return not self.success


@dataclass
class RawResponse:
    """Wrapper for raw binary API responses."""

    success: bool
    content: bytes = field(default_factory=bytes)
    content_type: str | None = None
    content_disposition: str | None = None
    error_code: ErrorCode | None = None
    message: str = ""

    @property
    def is_error(self) -> bool:
        """Check if response is an error."""
        return not self.success


@dataclass
class TokenCache:
    """In-memory cache for OAuth access token."""

    access_token: str | None = None
    expires_at: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def is_valid(self, buffer_seconds: int = TOKEN_REFRESH_BUFFER) -> bool:
        """Check if token is valid with buffer before expiry."""
        if not self.access_token:
            return False
        return time.time() < (self.expires_at - buffer_seconds)

    def clear(self) -> None:
        """Clear the cached token."""
        self.access_token = None
        self.expires_at = 0.0


@dataclass
class _CacheEntry:
    """A cached GET response with its expiry."""

    response: "APIResponse"
    expires_at: float


class _ResponseCache:
    """Process-wide TTL cache + in-flight coalescing for idempotent GETs.

    Two problems are solved together:

    * TTL cache (option 1): a repeated GET within its freshness window is served
      from memory instead of hitting TrainingPeaks again. Endpoints embed the
      athlete id in their path (e.g. ``/athletes/1402240/workouts/...``) or are
      coach-token scoped, so the endpoint+params key is inherently per-athlete —
      no cross-athlete leakage.
    * Request coalescing (option 2): concurrent identical GETs that arrive before
      the first response lands share a single in-flight request via an
      ``asyncio.Future`` instead of each stampeding the API.

    Runs single-threaded under asyncio, so no locking is required — there is no
    await between a cache read and the following mutation on the leader path.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _CacheEntry] = {}
        self._inflight: dict[str, asyncio.Future[APIResponse]] = {}
        # Bumped on every write-invalidation. A GET that began before a write
        # landed must NOT cache its (now stale) result, and a reader arriving
        # after the write must NOT coalesce onto that pre-write request.
        self._generation: int = 0

    @property
    def generation(self) -> int:
        """Current invalidation generation (increments on each write)."""
        return self._generation

    @staticmethod
    def make_key(endpoint: str, params: dict[str, Any] | None) -> str:
        """Stable cache key from endpoint + sorted query params."""
        if params:
            items = sorted((str(k), str(v)) for k, v in params.items())
            return endpoint + "?" + "&".join(f"{k}={v}" for k, v in items)
        return endpoint

    def get_fresh(self, key: str) -> "APIResponse | None":
        """Return a non-expired cached response, or None on miss/expiry."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if time.monotonic() >= entry.expires_at:
            self._entries.pop(key, None)
            return None
        return entry.response

    def inflight(self, key: str) -> "asyncio.Future[APIResponse] | None":
        """The pending request for this key, if one is already running."""
        return self._inflight.get(key)

    def begin(self, key: str) -> "asyncio.Future[APIResponse]":
        """Register this caller as the leader for ``key`` and return its Future."""
        fut: asyncio.Future[APIResponse] = asyncio.get_event_loop().create_future()
        self._inflight[key] = fut
        return fut

    def finish(self, key: str, future: asyncio.Future[APIResponse]) -> None:
        """Drop the in-flight marker for ``key`` once the leader is done.

        Only clears the entry when it is still *this* leader's future: a write
        may have detached the key mid-flight and a newer leader may have taken
        its place, which must not be evicted here.
        """
        if self._inflight.get(key) is future:
            self._inflight.pop(key, None)

    def store(self, key: str, response: "APIResponse", ttl: float) -> None:
        """Cache a successful response for ``ttl`` seconds."""
        self._entries[key] = _CacheEntry(
            response=response, expires_at=time.monotonic() + ttl
        )

    def invalidate_for_write(self, endpoint: str) -> int:
        """Drop cached GETs affected by a write to ``endpoint``.

        When the write path carries an ``/athletes/<id>/`` segment, only that
        athlete's cached GETs are dropped; otherwise (e.g. library writes) the
        whole cache is cleared to stay safe. In-flight requests for the affected
        keys are also detached (removed from the coalescing map) so a reader that
        arrives after this write starts a fresh request instead of joining the
        pre-write one. The generation counter is bumped so an in-flight leader
        that began before this write will not cache its stale result. Returns the
        number of cached entries removed for logging.
        """
        self._generation += 1
        match = re.search(r"/athletes/(\d+)/", endpoint)
        if match:
            token = f"/athletes/{match.group(1)}/"
            stale = [k for k in self._entries if token in k]
            for k in stale:
                self._entries.pop(k, None)
            for k in [k for k in self._inflight if token in k]:
                self._inflight.pop(k, None)
            return len(stale)
        count = len(self._entries)
        self._entries.clear()
        self._inflight.clear()
        return count


class TPClient:
    """Async HTTP client for TrainingPeaks API.

    Handles authentication, error handling, and response parsing.
    """

    # Class-level caches: persist across instances within the MCP server process
    _cached_athlete_id: int | None = None
    _cached_user_data: dict | None = None
    _shared_token_cache: TokenCache | None = None
    _shared_response_cache: "_ResponseCache | None" = None

    @classmethod
    def _get_token_cache(cls) -> TokenCache:
        """Get or create the shared token cache."""
        if cls._shared_token_cache is None:
            cls._shared_token_cache = TokenCache()
        return cls._shared_token_cache

    @classmethod
    def _get_response_cache(cls) -> "_ResponseCache":
        """Get or create the shared GET response cache."""
        if cls._shared_response_cache is None:
            cls._shared_response_cache = _ResponseCache()
        return cls._shared_response_cache

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        """Initialize the client.

        Args:
            timeout: Request timeout in seconds.
        """
        self.base_url = TP_API_BASE
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._athlete_id: int | None = None
        self._last_request_time: float = 0.0
        self._token_cache = TPClient._get_token_cache()

    async def __aenter__(self) -> "TPClient":
        """Enter async context."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        await self.close()

    async def _ensure_client(self) -> None:
        """Ensure the HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

    async def _throttle(self) -> None:
        """Enforce minimum interval between requests to avoid rate limiting."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            await asyncio.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with Bearer token authentication.

        Returns:
            Headers dict with Authorization header.
        """
        return {
            "Authorization": f"Bearer {self._token_cache.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _get_cookie_headers(self, cookie: str) -> dict[str, str]:
        """Get request headers with cookie authentication (for token exchange).

        Args:
            cookie: The Production_tpAuth cookie value.

        Returns:
            Headers dict with Cookie header.
        """
        return {
            "Cookie": f"Production_tpAuth={cookie}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def _exchange_cookie_for_token(self) -> APIResponse:
        """Exchange the stored cookie for an OAuth access token.

        Returns:
            APIResponse with token data or error.
        """
        await self._ensure_client()
        await self._throttle()
        assert self._client is not None

        cred = get_credential()
        if not cred.success or not cred.cookie:
            return APIResponse(
                success=False,
                error_code=ErrorCode.AUTH_INVALID,
                message="No credential stored. Run 'tp-mcp auth' to authenticate.",
            )

        url = f"{self.base_url}{TOKEN_ENDPOINT}"
        headers = self._get_cookie_headers(cred.cookie)

        try:
            response = await self._client.request(
                method="GET",
                url=url,
                headers=headers,
            )

            if response.status_code == 401:
                return APIResponse(
                    success=False,
                    error_code=ErrorCode.AUTH_EXPIRED,
                    message="Cookie expired. Use 'tp_refresh_auth' tool to re-authenticate.",
                )

            if response.status_code != 200:
                return APIResponse(
                    success=False,
                    error_code=ErrorCode.API_ERROR,
                    message=f"Token exchange failed: {response.status_code}",
                )

            data = response.json()
            if not data.get("success") or "token" not in data:
                return APIResponse(
                    success=False,
                    error_code=ErrorCode.API_ERROR,
                    message="Invalid token response format",
                )

            return APIResponse(success=True, data=data)

        except httpx.TimeoutException:
            return APIResponse(
                success=False,
                error_code=ErrorCode.NETWORK_ERROR,
                message="Token exchange timed out.",
            )
        except httpx.RequestError as e:
            return APIResponse(
                success=False,
                error_code=ErrorCode.NETWORK_ERROR,
                message=f"Network error during token exchange: {e}",
            )

    async def _ensure_access_token(self) -> APIResponse:
        """Ensure a valid access token is cached.

        Uses double-check locking to prevent concurrent refresh races.

        Returns:
            APIResponse indicating success or the error that occurred.
        """
        # Fast path: token is still valid
        if self._token_cache.is_valid():
            return APIResponse(success=True)

        # Slow path: need to refresh
        async with self._token_cache._lock:
            # Double-check after acquiring lock
            if self._token_cache.is_valid():
                return APIResponse(success=True)

            # Exchange cookie for token
            result = await self._exchange_cookie_for_token()
            if not result.success:
                return result

            # Cache the token
            token_data = result.data["token"]  # type: ignore[index, call-overload]
            self._token_cache.access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self._token_cache.expires_at = time.time() + expires_in

            return APIResponse(success=True)

    async def _request(
        self,
        method: str,
        endpoint: str,
        json: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
        _retry_on_401: bool = True,
    ) -> APIResponse:
        """Make an authenticated API request.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE).
            endpoint: API endpoint (e.g., "/users/v3/user").
            json: JSON body for POST/PUT requests.
            params: Query parameters.
            _retry_on_401: Internal flag to prevent infinite retry loops.

        Returns:
            APIResponse with data or error.
        """
        if _is_forbidden(endpoint):
            logger.error("BLOCKED forbidden endpoint: %s %s", method, endpoint)
            return APIResponse(
                success=False,
                error_code=ErrorCode.FORBIDDEN_ENDPOINT,
                message=(f"Endpoint {endpoint} is disabled in this connector "
                         "(destructive plan operation — has emptied a published plan). "
                         "Use the synthetic tp_apply_training_plan instead."),
            )

        await self._ensure_client()
        assert self._client is not None

        # Ensure we have a valid access token
        token_result = await self._ensure_access_token()
        if not token_result.success:
            return token_result

        await self._throttle()

        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()

        try:
            response = await self._client.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                params=params,
            )

            # Handle 401 with retry logic
            if response.status_code == 401 and _retry_on_401:
                # Token might have expired mid-request, clear and retry once
                self._token_cache.clear()
                return await self._request(method, endpoint, json=json, params=params, _retry_on_401=False)

            return self._handle_response(response)

        except httpx.TimeoutException:
            return APIResponse(
                success=False,
                error_code=ErrorCode.NETWORK_ERROR,
                message="Request timed out. Check your network connection.",
            )
        except httpx.RequestError as e:
            return APIResponse(
                success=False,
                error_code=ErrorCode.NETWORK_ERROR,
                message=f"Network error: {e}",
            )

    def _handle_response(self, response: httpx.Response) -> APIResponse:
        """Handle API response and convert to APIResponse.

        Args:
            response: The httpx response.

        Returns:
            APIResponse with data or error.
        """
        if response.status_code == 200:
            try:
                data = response.json()
                return APIResponse(success=True, data=data)
            except Exception:
                return APIResponse(success=True, data=None)

        if response.status_code == 201:
            try:
                data = response.json()
                return APIResponse(success=True, data=data)
            except Exception:
                return APIResponse(success=True, data=None)

        if response.status_code == 204:
            return APIResponse(success=True, data=None)

        if response.status_code == 401:
            # Don't auto-clear - could be temporary. User can run 'tp-mcp auth-clear' if needed.
            return APIResponse(
                success=False,
                error_code=ErrorCode.AUTH_EXPIRED,
                message="Session expired or invalid. Run 'tp-mcp auth' to re-authenticate.",
            )

        if response.status_code == 403:
            return APIResponse(
                success=False,
                error_code=ErrorCode.AUTH_INVALID,
                message="Access denied. Check your permissions or re-authenticate.",
            )

        if response.status_code == 404:
            return APIResponse(
                success=False,
                error_code=ErrorCode.NOT_FOUND,
                message="Resource not found.",
            )

        if response.status_code == 429:
            return APIResponse(
                success=False,
                error_code=ErrorCode.RATE_LIMITED,
                message="Rate limited. Please wait before making more requests.",
            )

        # Generic error
        return APIResponse(
            success=False,
            error_code=ErrorCode.API_ERROR,
            message=f"API error: {response.status_code}",
        )

    async def get(self, endpoint: str, params: dict[str, Any] | None = None) -> APIResponse:
        """Make a GET request, served from the TTL cache when fresh.

        Repeated identical GETs within the endpoint's freshness window are
        returned from memory (cache HIT). Concurrent identical GETs that arrive
        while the first is still in flight share that single request (cache
        COALESCE) rather than each hitting TrainingPeaks. Only successful
        responses are cached.

        Args:
            endpoint: API endpoint.
            params: Query parameters.

        Returns:
            APIResponse.
        """
        if CACHE_DISABLED:
            return await self._request("GET", endpoint, params=params)

        cache = TPClient._get_response_cache()
        key = cache.make_key(endpoint, params)

        cached = cache.get_fresh(key)
        if cached is not None:
            logger.info("cache HIT      %s", key)
            return cached

        pending = cache.inflight(key)
        if pending is not None:
            logger.info("cache COALESCE %s", key)
            return await pending

        logger.info("cache MISS     %s", key)
        gen_at_start = cache.generation
        future = cache.begin(key)
        try:
            response = await self._request("GET", endpoint, params=params)
            # Skip caching when a write invalidated this key mid-flight — the
            # response we just received predates that write and is now stale.
            if response.success and cache.generation == gen_at_start:
                cache.store(key, response, _cache_ttl_for(endpoint))
            if not future.done():
                future.set_result(response)
            return response
        except BaseException as exc:  # propagate to any coalesced waiters
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            cache.finish(key, future)

    async def post(self, endpoint: str, json: dict[str, Any] | list[Any] | None = None) -> APIResponse:
        """Make a POST request.

        Args:
            endpoint: API endpoint.
            json: JSON body.

        Returns:
            APIResponse.
        """
        response = await self._request("POST", endpoint, json=json)
        self._invalidate_cache_after_write(endpoint)
        return response

    async def put(self, endpoint: str, json: dict[str, Any] | list[Any] | None = None) -> APIResponse:
        """Make a PUT request.

        Args:
            endpoint: API endpoint.
            json: JSON body.

        Returns:
            APIResponse.
        """
        response = await self._request("PUT", endpoint, json=json)
        self._invalidate_cache_after_write(endpoint)
        return response

    async def delete(self, endpoint: str) -> APIResponse:
        """Make a DELETE request.

        Args:
            endpoint: API endpoint.

        Returns:
            APIResponse.
        """
        response = await self._request("DELETE", endpoint)
        self._invalidate_cache_after_write(endpoint)
        return response

    @staticmethod
    def _invalidate_cache_after_write(endpoint: str) -> None:
        """Drop cached GETs that a write to ``endpoint`` may have made stale."""
        if CACHE_DISABLED:
            return
        removed = TPClient._get_response_cache().invalidate_for_write(endpoint)
        if removed:
            logger.info("cache INVALIDATE %d entr%s after write %s",
                        removed, "y" if removed == 1 else "ies", endpoint)

    async def get_raw(self, endpoint: str, params: dict[str, Any] | None = None) -> RawResponse:
        """Make an authenticated GET request and return the raw binary response.

        Handles token refresh and retries on 401 exactly once, mirroring _request logic.

        Args:
            endpoint: API endpoint.
            params: Query parameters.

        Returns:
            RawResponse with binary content and headers, or error.
        """
        if _is_forbidden(endpoint):
            logger.error("BLOCKED forbidden endpoint: GET %s", endpoint)
            return RawResponse(
                success=False,
                error_code=ErrorCode.FORBIDDEN_ENDPOINT,
                message=f"Endpoint {endpoint} is disabled in this connector.",
            )
        await self._ensure_client()
        assert self._client is not None

        token_result = await self._ensure_access_token()
        if not token_result.success:
            return RawResponse(
                success=False,
                error_code=token_result.error_code,
                message=token_result.message,
            )

        await self._throttle()

        url = f"{self.base_url}{endpoint}"
        headers = {**self._get_headers(), "Accept": "*/*"}

        try:
            response = await self._client.request("GET", url=url, headers=headers, params=params)

            if response.status_code == 401:
                self._token_cache.clear()
                token_result = await self._ensure_access_token()
                if not token_result.success:
                    return RawResponse(
                        success=False,
                        error_code=token_result.error_code,
                        message=token_result.message,
                    )
                await self._throttle()
                headers = {**self._get_headers(), "Accept": "*/*"}
                response = await self._client.request("GET", url=url, headers=headers, params=params)

        except httpx.TimeoutException:
            return RawResponse(
                success=False,
                error_code=ErrorCode.NETWORK_ERROR,
                message="Request timed out. Check your network connection.",
            )
        except httpx.RequestError as e:
            return RawResponse(
                success=False,
                error_code=ErrorCode.NETWORK_ERROR,
                message=f"Network error: {e}",
            )

        if response.status_code == 401:
            return RawResponse(
                success=False,
                error_code=ErrorCode.AUTH_EXPIRED,
                message="Session expired or invalid. Run 'tp-mcp auth' to re-authenticate.",
            )
        if response.status_code == 404:
            return RawResponse(
                success=False,
                error_code=ErrorCode.NOT_FOUND,
                message="Resource not found.",
            )
        if response.status_code != 200:
            return RawResponse(
                success=False,
                error_code=ErrorCode.API_ERROR,
                message=f"API error: {response.status_code} - {response.text}",
            )

        return RawResponse(
            success=True,
            content=response.content,
            content_type=response.headers.get("Content-Type"),
            content_disposition=response.headers.get("Content-Disposition"),
        )

    @property
    def athlete_id(self) -> int | None:
        """Get the cached athlete ID."""
        return self._athlete_id

    @athlete_id.setter
    def athlete_id(self, value: int | None) -> None:
        """Set the athlete ID."""
        self._athlete_id = value

    async def _get_user_data(self) -> dict | None:
        """Get user data, using class-level cache to avoid redundant API calls."""
        if TPClient._cached_user_data is not None:
            return TPClient._cached_user_data

        response = await self.get("/users/v3/user")
        if not response.success or not response.data:
            return None

        user_data = response.data.get("user", response.data)
        TPClient._cached_user_data = user_data
        return user_data

    async def ensure_athlete_id(self) -> int | None:
        """Get athlete ID, resolving coach athlete targeting via context var.

        For coach accounts, reads the athlete_override context variable to
        target a specific athlete by name or ID. When no override is set,
        resolves to the coach's own athlete entry.

        Caches at class level only when no athlete override is active.
        """
        from tp_mcp.client.context import athlete_override

        athlete = athlete_override.get()

        # Use cache only when no specific athlete is requested
        if athlete is None:
            if TPClient._cached_athlete_id is not None:
                self._athlete_id = TPClient._cached_athlete_id
                return TPClient._cached_athlete_id

            if self._athlete_id is not None:
                TPClient._cached_athlete_id = self._athlete_id
                return self._athlete_id

        user_data = await self._get_user_data()
        if not user_data:
            return None

        person_id = user_data.get("personId")
        athletes = user_data.get("athletes", [])

        athlete_id = None

        if athlete is not None:
            # Resolve by explicit athlete name or ID
            try:
                target_id = int(athlete)
                for a in athletes:
                    if a.get("athleteId") == target_id:
                        athlete_id = target_id
                        break
            except ValueError:
                # Search by name (case-insensitive), detecting ambiguity
                search = athlete.lower()
                matches = []
                for a in athletes:
                    first = (a.get("firstName") or "").lower()
                    last = (a.get("lastName") or "").lower()
                    full = f"{first} {last}"
                    if search in (first, last, full):
                        matches.append(a)

                if len(matches) == 1:
                    athlete_id = matches[0].get("athleteId")
                elif len(matches) > 1:
                    names = [
                        f"{a.get('firstName', '')} {a.get('lastName', '')} (ID: {a.get('athleteId')})"
                        for a in matches
                    ]
                    raise ValueError(
                        f"Ambiguous athlete name '{athlete}' matches {len(matches)} athletes: "
                        + ", ".join(names)
                        + ". Use full name or athlete ID to disambiguate."
                    ) from None
        elif athletes:
            # Default: find the coach's own athlete entry
            for a in athletes:
                if a.get("coachedBy") == person_id and a.get("email", "").lower() == user_data.get("email", "").lower():
                    athlete_id = a.get("athleteId")
                    break
            # Fallback: match by last name
            if not athlete_id:
                user_last = (user_data.get("lastName") or "").lower()
                for a in athletes:
                    if (a.get("lastName") or "").lower() == user_last and a.get("coachedBy") == person_id:
                        athlete_id = a.get("athleteId")
                        break
            # Final fallback: personId or first athlete entry (non-coach accounts)
            if not athlete_id:
                athlete_id = person_id or (athletes[0].get("athleteId") if athletes else None)
        else:
            athlete_id = person_id

        if athlete_id:
            self._athlete_id = athlete_id
            if athlete is None:
                TPClient._cached_athlete_id = athlete_id

        return athlete_id

    async def test_token_exchange(self) -> dict[str, Any]:
        """Test the full token exchange flow for diagnostics.

        Returns:
            Dict with test results including success status and step details.
        """
        result: dict[str, Any] = {"success": False, "step": "init", "details": {}}

        # Step 1: Check credential
        cred = get_credential()
        if not cred.success or not cred.cookie:
            result["step"] = "credential_check"
            result["error"] = "No credential stored. Run 'tp-mcp auth' to authenticate."
            return result

        result["details"]["has_credential"] = True

        # Step 2: Exchange cookie for token
        exchange_result = await self._exchange_cookie_for_token()
        if not exchange_result.success:
            result["step"] = "token_exchange"
            result["error"] = exchange_result.message
            result["error_code"] = exchange_result.error_code.value if exchange_result.error_code else None
            return result

        result["details"]["token_exchange"] = "success"
        token_data = exchange_result.data["token"]  # type: ignore[index, call-overload]
        result["details"]["expires_in"] = token_data.get("expires_in")

        # Step 3: Verify token structure
        if "access_token" not in token_data:
            result["step"] = "token_validation"
            result["error"] = "Token response missing access_token"
            return result

        result["details"]["token_valid"] = True

        # Step 4: Make test API call
        test_response = await self.get("/users/v3/user")
        if not test_response.success:
            result["step"] = "api_test"
            result["error"] = test_response.message
            result["error_code"] = test_response.error_code.value if test_response.error_code else None
            return result

        result["details"]["api_test"] = "success"
        if isinstance(test_response.data, dict):
            result["details"]["user_id"] = test_response.data.get("Id")

        result["success"] = True
        result["step"] = "complete"
        return result
