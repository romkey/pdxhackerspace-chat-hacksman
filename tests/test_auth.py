import asyncio
import base64
import json
import time
from collections.abc import Callable

# Any: ID token claims are JSON.
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from chat_hacksman.auth import (
    SESSION_COOKIE,
    OidcClient,
    OidcError,
    login_page,
    safe_next,
    username_from_claims,
)
from tests.conftest import Handler, MakeClient

MockHttp = Callable[[Handler], None]

ISSUER = "https://auth.example.org/application/o/chat/"
CLIENT_ID = "dummy-client"
PASSWORD = "dummy-password"  # noqa: S105 - a dummy credential for these tests


def _jwt(claims: dict[str, Any]) -> str:
    def part(data: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'RS256'})}.{part(claims)}.signature"


class FakeProvider:
    """An OpenID provider: discovery, and a token endpoint that returns an
    ID token with the claims given (the nonce copied from the login)."""

    def __init__(self, **claims: Any) -> None:  # Any: claim values are JSON.
        self.claims: dict[str, Any] = {
            "iss": ISSUER,
            "aud": CLIENT_ID,
            "exp": int(time.time()) + 300,
            "preferred_username": "alice",
            **claims,
        }
        self.nonce = ""
        self.token_requests: list[dict[str, list[str]]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/.well-known/openid-configuration"):
            return httpx.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": "https://auth.example.org/authorize",
                    "token_endpoint": "https://auth.example.org/token",
                },
            )
        if request.url.path == "/token":
            form = parse_qs(request.content.decode())
            self.token_requests.append(form)
            claims = {"nonce": self.nonce, **self.claims}
            return httpx.Response(200, json={"id_token": _jwt(claims), "access_token": "x"})
        return httpx.Response(404)


def _start_sso(client: Any, provider: FakeProvider, next_path: str = "/") -> str:
    """Follows the SSO button to the provider; returns the state."""
    response = client.get(f"/auth/oidc/login?next={next_path}", follow_redirects=False)
    assert response.status_code == 303
    location = urlparse(response.headers["location"])
    assert location.netloc == "auth.example.org"
    query = parse_qs(location.query)
    assert query["client_id"] == [CLIENT_ID]
    assert query["code_challenge_method"] == ["S256"]
    assert query["redirect_uri"] == ["http://testserver/auth/oidc/callback"]
    provider.nonce = query["nonce"][0]
    state: str = query["state"][0]
    return state


def test_no_login_configured_leaves_the_app_open(make_client: MakeClient) -> None:
    with make_client() as client:
        assert client.get("/api/settings").status_code == 200
        assert client.get("/api/whoami").json() == {"login_required": False, "username": None}
        response = client.get("/login", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/"


def test_local_login_protects_routes(make_client: MakeClient) -> None:
    with make_client(local_login_username="alice", local_login_password=PASSWORD) as client:
        assert client.get("/api/settings").status_code == 401
        page = client.get("/static/app.js", follow_redirects=False)
        assert page.status_code == 303
        assert page.headers["location"] == "/login?next=%2Fstatic%2Fapp.js"
        assert client.get("/health").status_code == 200
        assert client.get("/healthz").status_code == 200

        login = client.get("/login")
        assert login.status_code == 200
        assert 'action="/login"' in login.text
        assert "<details" not in login.text  # only the local login: shown open
        assert "Login via Authentik" not in login.text

        wrong = client.post("/login", data={"username": "alice", "password": "nope", "next": "/"})
        assert wrong.status_code == 401
        assert "Wrong username or password" in wrong.text
        assert client.get("/api/settings").status_code == 401

        ok = client.post(
            "/login",
            data={"username": "alice", "password": PASSWORD, "next": "/?tab=history"},
            follow_redirects=False,
        )
        assert ok.status_code == 303
        assert ok.headers["location"] == "/?tab=history"
        assert "httponly" in ok.headers["set-cookie"].lower()
        assert client.get("/api/settings").status_code == 200
        assert client.get("/").status_code == 200
        assert client.get("/api/whoami").json() == {"login_required": True, "username": "alice"}

        logout = client.post("/logout", follow_redirects=False)
        assert logout.status_code == 303
        assert logout.headers["location"] == "/login"
        assert client.get("/api/settings").status_code == 401


def test_session_survives_a_restart(make_client: MakeClient) -> None:
    settings = {"local_login_username": "alice", "local_login_password": PASSWORD}
    with make_client(**settings) as first:
        first.post("/login", data={"username": "alice", "password": PASSWORD})
        token = first.cookies[SESSION_COOKIE]
    with make_client(**settings) as second:
        second.cookies.set(SESSION_COOKIE, token)
        assert second.get("/api/settings").status_code == 200
        second.cookies.set(SESSION_COOKIE, "forged")
        assert second.get("/api/settings").status_code == 401


def test_login_never_redirects_off_site(make_client: MakeClient) -> None:
    with make_client(local_login_username="alice", local_login_password=PASSWORD) as client:
        response = client.post(
            "/login",
            data={"username": "alice", "password": PASSWORD, "next": "//evil.example"},
            follow_redirects=False,
        )
    assert response.headers["location"] == "/"


def test_login_page_with_both_methods_hides_local_login(make_client: MakeClient) -> None:
    with make_client(
        local_login_username="alice",
        local_login_password=PASSWORD,
        oidc_issuer=ISSUER,
        oidc_client_id=CLIENT_ID,
        oidc_login_label="Log in <with> SSO",
    ) as client:
        page = client.get("/login").text
        assert "Log in &lt;with&gt; SSO" in page
        assert "<details><summary>Local login</summary>" in page
        assert page.index("Log in &lt;with&gt; SSO") < page.index("<details>")
        wrong = client.post("/login", data={"username": "alice", "password": "nope"})
        assert "<details open><summary>Local login</summary>" in wrong.text


def test_oidc_login(make_client: MakeClient, mock_http: MockHttp) -> None:
    provider = FakeProvider()
    mock_http(provider.handler)
    with make_client(
        oidc_issuer=ISSUER,
        oidc_client_id=CLIENT_ID,
        oidc_client_secret="dummy-secret",  # noqa: S106
    ) as client:
        page = client.get("/login").text
        assert ">Login via Authentik</a>" in page
        assert 'action="/login"' not in page

        state = _start_sso(client, provider, "/feedback")
        callback = client.get(f"/auth/oidc/callback?state={state}&code=abc", follow_redirects=False)
        assert callback.status_code == 303
        assert callback.headers["location"] == "/feedback"
        assert client.get("/api/whoami").json()["username"] == "alice"

        token_form = provider.token_requests[0]
        assert token_form["code"] == ["abc"]
        assert token_form["client_secret"] == ["dummy-secret"]
        assert token_form["code_verifier"][0]

        # A state is good once.
        replay = client.get(f"/auth/oidc/callback?state={state}&code=abc")
        assert replay.status_code == 401
        assert "unknown or expired login" in replay.text

        local = client.post("/login", data={"username": "alice", "password": PASSWORD})
        assert local.status_code == 403


@pytest.mark.parametrize(
    ("claims", "message"),
    [
        ({"iss": "https://other.example.org/"}, "another issuer"),
        ({"aud": ["someone-else"]}, "another client"),
        ({"exp": 1000}, "expired"),
        ({"nonce": "wrong"}, "nonce"),
    ],
)
def test_oidc_rejects_untrustworthy_id_tokens(
    make_client: MakeClient, mock_http: MockHttp, claims: dict[str, Any], message: str
) -> None:
    provider = FakeProvider(**claims)
    mock_http(provider.handler)
    with make_client(oidc_issuer=ISSUER, oidc_client_id=CLIENT_ID) as client:
        state = _start_sso(client, provider)
        response = client.get(f"/auth/oidc/callback?state={state}&code=abc")
        assert response.status_code == 401
        assert message in response.text
        assert client.get("/api/settings").status_code == 401
    # A public client sends no secret.
    assert "client_secret" not in provider.token_requests[0]


def test_oidc_provider_errors(make_client: MakeClient, mock_http: MockHttp) -> None:
    mock_http(lambda request: httpx.Response(500))
    with make_client(oidc_issuer=ISSUER, oidc_client_id=CLIENT_ID) as client:
        down = client.get("/auth/oidc/login")
        assert down.status_code == 502
        assert "unavailable" in down.text
        refused = client.get("/auth/oidc/callback?error=access_denied&error_description=No+way")
        assert refused.status_code == 401
        assert "No way" in refused.text


def test_oidc_routes_without_oidc(make_client: MakeClient) -> None:
    with make_client(local_login_username="alice", local_login_password=PASSWORD) as client:
        assert client.get("/auth/oidc/login").status_code == 404
        assert client.get("/auth/oidc/callback?state=x&code=y").status_code == 404


def test_oidc_redirect_uri_setting(make_client: MakeClient, mock_http: MockHttp) -> None:
    mock_http(FakeProvider().handler)
    with make_client(
        oidc_issuer=ISSUER,
        oidc_client_id=CLIENT_ID,
        oidc_redirect_uri="https://chat.example.org/auth/oidc/callback",
    ) as client:
        response = client.get("/auth/oidc/login", follow_redirects=False)
    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["redirect_uri"] == ["https://chat.example.org/auth/oidc/callback"]


def test_secure_cookie_behind_https_proxy(make_client: MakeClient) -> None:
    with make_client(local_login_username="alice", local_login_password=PASSWORD) as client:
        response = client.post(
            "/login",
            data={"username": "alice", "password": PASSWORD},
            headers={"X-Forwarded-Proto": "https"},
            follow_redirects=False,
        )
    assert "secure" in response.headers["set-cookie"].lower()


def test_discovery_url_accepts_issuer_or_discovery_url() -> None:
    issuer = OidcClient(issuer=ISSUER, client_id="x", client_secret=None, scopes="openid")
    assert issuer.discovery_url == (
        "https://auth.example.org/application/o/chat/.well-known/openid-configuration"
    )
    direct = OidcClient(
        issuer=issuer.discovery_url, client_id="x", client_secret=None, scopes="openid"
    )
    assert direct.discovery_url == issuer.discovery_url


def test_discovery_without_endpoints_is_an_error(mock_http: MockHttp) -> None:
    mock_http(lambda request: httpx.Response(200, json={"issuer": ISSUER}))
    client = OidcClient(issuer=ISSUER, client_id="x", client_secret=None, scopes="openid")
    with pytest.raises(OidcError, match="missing endpoints"):
        asyncio.run(client.metadata())


def test_username_from_claims() -> None:
    assert username_from_claims({"email": "a@example.org", "sub": "1"}) == "a@example.org"
    assert username_from_claims({"sub": "1"}) == "1"
    with pytest.raises(OidcError, match="names no user"):
        username_from_claims({"preferred_username": " "})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "/"),
        ("", "/"),
        ("/history?x=1", "/history?x=1"),
        ("https://evil.example", "/"),
        ("//evil.example", "/"),
        ("/\\evil.example", "/"),
    ],
)
def test_safe_next(value: str | None, expected: str) -> None:
    assert safe_next(value) == expected


def test_login_page_escapes_next() -> None:
    page = login_page(oidc_label=None, local_enabled=True, next_path='/"><script>')
    assert "<script>" not in page
