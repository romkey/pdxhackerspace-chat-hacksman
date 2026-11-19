"""Logging in: the one local account, and OpenID Connect single sign-on.

OIDC uses the authorization code flow with PKCE. The ID token comes straight
from the provider's token endpoint over TLS, so its signature isn't checked
(OpenID Connect Core 3.1.3.7 allows this); its issuer, audience, expiry and
nonce are.
"""

import base64
import binascii
import hashlib
import html
import json
import secrets
import time
from dataclasses import dataclass, field

# Any: discovery documents and ID token claims are untyped JSON.
from typing import Any
from urllib.parse import urlencode

import httpx

SESSION_COOKIE = "chat_hacksman_session"
DISCOVERY_PATH = "/.well-known/openid-configuration"
# How long a browser has to come back from the provider.
PENDING_LOGIN_SECONDS = 600
# Leeway for clock differences when checking the ID token's expiry.
CLOCK_SKEW_SECONDS = 120


class OidcError(Exception):
    """The provider couldn't be reached, or its answer can't be trusted."""


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def safe_next(value: str | None) -> str:
    """A path on this app to return to after logging in; "/" for anything else."""
    if not value or not value.startswith("/") or value.startswith("//") or "\\" in value:
        return "/"
    return value


def local_login_valid(
    expected_username: str | None, expected_password: str | None, username: str, password: str
) -> bool:
    if not expected_username or not expected_password:
        return False
    # Both compared every time, so the timing doesn't say which was wrong.
    username_ok = secrets.compare_digest(username.encode(), expected_username.encode())
    password_ok = secrets.compare_digest(password.encode(), expected_password.encode())
    return username_ok and password_ok


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _jwt_claims(token: str) -> dict[str, Any]:
    """The payload of a JWT, without checking its signature."""
    try:
        payload = token.split(".")[1]
        decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        claims = json.loads(decoded)
    except (IndexError, ValueError, binascii.Error) as exc:
        raise OidcError("the ID token is not a JWT") from exc
    if not isinstance(claims, dict):
        raise OidcError("the ID token is not a JWT")
    return claims


@dataclass(slots=True)
class PendingLogin:
    nonce: str
    code_verifier: str
    redirect_uri: str
    next_path: str
    expires: float


@dataclass(slots=True)
class OidcClient:
    issuer: str
    client_id: str
    client_secret: str | None
    scopes: str
    timeout_seconds: float = 15.0
    _metadata: dict[str, Any] | None = None
    # Logins started here and not finished yet, by state. One process, so
    # memory is enough; a restart only makes someone click the button again.
    _pending: dict[str, PendingLogin] = field(default_factory=dict)

    @property
    def discovery_url(self) -> str:
        if self.issuer.endswith(DISCOVERY_PATH):
            return self.issuer
        return self.issuer.rstrip("/") + DISCOVERY_PATH

    async def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.get(self.discovery_url)
                    response.raise_for_status()
                    metadata = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise OidcError(f"discovery at {self.discovery_url} failed: {exc}") from exc
            needed = ("issuer", "authorization_endpoint", "token_endpoint")
            if not isinstance(metadata, dict) or not all(
                isinstance(metadata.get(key), str) for key in needed
            ):
                raise OidcError(f"discovery at {self.discovery_url} is missing endpoints")
            self._metadata = metadata
        return self._metadata

    def _drop_expired(self) -> None:
        now = time.monotonic()
        for state in [state for state, login in self._pending.items() if login.expires < now]:
            del self._pending[state]

    async def authorization_url(self, redirect_uri: str, next_path: str) -> str:
        """Where to send the browser to log in; remembers what the callback needs."""
        metadata = await self.metadata()
        self._drop_expired()
        state = secrets.token_urlsafe(24)
        login = PendingLogin(
            nonce=secrets.token_urlsafe(24),
            code_verifier=secrets.token_urlsafe(48),
            redirect_uri=redirect_uri,
            next_path=next_path,
            expires=time.monotonic() + PENDING_LOGIN_SECONDS,
        )
        self._pending[state] = login
        challenge = _b64url(hashlib.sha256(login.code_verifier.encode("ascii")).digest())
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self.client_id,
                "redirect_uri": redirect_uri,
                "scope": self.scopes,
                "state": state,
                "nonce": login.nonce,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        endpoint = str(metadata["authorization_endpoint"])
        return f"{endpoint}{'&' if '?' in endpoint else '?'}{query}"

    async def finish(self, state: str, code: str) -> tuple[str, str]:
        """The username and the path to return to, for a callback's state and code."""
        self._drop_expired()
        login = self._pending.pop(state, None)
        if login is None:
            raise OidcError("unknown or expired login; try again")
        metadata = await self.metadata()
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": login.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": login.code_verifier,
        }
        # A confidential client sends its secret (client_secret_post); a public
        # one relies on PKCE alone.
        if self.client_secret:
            form["client_secret"] = self.client_secret
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(str(metadata["token_endpoint"]), data=form)
                response.raise_for_status()
                tokens = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OidcError(f"the token request failed: {exc}") from exc
        id_token = tokens.get("id_token") if isinstance(tokens, dict) else None
        if not isinstance(id_token, str):
            raise OidcError("the provider returned no ID token; is the openid scope set?")
        claims = _jwt_claims(id_token)
        self._check_claims(claims, str(metadata["issuer"]), login.nonce)
        return username_from_claims(claims), login.next_path

    def _check_claims(self, claims: dict[str, Any], issuer: str, nonce: str) -> None:
        if claims.get("iss") != issuer:
            raise OidcError("the ID token is from another issuer")
        audience = claims.get("aud")
        audiences = audience if isinstance(audience, list) else [audience]
        if self.client_id not in audiences:
            raise OidcError("the ID token is for another client")
        expires = claims.get("exp")
        if not isinstance(expires, int | float) or expires + CLOCK_SKEW_SECONDS < time.time():
            raise OidcError("the ID token has expired")
        if claims.get("nonce") != nonce:
            raise OidcError("the ID token's nonce doesn't match")


def username_from_claims(claims: dict[str, Any]) -> str:
    for key in ("preferred_username", "email", "name", "sub"):
        value = claims.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise OidcError("the ID token names no user")


LOGIN_PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Log in - Chat Hacksman</title>
    <style>
      body {{ font-family: system-ui, sans-serif; margin: 0; background: #111827;
        color: #f3f4f6; min-height: 100vh; display: grid; place-items: center; }}
      .card {{ background: #1f2937; border-radius: 8px; padding: 1.5rem;
        width: min(340px, calc(100vw - 2rem)); box-sizing: border-box; }}
      h1 {{ font-size: 1.3rem; margin: 0 0 1rem; }}
      .error {{ background: #7f1d1d; border-radius: 6px; padding: 0.6rem; margin-bottom: 1rem; }}
      .sso {{ display: block; text-align: center; text-decoration: none; color: #f3f4f6;
        background: #2563eb; border-radius: 6px; padding: 0.7rem; font-weight: 600; }}
      details {{ margin-top: 1.2rem; }}
      summary {{ cursor: pointer; color: #9ca3af; font-size: 0.9rem; }}
      form {{ display: grid; gap: 0.6rem; margin-top: 0.8rem; }}
      label {{ display: grid; gap: 0.3rem; font-size: 0.9rem; color: #d1d5db; }}
      input, button {{ border-radius: 6px; border: 1px solid #4b5563; background: #111827;
        color: #f3f4f6; padding: 0.5rem; font: inherit; }}
      button {{ cursor: pointer; background: #374151; }}
    </style>
  </head>
  <body>
    <main class="card">
      <h1>Chat Hacksman</h1>
      {error}{body}
    </main>
  </body>
</html>
"""

LOCAL_FORM = """<form method="post" action="/login">
        <input type="hidden" name="next" value="{next}" />
        <label>Username <input name="username" autocomplete="username" required /></label>
        <label>Password
          <input name="password" type="password" autocomplete="current-password" required />
        </label>
        <button type="submit">Log in</button>
      </form>"""


def login_page(
    *,
    oidc_label: str | None,
    local_enabled: bool,
    next_path: str,
    error: str | None = None,
    local_open: bool = False,
) -> str:
    """The login page: the single sign-on button, with the local login tucked
    under a <details> when both are on."""
    parts = []
    if oidc_label:
        href = "/auth/oidc/login?" + urlencode({"next": next_path})
        parts.append(f'<a class="sso" href="{html.escape(href)}">{html.escape(oidc_label)}</a>')
    if local_enabled:
        form = LOCAL_FORM.format(next=html.escape(next_path))
        if oidc_label:
            open_attr = " open" if local_open else ""
            form = f"<details{open_attr}><summary>Local login</summary>{form}</details>"
        parts.append(form)
    error_html = f'<p class="error" role="alert">{html.escape(error)}</p>' if error else ""
    return LOGIN_PAGE.format(error=error_html, body="\n      ".join(parts))
