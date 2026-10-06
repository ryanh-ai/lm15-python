"""PR 1: existing Responses dialect bound to two explicit AWS doors.

Profile prefix/ARN vectors are serialization tests, NOT model eligibility.
Live bodies come from the pinned companion contract; LM15_CONTRACT_DIR is a
local-development override while that companion commit awaits landing.
"""
from __future__ import annotations

from dataclasses import replace
import datetime as dt
import json
import os
from pathlib import Path

import pytest

from lm15 import Config, Message, Reasoning, Request, access
from lm15.compat import OpenAIResponsesCompat
from lm15.credentials import AwsCredentials, BearerToken
from lm15.errors import NotConfiguredError, UnsupportedFeatureError, UnsupportedModelError
from lm15.features import EndpointSupport
from lm15.providers import AsyncOpenAILM, OpenAILM
from lm15.registry import PROVIDERS, ProviderDefinition, lookup
from lm15.result import amaterialize_response, materialize_response
from lm15.router import AsyncLMRouter, LMRouter, RouterConfig
from lm15.testing import FakeResponse, FakeTransport
from lm15 import serde
from lm15.vet import normalize_transport_request

from .test_async_adapters import FakeAsyncTransport

NOW = dt.datetime(2026, 10, 5, 12, tzinfo=dt.timezone.utc)
AWS = AwsCredentials("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
DOORS = (
    ("bedrock-responses", "bedrock-runtime.us-west-2.amazonaws.com", "bedrock"),
    ("bedrock-mantle-responses", "bedrock-mantle.us-west-2.api.aws", "bedrock-mantle"),
)
# Deliberately opaque synthetic vectors. No invented GPT/Grok EU eligibility.
IDS = (
    "us.openai.gpt-6-astra", "global.xai.grok-4.6",
    "eu.serialization-only", "au.serialization-only", "jp.serialization-only",
    "us-gov.serialization-only", "future-scope.serialization-only",
    "arn:aws:bedrock:us-west-2:000000000000:inference-profile/opaque:id",
    "arn:aws:bedrock:us-west-2:000000000000:application-inference-profile/opaque:id",
)
CONTRACT = Path(os.environ.get("LM15_CONTRACT_DIR", str(Path(__file__).resolve().parents[2] / "lm15-contract")))
CAPTURES = (
    ("bedrock-responses", "astra_us"),
    ("bedrock-mantle-responses", "gemma_31b"), ("bedrock-mantle-responses", "astra"),
)


def request(model):
    return Request(model=model, messages=(Message.user("What is 2 + 2? Answer briefly."),),
                   config=Config(max_tokens=256, reasoning=Reasoning(effort="low"), store=False))


def config(pid, **kwargs):
    return RouterConfig(api_keys={pid: AWS}, settings={pid: {"region": "us-west-2"}}, env={}, **kwargs)


@pytest.mark.parametrize("pid,host,service", DOORS)
def test_registration_is_existing_dialect_and_honest_surfaces(pid, host, service):
    entry = lookup(pid)
    assert entry is lookup(pid.replace("-", "_"))
    assert entry.adapter is OpenAILM and entry.async_adapter is AsyncOpenAILM
    assert entry.hosted and entry.bound and entry.dialect == "openai-responses"
    assert entry.compat == "openai"
    assert entry.access in access.CLOUD_HOST_POLICIES
    assert entry.access.credential_policy == "aws-chain"
    assert entry.access.auth_modes == entry.access.auth_scheme == ("sigv4", "bearer")
    assert entry.supports == EndpointSupport(complete=True, stream=True, responses_api=True)
    assert entry.access.host.sigv4_service == service
    assert entry.access.host.settings[0].default is None


@pytest.mark.parametrize("pid,host,service", DOORS)
@pytest.mark.parametrize("model", IDS)
@pytest.mark.parametrize("stream", (False, True))
def test_sync_route_preserves_opaque_model_and_source_region(pid, host, service, model, stream):
    with LMRouter(config(pid)) as router:
        resolution = router.resolve(f"{pid}:{model}")
        assert resolution.provider == pid and resolution.model == model and not resolution.declared
        lm = router.lm(f"{pid}:{model}")
        wire = lm.build_request(request(model), stream=stream)
        assert wire.url == f"https://{host}/openai/v1/responses"
        body = json.loads(wire.body)
        assert body["model"] == model and body["stream"] is stream
        assert body["max_output_tokens"] == 256 and body["reasoning"]["effort"] == "low"
        assert body["store"] is False
        assert f"/us-west-2/{service}/aws4_request" in {k.lower(): v for k, v in wire.headers}["authorization"]


@pytest.mark.parametrize("pid,host,service", DOORS)
@pytest.mark.parametrize("model", IDS)
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.asyncio
async def test_async_route_preserves_opaque_model_and_source_region(pid, host, service, model, stream):
    # Captured bytes answer the test double, not the synthetic ID on AWS.
    feature = "astra_us" if pid == "bedrock-responses" else "astra"
    _, body = captured(pid, feature + ("_streaming" if stream else "_basic_text"))
    transport = FakeAsyncTransport(body)
    async with AsyncLMRouter(config(pid, transport=transport)) as router:
        assert isinstance(router.lm(f"{pid}:{model}"), AsyncOpenAILM)
        req = request(f"{pid}:{model}")
        if stream:
            await amaterialize_response(router.stream(req), req)
        else:
            await router.complete(req)
        wire = transport.requests[0]
        assert wire.url == f"https://{host}/openai/v1/responses"
        assert json.loads(wire.body)["model"] == model
        assert json.loads(wire.body)["stream"] is stream
        assert f"/us-west-2/{service}/aws4_request" in {k.lower(): v for k, v in wire.headers}["authorization"]


@pytest.mark.parametrize("pid,host,service", DOORS)
@pytest.mark.parametrize("key", ("test-key", BearerToken("test-token")))
def test_bearer_serialization_does_not_change_door(pid, host, service, key):
    lm = OpenAILM(api_key=key, access=PROVIDERS[pid].access, settings={"region": "us-west-2"})
    wire = lm.build_request(request("opaque-model"), stream=False)
    assert wire.url == f"https://{host}/openai/v1/responses"
    assert dict(wire.headers)["Authorization"] == ("Bearer test-key" if isinstance(key, str) else "Bearer test-token")
    # Serialization coverage only: no live bearer validity claim on the new doors.


@pytest.mark.parametrize("pid,host,service", DOORS)
def test_missing_region_cannot_fall_back_or_default(pid, host, service, monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "missing-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "missing-credentials"))
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with LMRouter(RouterConfig(api_keys={pid: AWS}, env={
        "AWS_CONFIG_FILE": str(tmp_path / "missing-config"),
        "AWS_SHARED_CREDENTIALS_FILE": str(tmp_path / "missing-credentials"),
    })) as router:
        with pytest.raises(NotConfiguredError, match="AWS_REGION"):
            router.lm(f"{pid}:opaque-model")


@pytest.mark.parametrize("pid,host,service", DOORS)
def test_endpoint_override_keeps_door_path_and_signature(pid, host, service):
    endpoint_var = "AWS_ENDPOINT_URL_BEDROCK_MANTLE" if service == "bedrock-mantle" else "AWS_ENDPOINT_URL_BEDROCK_RUNTIME"
    cfg = replace(config(pid), env={endpoint_var: "https://private.example.test", "AWS_ENDPOINT_URL": "https://wrong.example.test"})
    with LMRouter(cfg) as router:
        wire = router.lm(f"{pid}:opaque-model").build_request(request("opaque-model"), stream=False)
        assert wire.url == "https://private.example.test/openai/v1/responses"
        assert f"/us-west-2/{service}/aws4_request" in {k.lower(): v for k, v in wire.headers}["authorization"]
    with LMRouter(replace(cfg, base_urls={pid: "https://explicit.example.test/openai/v1"})) as router:
        assert router.lm(f"{pid}:opaque-model").base_url == "https://explicit.example.test/openai/v1"


@pytest.mark.parametrize("pid,host,service", DOORS)
def test_unverified_models_surface_is_refused(pid, host, service):
    with LMRouter(config(pid)) as router:
        with pytest.raises(UnsupportedFeatureError):
            router.lm(f"{pid}:opaque-model").list_models()


def captured(pid, feature):
    path = CONTRACT / "cases" / pid / f"{feature}.json"
    case = json.loads(path.read_text(encoding="utf-8"))  # Never skip missing required evidence.
    body = (CONTRACT / "bodies" / case["id"] / case["pinned_body"]).read_bytes()
    return case, body


@pytest.mark.parametrize("pid,feature", CAPTURES)
@pytest.mark.parametrize("stream", (False, True))
def test_captured_wire_and_sync_response_execute_binding(pid, feature, stream):
    case, body = captured(pid, feature + ("_streaming" if stream else "_basic_text"))
    req = serde.request_from_dict(case["canonical_request"])
    fixed = dt.datetime.fromisoformat(case["now"].replace("Z", "+00:00"))
    lm = OpenAILM(api_key=AWS, access=PROVIDERS[pid].access, settings=case["settings"], clock=lambda: fixed,
                  transport=FakeTransport([FakeResponse(200, body)]))
    assert normalize_transport_request(lm.build_request(req, stream=stream)) == case["request"]
    response = materialize_response(lm.stream(req), req) if stream else lm.complete(req)
    assert "4" in response.text
    assert response.finish_reason == "stop" and response.usage.total_tokens > 0
    assert not any(p.type == "data" for p in response.message.parts)
    assert len(lm.transport.requests) == 1
    routed_transport = FakeTransport([FakeResponse(200, body)])
    with LMRouter(config(pid, transport=routed_transport)) as router:
        routed = replace(req, model=f"{pid}:{req.model}")
        actual = materialize_response(router.stream(routed), routed) if stream else router.complete(routed)
        assert actual.text == response.text and actual.finish_reason == response.finish_reason
        assert len(routed_transport.requests) == 1
        wire = routed_transport.requests[0]
        assert wire.url == case["request"]["url"]
        assert json.loads(wire.body) == case["request"]["body"]


@pytest.mark.parametrize("pid,feature", CAPTURES)
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.asyncio
async def test_captured_async_router_complete_and_stream(pid, feature, stream):
    case, body = captured(pid, feature + ("_streaming" if stream else "_basic_text"))
    req = serde.request_from_dict(case["canonical_request"])
    transport = FakeAsyncTransport(body)
    async with AsyncLMRouter(config(pid, transport=transport)) as router:
        routed = replace(req, model=f"{pid}:{req.model}")
        response = await amaterialize_response(router.stream(routed), routed) if stream else await router.complete(routed)
        assert "4" in response.text and response.finish_reason == "stop"
        assert response.usage.total_tokens > 0
        assert not any(p.type == "data" for p in response.message.parts)
        assert len(transport.requests) == 1
        assert transport.requests[0].url == case["request"]["url"]
        assert json.loads(transport.requests[0].body) == case["request"]["body"]


def test_gemma_before_and_after_router_ux_same_wire_with_profile(monkeypatch, tmp_path):
    pid = "bedrock-mantle-responses"
    case, body = captured(pid, "gemma_31b_basic_text")
    credentials = tmp_path / "credentials"
    credentials.write_text("[fixture-profile]\naws_access_key_id = AKIDEXAMPLE\naws_secret_access_key = wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY\n", encoding="utf-8")
    env = {"AWS_PROFILE": "fixture-profile", "AWS_REGION": "us-west-2", "AWS_SHARED_CREDENTIALS_FILE": str(credentials),
           "AWS_CONFIG_FILE": str(tmp_path / "absent-config")}
    # In the post-registration checkout a declaration must use a distinct ID.
    # Exact pre-registration issue snippets are also validated at the base SHA.
    legacy = "legacy-mantle-responses"
    before = ProviderDefinition.responses(
        replace(access.BEDROCK_MANTLE_CHAT, provider=legacy,
                supports=EndpointSupport(complete=True, stream=True, responses_api=True),
                host=replace(access.BEDROCK_MANTLE_CHAT.host, base_url="https://bedrock-mantle.{region}.api.aws/openai/v1")),
        compat=OpenAIResponsesCompat.preset("openai"))
    wires = []
    for providers in ((before,), ()):
        route = legacy if providers else pid
        transport = FakeTransport([FakeResponse(200, body)])
        with LMRouter(RouterConfig(providers=providers, credentials={route: "cli"}, env=env, transport=transport)) as router:
            assert router.resolve(f"{route}:google.gemma-4-31b").declared is bool(providers)
            assert "4" in router.complete(request(f"{route}:google.gemma-4-31b")).text
        wire = transport.requests[0]
        wires.append((wire.url, json.loads(wire.body)))
    assert wires[0] == wires[1]
    assert wires[1][0] == case["request"]["url"]


@pytest.mark.parametrize("pid,host,service", DOORS)
def test_rejected_model_never_retries_another_host_or_openai(pid, host, service):
    body = b'{"error":{"code":"model_not_found","message":"No such model"}}'
    transport = FakeTransport([FakeResponse(404, body)])
    with LMRouter(config(pid, transport=transport)) as router:
        with pytest.raises(UnsupportedModelError):
            router.complete(request(f"{pid}:unknown.serialization-only"))
    assert len(transport.requests) == 1
    assert transport.requests[0].url == f"https://{host}/openai/v1/responses"


@pytest.mark.parametrize("pid,host,service", DOORS)
@pytest.mark.asyncio
async def test_async_rejected_model_never_retries_another_host_or_openai(pid, host, service):
    transport = FakeAsyncTransport(b'{"error":{"code":"model_not_found","message":"No such model"}}', status=404)
    async with AsyncLMRouter(config(pid, transport=transport)) as router:
        with pytest.raises(UnsupportedModelError):
            await router.complete(request(f"{pid}:unknown.serialization-only"))
    assert len(transport.requests) == 1
    assert transport.requests[0].url == f"https://{host}/openai/v1/responses"


def test_existing_bedrock_doors_keep_their_dialects_and_roots():
    expected = {
        "bedrock-chat": ("openai-chat", "https://bedrock-runtime.{region}.amazonaws.com/openai/v1", "bedrock"),
        "bedrock-mantle-chat": ("openai-chat", "https://bedrock-mantle.{region}.api.aws/v1", "bedrock-mantle"),
        "bedrock-anthropic": ("anthropic", "https://bedrock-mantle.{region}.api.aws/anthropic/v1", "bedrock-mantle"),
    }
    for pid, (dialect, root, service) in expected.items():
        entry = PROVIDERS[pid]
        assert (entry.dialect, entry.access.host.base_url, entry.access.host.sigv4_service) == (dialect, root, service)
    assert PROVIDERS["bedrock-mantle-chat"].supports.models is True
