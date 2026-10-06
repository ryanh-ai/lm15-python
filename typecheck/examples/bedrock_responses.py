"""Cloud-hosts documentation: built-in sync/async Responses registrations."""
from lm15 import AsyncLMRouter, Config, LMRouter, Message, Reasoning, Request, RouterConfig

cfg = RouterConfig(credentials={"bedrock-mantle-responses": "cli"})
request = Request(
    model="bedrock-mantle-responses:google.gemma-4-31b",
    messages=(Message.user("What is 2 + 2?"),),
    config=Config(max_tokens=256, reasoning=Reasoning(effort="low"), store=False),
)
with LMRouter(cfg) as router:
    response = router.complete(request)
    print(response.text)


async def complete_async() -> None:
    async with AsyncLMRouter(cfg) as router:
        response = await router.complete(request)
        print(response.text)
