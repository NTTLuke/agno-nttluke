from typing import Optional

import pytest

from agno.agent import Agent, RunResponse  # noqa
from agno.models.azure import AzureOpenAI
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.exa import ExaTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.decorator import tool


def test_tool_use():
    agent = Agent(
        model=AzureOpenAI(id="gpt-4o-mini"),
        tools=[YFinanceTools(cache_results=True)],
        show_tool_calls=True,
        markdown=True,
        telemetry=False,
        monitoring=False,
    )

    response = agent.run("What is the current price of TSLA?")

    # Verify tool usage
    assert any(msg.tool_calls for msg in response.messages)
    assert response.content is not None
    assert "TSLA" in response.content


def test_tool_use_stream():
    agent = Agent(
        model=AzureOpenAI(id="gpt-4o-mini"),
        tools=[YFinanceTools(cache_results=True)],
        show_tool_calls=True,
        markdown=True,
        telemetry=False,
        monitoring=False,
    )

    response_stream = agent.run("What is the current price of TSLA?", stream=True)

    responses = []
    tool_call_seen = False

    for chunk in response_stream:
        assert isinstance(chunk, RunResponse)
        responses.append(chunk)
        if chunk.tools:
            if any(tc.get("tool_name") for tc in chunk.tools):
                tool_call_seen = True

    assert len(responses) > 0
    assert tool_call_seen, "No tool calls observed in stream"
    assert any("TSLA" in r.content for r in responses if r.content)


@pytest.mark.asyncio
async def test_async_tool_use():
    agent = Agent(
        model=AzureOpenAI(id="gpt-4o-mini"),
        tools=[YFinanceTools(cache_results=True)],
        show_tool_calls=True,
        markdown=True,
        telemetry=False,
        monitoring=False,
    )

    response = await agent.arun("What is the current price of TSLA?")

    # Verify tool usage
    assert any(msg.tool_calls for msg in response.messages if msg.role == "assistant")
    assert response.content is not None
    assert "TSLA" in response.content


@pytest.mark.asyncio
async def test_async_tool_use_stream():
    agent = Agent(
        model=AzureOpenAI(id="gpt-4o-mini"),
        tools=[YFinanceTools(cache_results=True)],
        show_tool_calls=True,
        markdown=True,
        telemetry=False,
        monitoring=False,
    )

    response_stream = await agent.arun("What is the current price of TSLA?", stream=True)

    responses = []
    tool_call_seen = False

    async for chunk in response_stream:
        assert isinstance(chunk, RunResponse)
        responses.append(chunk)
        if chunk.tools:
            if any(tc.get("tool_name") for tc in chunk.tools):
                tool_call_seen = True

    assert len(responses) > 0
    assert tool_call_seen, "No tool calls observed in stream"
    assert any("TSLA" in r.content for r in responses if r.content)


def test_parallel_tool_calls():
    agent = Agent(
        model=AzureOpenAI(id="gpt-4o-mini"),
        tools=[YFinanceTools(cache_results=True)],
        show_tool_calls=True,
        markdown=True,
        telemetry=False,
        monitoring=False,
    )

    response = agent.run("What is the current price of TSLA and AAPL?")

    # Verify tool usage
    tool_calls = []
    for msg in response.messages:
        if msg.tool_calls:
            tool_calls.extend(msg.tool_calls)
    assert len([call for call in tool_calls if call.get("type", "") == "function"]) == 2  # Total of 2 tool calls made
    assert response.content is not None
    assert "TSLA" in response.content and "AAPL" in response.content


def test_multiple_tool_calls():
    agent = Agent(
        model=AzureOpenAI(id="gpt-4o-mini"),
        tools=[YFinanceTools(cache_results=True), DuckDuckGoTools(cache_results=True)],
        show_tool_calls=True,
        markdown=True,
        telemetry=False,
        monitoring=False,
    )

    response = agent.run("What is the current price of TSLA and what is the latest news about it?")

    # Verify tool usage
    tool_calls = []
    for msg in response.messages:
        if msg.tool_calls:
            tool_calls.extend(msg.tool_calls)
    assert len([call for call in tool_calls if call.get("type", "") == "function"]) == 2  # Total of 2 tool calls made
    assert response.content is not None
    assert "TSLA" in response.content and "latest news" in response.content.lower()


def test_tool_call_custom_tool_no_parameters():
    def get_the_weather_in_tokyo():
        """
        Get the weather in Tokyo
        """
        return "It is currently 70 degrees and cloudy in Tokyo"

    agent = Agent(
        model=AzureOpenAI(id="gpt-4o-mini"),
        tools=[get_the_weather_in_tokyo],
        show_tool_calls=True,
        markdown=True,
        telemetry=False,
        monitoring=False,
    )

    response = agent.run("What is the weather in Tokyo?")

    # Verify tool usage
    assert any(msg.tool_calls for msg in response.messages)
    assert response.content is not None
    assert "70" in response.content


def test_tool_call_custom_tool_optional_parameters():
    def get_the_weather(city: Optional[str] = None):
        """
        Get the weather in a city

        Args:
            city: The city to get the weather for
        """
        if city is None:
            return "It is currently 70 degrees and cloudy in Tokyo"
        else:
            return f"It is currently 70 degrees and cloudy in {city}"

    agent = Agent(
        model=AzureOpenAI(id="gpt-4o-mini"),
        tools=[get_the_weather],
        show_tool_calls=True,
        markdown=True,
        telemetry=False,
        monitoring=False,
    )

    response = agent.run("What is the weather in Paris?")

    # Verify tool usage
    assert any(msg.tool_calls for msg in response.messages)
    assert response.content is not None
    assert "70" in response.content


def test_tool_call_list_parameters():
    agent = Agent(
        model=AzureOpenAI(id="gpt-4o-mini"),
        tools=[ExaTools()],
        instructions="Use a single tool call if possible",
        show_tool_calls=True,
        markdown=True,
        telemetry=False,
        monitoring=False,
    )

    response = agent.run(
        "What are the papers at https://arxiv.org/pdf/2307.06435 and https://arxiv.org/pdf/2502.09601 about?"
    )

    # Verify tool usage
    assert any(msg.tool_calls for msg in response.messages)
    tool_calls = []
    for msg in response.messages:
        if msg.tool_calls:
            tool_calls.extend(msg.tool_calls)
    for call in tool_calls:
        if call.get("type", "") == "function":
            assert call["function"]["name"] in ["get_contents", "exa_answer"]
    assert response.content is not None



# Test case for tool returning None
def test_tool_returns_none():
    """Tests that the agent handles a tool returning None correctly."""

    @tool
    def tool_that_returns_none(query: str) -> None:
        """A simple tool that processes a query but returns None."""
        print(f"Processed query: {query}, returning None")
        return None

    agent = Agent(
        model=AzureOpenAI(id="gpt-4o-mini"),
        tools=[tool_that_returns_none],
        show_tool_calls=True,
        markdown=True,
        telemetry=False,
        monitoring=False,
    )

    # Run the agent with a prompt designed to trigger the tool
    response = agent.run("Process this query using the special tool: hello")

    # Verify that a tool call happened and the corresponding ToolMessage has empty content
    tool_call_made = False
    tool_message_found_with_empty_content = False
    tool_id = None

    for msg in response.messages:
        if msg.role == "assistant" and msg.tool_calls:
            tool_call_made = True
            # Assuming one tool call for simplicity in this test
            if len(msg.tool_calls) == 1:
                tool_id = msg.tool_calls[0].get("id")
        elif msg.role == "tool" and msg.tool_call_id == tool_id:
            assert msg.content == "", f"ToolMessage content should be empty string, but got: {msg.content!r}"
            tool_message_found_with_empty_content = True

    assert tool_call_made, "Assistant message with tool call not found."
    assert tool_message_found_with_empty_content, "Tool message with empty content not found."
    # Also assert that the final response content is not None (agent should summarize or respond)
    assert response.content is not None, "Agent final response content should not be None."


# To run only this test file, use one of the following commands:
# pytest libs/agno/tests/integration/models/azure/openai/test_tool_use.py -v
# 
# To run a specific test in this file:
# pytest libs/agno/tests/integration/models/azure/openai/test_tool_use.py::test_tool_returns_none -v