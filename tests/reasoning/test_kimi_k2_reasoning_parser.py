# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "kimi_k2"
start_token = "<think>"
end_token = "</think>"
tool_call_start_token = "<|tool_calls_section_begin|>"

# Use Kimi K2.5 model tokenizer for testing
REASONING_MODEL_NAME = "moonshotai/Kimi-K2.5"


@pytest.fixture(scope="module")
def kimi_k2_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME, trust_remote_code=True)


# Standard cases (inherited from DeepSeek R1)
SIMPLE_REASONING = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}

COMPLETE_REASONING = {
    "output": "This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}

REASONING_WITH_THINK = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}

# NEW: Implicit think end cases (tool call starts without explicit </think>)
IMPLICIT_END_WITH_TOOL_CALL = {
    "output": "<think>This is reasoning<|tool_calls_section_begin|>tool content here",
    "reasoning": "This is reasoning",
    "content": "<|tool_calls_section_begin|>tool content here",
    "is_reasoning_end": True,
}

IMPLICIT_END_WITH_TOOL_CALL_NO_EXPLICIT_START = {
    "output": "This is reasoning<|tool_calls_section_begin|>tool content here",
    "reasoning": "This is reasoning",
    "content": "<|tool_calls_section_begin|>tool content here",
    "is_reasoning_end": True,
}

IMPLICIT_END_SINGULAR_VARIANT = {
    "output": "<think>This is reasoning<|tool_call_section_begin|>tool content here",
    "reasoning": "This is reasoning",
    "content": "<|tool_call_section_begin|>tool content here",
    "is_reasoning_end": True,
}

EXPLICIT_END_BEFORE_TOOL_CALL = {
    "output": "<think>This is reasoning</think>content<|tool_calls_section_begin|>",
    "reasoning": "This is reasoning",
    "content": "content<|tool_calls_section_begin|>",
    "is_reasoning_end": True,
}

TOOL_CALL_IN_MIDDLE = {
    "output": "<think>reasoning<|tool_calls_section_begin|>tool<|tool_calls_section_end|>more",
    "reasoning": "reasoning",
    "content": "<|tool_calls_section_begin|>tool<|tool_calls_section_end|>more",
    "is_reasoning_end": True,
}

NO_TOOL_CALL_NO_END = {
    "output": "<think>This is reasoning without end",
    "reasoning": "This is reasoning without end",
    "content": None,
    "is_reasoning_end": False,
}

TOOL_CALL_ONLY = {
    "output": "<|tool_calls_section_begin|>tool content",
    "reasoning": "",
    "content": "<|tool_calls_section_begin|>tool content",
    "is_reasoning_end": True,
}

# Streaming-specific expectations
NO_TOOL_CALL_NO_END_STREAMING = {
    "output": "<think>This is reasoning without end",
    "reasoning": "This is reasoning without end",
    "content": None,
    "is_reasoning_end": False,
}

TOOL_CALL_ONLY_STREAMING = {
    "output": "<|tool_calls_section_begin|>tool content",
    "reasoning": None,
    "content": "<|tool_calls_section_begin|>tool content",
    "is_reasoning_end": True,
}

TEST_CASES = [
    # Standard cases (should work like DeepSeek R1)
    pytest.param(
        False,
        SIMPLE_REASONING,
        id="simple_reasoning",
    ),
    pytest.param(
        True,
        SIMPLE_REASONING,
        id="simple_reasoning_streaming",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        id="complete_reasoning",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        id="complete_reasoning_streaming",
    ),
    pytest.param(
        False,
        REASONING_WITH_THINK,
        id="reasoning_with_think",
    ),
    pytest.param(
        True,
        REASONING_WITH_THINK,
        id="reasoning_with_think_streaming",
    ),
    # NEW: Implicit think end cases
    pytest.param(
        False,
        IMPLICIT_END_WITH_TOOL_CALL,
        id="implicit_end_with_tool_call",
    ),
    pytest.param(
        True,
        IMPLICIT_END_WITH_TOOL_CALL,
        id="implicit_end_with_tool_call_streaming",
    ),
    pytest.param(
        False,
        IMPLICIT_END_WITH_TOOL_CALL_NO_EXPLICIT_START,
        id="implicit_end_no_explicit_start",
    ),
    pytest.param(
        True,
        IMPLICIT_END_WITH_TOOL_CALL_NO_EXPLICIT_START,
        id="implicit_end_no_explicit_start_streaming",
    ),
    pytest.param(
        False,
        IMPLICIT_END_SINGULAR_VARIANT,
        id="implicit_end_singular_variant",
    ),
    pytest.param(
        True,
        IMPLICIT_END_SINGULAR_VARIANT,
        id="implicit_end_singular_variant_streaming",
    ),
    pytest.param(
        False,
        EXPLICIT_END_BEFORE_TOOL_CALL,
        id="explicit_end_before_tool_call",
    ),
    pytest.param(
        True,
        EXPLICIT_END_BEFORE_TOOL_CALL,
        id="explicit_end_before_tool_call_streaming",
    ),
    pytest.param(
        False,
        TOOL_CALL_IN_MIDDLE,
        id="tool_call_in_middle",
    ),
    pytest.param(
        True,
        TOOL_CALL_IN_MIDDLE,
        id="tool_call_in_middle_streaming",
    ),
    pytest.param(
        False,
        NO_TOOL_CALL_NO_END,
        id="no_tool_call_no_end",
    ),
    pytest.param(
        True,
        NO_TOOL_CALL_NO_END_STREAMING,
        id="no_tool_call_no_end_streaming",
    ),
    pytest.param(
        False,
        TOOL_CALL_ONLY,
        id="tool_call_only",
    ),
    pytest.param(
        True,
        TOOL_CALL_ONLY_STREAMING,
        id="tool_call_only_streaming",
    ),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    kimi_k2_tokenizer,
):
    output = kimi_k2_tokenizer.tokenize(param_dict["output"])
    # decode everything to tokens
    output_tokens: list[str] = [
        kimi_k2_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        kimi_k2_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    output_ids = kimi_k2_tokenizer.convert_tokens_to_ids(output)
    is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == param_dict["is_reasoning_end"]

    # Test extract_content_ids
    if param_dict["content"] is not None:
        content_ids = parser.extract_content_ids(output_ids)
        expected_content_ids = kimi_k2_tokenizer.convert_tokens_to_ids(
            kimi_k2_tokenizer.tokenize(param_dict["content"])
        )
        assert content_ids == expected_content_ids
    else:
        content_ids = parser.extract_content_ids(output_ids)
        assert content_ids == []


def test_parser_initialization(kimi_k2_tokenizer):
    """Test that the parser initializes correctly with tool call tokens."""
    parser = ReasoningParserManager.get_reasoning_parser(parser_name)(kimi_k2_tokenizer)
    
    # Verify tool call tokens are registered
    assert parser.tool_calls_start_token is not None
    assert parser.tool_calls_start_token_id is not None
    assert len(parser.tool_calls_start_token_ids) > 0
    
    # Verify we still have the standard think tokens
    assert parser.start_token == "<think>"
    assert parser.end_token == "</think>"
    assert parser.start_token_id is not None
    assert parser.end_token_id is not None
