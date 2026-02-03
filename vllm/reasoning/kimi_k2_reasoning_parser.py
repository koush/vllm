# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import DeltaMessage
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.tokenizers import TokenizerLike


class KimiK2ReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for Kimi K2 series models.

    The Kimi K2 models use <think>...</think> tokens like DeepSeek R1,
    but may end reasoning implicitly when a tool call starts with
    <|tool_calls_section_begin|> token, without an explicit </think> token.

    This parser extends DeepSeekR1ReasoningParser to handle this implicit
    reasoning end when tool calls begin.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        # Tool call section markers (support both plural and singular variants)
        self.tool_calls_start_token_variants = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_section_begin|>",  # singular variant
        ]

        # Get token IDs for all variants
        self.tool_calls_start_token_ids = [
            tid
            for variant in self.tool_calls_start_token_variants
            if (tid := self.vocab.get(variant)) is not None
        ]

        # Use the first available variant as the primary token
        self.tool_calls_start_token = None
        self.tool_calls_start_token_id = None
        for variant in self.tool_calls_start_token_variants:
            tid = self.vocab.get(variant)
            if tid is not None:
                self.tool_calls_start_token = variant
                self.tool_calls_start_token_id = tid
                break

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """
        Check if reasoning ends either by explicit </think> or implicit
        tool call start token.
        """
        # First check for explicit end token (</think>)
        if super().is_reasoning_end(input_ids):
            return True

        # Check if any tool call start token variant appears after start token
        start_token_id = self.start_token_id
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                # Found start token, check if any tool call token appears after it
                for tool_call_id in self.tool_calls_start_token_ids:
                    if tool_call_id in input_ids[i + 1 :]:
                        return True
                return False
            # If we find explicit end token before start, use parent logic
            if input_ids[i] == self.end_token_id:
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        """
        Check if reasoning ends in streaming mode, either by explicit </think>
        or implicit tool call start token in the delta.
        """
        # Check for explicit end token
        if super().is_reasoning_end_streaming(input_ids, delta_ids):
            return True

        # Check if any tool call start token variant appears in delta
        for tool_call_id in self.tool_calls_start_token_ids:
            if tool_call_id in delta_ids:
                return True

        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token IDs after the reasoning section.
        Handles both explicit </think> and implicit tool call start.
        """
        # First try parent implementation (explicit end token)
        content_ids = super().extract_content_ids(input_ids)
        if content_ids:
            return content_ids

        # Check for implicit end via tool call start token
        if not self.tool_calls_start_token_ids:
            return []

        # Find the first occurrence of any tool call start token
        earliest_tool_call_idx = None
        for tool_call_id in self.tool_calls_start_token_ids:
            try:
                idx = input_ids.index(tool_call_id)
                if earliest_tool_call_idx is None or idx < earliest_tool_call_idx:
                    earliest_tool_call_idx = idx
            except ValueError:
                continue

        if earliest_tool_call_idx is not None:
            # Return everything from the tool call start token onwards
            return input_ids[earliest_tool_call_idx:]

        return []

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning in streaming mode, treating tool call start as
        implicit </think> token.
        """
        # Check if any tool call start token variant appears in delta
        has_tool_call_start = any(
            tid in delta_token_ids for tid in self.tool_calls_start_token_ids
        )

        if has_tool_call_start:
            # Find which variant and its position in delta_text
            tool_call_token = None
            tool_call_idx = -1
            for variant in self.tool_calls_start_token_variants:
                idx = delta_text.find(variant)
                if idx != -1:
                    if tool_call_idx == -1 or idx < tool_call_idx:
                        tool_call_token = variant
                        tool_call_idx = idx

            if tool_call_token and tool_call_idx != -1:
                # Check if we're already past the thinking section
                if self.start_token_id in previous_token_ids:
                    if self.end_token_id not in previous_token_ids:
                        # We're in thinking section, tool call ends it implicitly
                        reasoning = delta_text[:tool_call_idx]
                        content = delta_text[tool_call_idx:]
                        return DeltaMessage(
                            reasoning=reasoning if reasoning else None,
                            content=content if content else None,
                        )
                    else:
                        # Already ended thinking, this is content
                        return DeltaMessage(content=delta_text)
                elif self.start_token_id in delta_token_ids:
                    # Start token in same delta as tool call token
                    # Extract reasoning between start and tool call
                    start_idx = delta_text.find(self.start_token)
                    if start_idx != -1:
                        reasoning = delta_text[
                            start_idx + len(self.start_token) : tool_call_idx
                        ]
                        content = delta_text[tool_call_idx:]
                        return DeltaMessage(
                            reasoning=reasoning if reasoning else None,
                            content=content if content else None,
                        )

        # Fall back to parent implementation for normal cases
        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

    def extract_reasoning(
        self, model_output: str, request
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete model output.
        Handles implicit reasoning end when tool call starts.
        """
        # Check for tool call start tokens
        earliest_tool_call_idx = -1
        tool_call_token_found = None
        for variant in self.tool_calls_start_token_variants:
            idx = model_output.find(variant)
            if idx != -1:
                if earliest_tool_call_idx == -1 or idx < earliest_tool_call_idx:
                    earliest_tool_call_idx = idx
                    tool_call_token_found = variant

        # If tool call token found, check if it's before </think>
        end_token_idx = model_output.find(self.end_token)

        if (
            tool_call_token_found
            and earliest_tool_call_idx != -1
            and (end_token_idx == -1 or earliest_tool_call_idx < end_token_idx)
        ):
            # Tool call starts before explicit end token (or no end token)
            # Remove <think> if present
            model_output_parts = model_output.partition(self.start_token)
            model_output = (
                model_output_parts[2]
                if model_output_parts[1]
                else model_output_parts[0]
            )

            # Split at tool call token
            reasoning, _, content = model_output.partition(tool_call_token_found)
            # Add the tool call token back to content
            final_content = tool_call_token_found + content
            return reasoning, final_content

        # Fall back to parent implementation for normal cases
        return super().extract_reasoning(model_output, request)
