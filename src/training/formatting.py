from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional


class ExampleFormatter(ABC):
    """
    Abstract interface for turning a raw dataset row into (prompt, target).
    """

    @abstractmethod
    def format_example(self, example: Dict) -> Tuple[str, str]:
        """
        Return (prompt, target) strings for a single example.
        """
        ...


class InstructionFormatter(ExampleFormatter):
    """
    Generic "instruction + optional input -> response" formatter.

    You just tell it which fields are instruction / input / target,
    and optionally how to format the prompt.
    """

    def __init__(
        self,
        instruction_field: str,
        target_field: str,
        input_field: Optional[str] = None,
        template: Optional[str] = None,
    ) -> None:
        """
        template can use {instruction} and {input}, e.g.:

            "Instruction: {instruction}\nInput: {input}\n\nResponse:"

        If template is None, a simple default is used.
        """
        self.instruction_field = instruction_field
        self.input_field = input_field
        self.target_field = target_field
        self.template = template

    def format_example(self, example: Dict) -> Tuple[str, str]:
        instruction = example.get(self.instruction_field, "")
        inp = example.get(self.input_field, "") if self.input_field else ""
        target = example.get(self.target_field, "")

        if self.template is not None:
            prompt = self.template.format(instruction=instruction, input=inp)
        else:
            if inp:
                prompt = (
                    f"Instruction: {instruction}\n" f"Input: {inp}\n\n" f"Response:"
                )
            else:
                prompt = f"Instruction: {instruction}\n\nResponse:"

        return prompt, target
