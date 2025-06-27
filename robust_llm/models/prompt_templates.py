from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property

from robust_llm.dist_utils import DistributedRNG


class ChatRole(Enum):
    USER = 0
    ASSISTANT = 1


class PromptTemplate:
    r"""This is a general class for prompt templates.

    Should encompass both chat models and non-chat models.

    The basic idea is that there is some part before user input, and
    some part after user input but before model input, and that should
    be all of the content in the prompt.

    NOTE(2025/05/20): We recently changed this to use .format rather than
    adding strings together. This allows more precision in attack position,
    and to have the same attack string in multiple places.

    For example, for a simple chat format:
        before_attack="User: Hi, I'm an user! "
        after_attack="!ATTACK_STRING_PLACEHOLDER!\nAssistant:"
    """

    def __init__(
        self,
        before_attack: str,
        after_attack: str,
        add_attack_string_placeholder: bool = True,
    ):
        self.before_attack = before_attack
        self.add_attack_string_placeholder = add_attack_string_placeholder
        self.after_attack = after_attack
        if add_attack_string_placeholder:
            self.after_attack = "!ATTACK_STRING_PLACEHOLDER!" + self.after_attack

    def build_prompt(self, *, attack_text: str = "", target: str = "") -> str:
        # prompt = self.before_attack + attack_text + self.after_attack + target
        prompt = (self.before_attack + self.after_attack).replace(
            "!ATTACK_STRING_PLACEHOLDER!", attack_text
        )
        return prompt + target


@dataclass(frozen=True)
class AttackChunks:
    unmodifiable_prefix: str
    modifiable_infix: str
    unmodifiable_suffix: str

    def get_prompt_template(
        self, perturb_min: float | None, perturb_max: float | None, rng: DistributedRNG
    ) -> PromptTemplate:
        """Split the chunks into before/after attack and return a prompt template.

        Args:
            perturb_min ([0, 1]): The earliest fraction of the infix at which to
                attack. If None, then we assume !ATTACK_STRING_PLACEHOLDER! is
                already present in the infix.
            perturb_max ([0, 1]): The latest fraction of the infix at which to
                attack. If None, then we assume !ATTACK_STRING_PLACEHOLDER! is
                already present in the infix.
            rng: A random number generator.

        Returns:
            PromptTemplate: The prompt template for the attack.

        A few remarks:
        - If perturb_min and perturb_max are sufficiently close as to correspond to the
            same position, the attack position will be deterministic.
        - If perturb_min is 1, then this will be a suffix attack.
        - If perturb_max is 0, then this will be a prefix attack.
        """
        if perturb_min is not None:
            assert perturb_max is not None
            assert 0 <= perturb_min <= perturb_max <= 1
            perturb_length = len(self.modifiable_infix)
            perturb_min_position = int(perturb_length * perturb_min)
            perturb_max_position = int(perturb_length * perturb_max)
            perturb_position = rng.randint(perturb_min_position, perturb_max_position)
            # We insert !ATTACK_STRING_PLACEHOLDER! at the perturb_position
            modifiable_infix = (
                self.modifiable_infix[:perturb_position]
                + "!ATTACK_STRING_PLACEHOLDER!"
                + self.modifiable_infix[perturb_position:]
            )
        else:
            perturb_position = self.modifiable_infix.index(
                "!ATTACK_STRING_PLACEHOLDER!"
            )
            modifiable_infix = self.modifiable_infix
        return PromptTemplate(
            before_attack=self.unmodifiable_prefix
            + modifiable_infix[:perturb_position],
            after_attack=modifiable_infix[perturb_position:] + self.unmodifiable_suffix,
            # We added the attack string placeholder above.
            add_attack_string_placeholder=False,
        )


@dataclass
class Conversation:
    """A template for generating chat prompts.

    Attributes:
        prompt_prefix: The prefix for the entire prompt.
        system_prefix: The prefix for the system prompt.
        system_suffix: The suffix for the system prompt.
        user_prefix: The prefix for the user input.
        user_suffix: The suffix for the user input, including the prefix
            for the assistant response.
        assistant_prefix: The prefix for the assistant response.
        assistant_suffix: The suffix for the assistant response.
        system_prompt: The system prompt to include in the prompt.
        repeat_prompt_prefix: Whether to repeat the prompt prefix for
            each user/assistant interaction. Defaults to False, i.e. the prompt
            prefix is only included once at the very start.
        require_leading_whitespace: Whether to require leading whitespace
            in the assistant response. Defaults to False.
        special_strings (list[str]): A list of special strings that act as delimiters
            for different components of the chat. These are used to clean the prompt
            when we just want text generations in natural language.

    """

    prompt_prefix: str
    system_prefix: str
    system_suffix: str
    user_prefix: str
    user_suffix: str
    assistant_prefix: str
    assistant_suffix: str
    system_prompt: str | None = None
    repeat_prompt_prefix: bool = False
    require_leading_whitespace: bool = False
    _messages: list[tuple[ChatRole, str]] = field(default_factory=list)

    def _append_message(self, role: ChatRole, message: str):
        assert (
            role.value == len(self._messages) % 2
        ), f"Expected role {ChatRole(len(self._messages) % 2)} next but got {role}"
        self._messages.append((role, message))
        return self

    def append_user_message(self, message: str):
        return self._append_message(ChatRole.USER, message)

    def append_assistant_message(self, message: str):
        if self.require_leading_whitespace and message and not message.startswith(" "):
            message = " " + message
        return self._append_message(ChatRole.ASSISTANT, message)

    def append_to_last_message(self, text: str):
        role, message = self._messages[-1]
        self._messages[-1] = (role, message + text)
        return self

    def get_system_message(self) -> str:
        if self.system_prompt is None:
            return ""
        return f"{self.system_prefix}{self.system_prompt}{self.system_suffix}"

    def wrap_prompt_template(self, prompt_template: PromptTemplate):
        before_attack = self.prompt_prefix + self.get_system_message()
        before_attack += self.user_prefix
        before_attack += prompt_template.before_attack
        after_attack = (
            f"{prompt_template.after_attack}{self.user_suffix}{self.assistant_prefix}"
        )
        return PromptTemplate(
            before_attack=before_attack,
            after_attack=after_attack,
            # attack string should already be there.
            add_attack_string_placeholder=False,
        )

    def get_prompt(self, skip_last_suffix: bool = True):
        """Get the full prompt for the conversation.

        Args:
            skip_last_suffix (bool, optional): Whether to skip the last suffix.
                Defaults to True.
                This is normally set to false when generating from a prompt.
        """
        prompt = self.prompt_prefix + self.get_system_message()
        for idx, (role, message) in enumerate(self._messages):
            if self.repeat_prompt_prefix and role == ChatRole.USER and idx > 0:
                prompt += self.prompt_prefix
            if role == ChatRole.USER:
                prompt += self.user_prefix
            else:
                prompt += self.assistant_prefix
            prompt += message
            if skip_last_suffix and (idx == len(self._messages) - 1):
                break
            if role == ChatRole.USER:
                prompt += self.user_suffix
            else:
                prompt += self.assistant_suffix
        return prompt

    @cached_property
    def special_strings(self) -> list[str]:
        return [
            sub_str
            for component in [
                self.prompt_prefix,
                self.system_prefix,
                self.system_suffix,
                self.user_prefix,
                self.user_suffix,
                self.assistant_prefix,
                self.assistant_suffix,
            ]
            for sub_str in component.split("\n")
            if sub_str
        ]

    def clean_special_strings(self, text: str) -> str:
        for sub_str in self.special_strings:
            text = text.replace(sub_str, "")
        return text
