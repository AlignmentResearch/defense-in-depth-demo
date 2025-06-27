import random
from collections.abc import Sequence
from dataclasses import dataclass

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from robust_llm.config.dataset_configs import MessageFilter
from robust_llm.utils import deterministic_string


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def to_dict(self):
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class MessageList:
    messages: list[Message]

    def __post_init__(self):
        if len(self.messages) == 0:
            raise ValueError("MessageList must contain at least one message")

        self._check_message_structure()

    def _check_message_structure(self):
        """Check that messages are: optional system, then alternating user/assistant."""
        if len(self.messages) == 1:
            # Single message already validated in __post_init__
            return

        start_idx = 0
        if self.messages[0].role == "system":
            start_idx = 1

        # After optional system message, messages should alternate between user
        # and assistant.
        for i, message in enumerate(self.messages[start_idx:]):
            if i % 2 == 0 and message.role != "user":
                raise ValueError(
                    "Messages must alternate between user and assistant after"
                    " optional system message."
                    f" Got roles: {[m.role for m in self.messages]}"
                )
            elif i % 2 == start_idx + 1 and message.role != "assistant":
                raise ValueError(
                    "Messages must alternate between user and assistant after"
                    " optional system message."
                    f" Got roles: {[m.role for m in self.messages]}"
                )

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index: int) -> Message:
        return self.messages[index]

    def __iter__(self):
        return iter(self.messages)

    def to_dicts(self):
        return [message.to_dict() for message in self.messages]

    def append(self, message: Message):
        """Add a new message to the end of the message list."""
        self.messages.append(message)

    def __add__(self, other: "Message | MessageList") -> "MessageList":
        """Concatenate a Message or MessageList to create a new MessageList."""
        if isinstance(other, Message):
            return MessageList(self.messages + [other])
        elif isinstance(other, MessageList):
            return MessageList(self.messages + other.messages)
        raise TypeError(f"Cannot add MessageList with {type(other)}")

    def __iadd__(self, other: "Message | MessageList") -> "MessageList":
        """In-place concatenation of a Message or MessageList."""
        if isinstance(other, Message):
            self.messages.append(other)
        elif isinstance(other, MessageList):
            self.messages.extend(other.messages)
        else:
            raise TypeError(f"Cannot add MessageList with {type(other)}")
        return self

    def to_text(self, delimiter: str = "", include_role: bool = False) -> str:
        """Convert the MessageList to a string with no formatting.

        Args:
            delimiter: The delimiter to use between messages.
            include_role: Whether to include the role in the output.
        """
        return delimiter.join(
            [
                (
                    f"{message.role}: {message.content}"
                    if include_role
                    else message.content
                )
                for message in self.messages
            ]
        )

    def format(
        self,
        message_filter: MessageFilter,
        tokenizer: PreTrainedTokenizerBase,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        seed: int | None = None,
    ) -> str:
        """Format the message list as a string for a given message filter.

        Args:
            message_filter: The message filter to use to format the message list.
            tokenizer: The tokenizer with a chat template if needed to format the
                message list.
            add_generation_prompt: Whether to add a generation prompt to the message
                list. Set to False for GCG attacks.
            continue_final_message: Whether to continue the final message, as
                opposed to ending it (e.g., with '<|im_end|>' for Qwen).
                Set to True for GCG attacks.
            seed: The seed to use to format the message list.
        """
        if message_filter == MessageFilter.TRANSCRIPT:
            assert seed is not None
            rng = random.Random(seed)
            formatted = self.format_transcript(rng=rng)
        elif tokenizer.chat_template is not None:
            templated = tokenizer.apply_chat_template(
                self.to_dicts(),
                tokenize=False,
                add_generation_prompt=add_generation_prompt
                and self.messages[-1].role == "user",
                continue_final_message=continue_final_message,
                enable_thinking=False,
            )
            assert isinstance(templated, str)
            formatted = templated
        else:
            formatted = self.to_text()
        return formatted

    def format_transcript(self, rng: random.Random | None = None) -> str:
        """Format the message list as a transcript for a transcript filter."""
        if len(self.messages) == 1:
            return self.messages[0].content
        assert rng is not None
        user_id = deterministic_string(rng)
        assistant_id = deterministic_string(rng)
        messages = [
            (
                f"<input id={user_id}>{message.content}</input id={user_id}>"
                if message.role == "user"
                else f"<output id={assistant_id}>{message.content}</output id={assistant_id}>"  # noqa: E501
            )
            for message in self.messages
            if message.role != "system"
        ]
        return " ".join(messages)

    def get_first_user_message(self) -> Message:
        """Get the first user message."""
        for message in self.messages:
            if message.role == "user":
                return message
        raise ValueError("No user message found in the message list.")

    def replace_first_user_message(self, new_content: str) -> "MessageList":
        """Replace the first user message with a new message."""
        messages_to_keep = []
        for i, message in enumerate(self.messages):
            if message.role == "user":
                messages_to_keep.append(Message(role="user", content=new_content))
                return MessageList(messages_to_keep + self.messages[i + 1 :])
            else:
                messages_to_keep.append(message)
        raise ValueError("No user message found in the message list.")

    @staticmethod
    def from_user_message(content: str) -> "MessageList":
        return MessageList(messages=[Message(role="user", content=content)])

    def get_final_response(self) -> str:
        """Get the last message, which should be an assistant message."""
        if self.messages[-1].role != "assistant":
            raise ValueError(
                "get_final_response expects the last message to be from 'assistant'."
            )
        return self.messages[-1].content

    def update_system_prompt(self, system_prompt: str | None) -> "MessageList":
        """Update the system prompt in the message list."""
        if self.messages[0].role == "system" and system_prompt is None:
            return MessageList(messages=self.messages[1:])
        elif self.messages[0].role == "system" and system_prompt is not None:
            return MessageList(
                messages=[Message(role="system", content=system_prompt)]
                + self.messages[1:]
            )
        elif self.messages[0].role != "system" and system_prompt is not None:
            return MessageList(
                messages=[Message(role="system", content=system_prompt)] + self.messages
            )
        elif self.messages[0].role != "system" and system_prompt is None:
            return self
        else:
            raise ValueError("Invalid message list structure.")

    def filter(self, filter: MessageFilter) -> "MessageList":
        """Filter the message list based on the filter."""
        return MessageList(
            [
                message
                for message in self.messages
                if (message.role == "system" and filter != MessageFilter.OUTPUT)
                or (
                    message.role == "user"
                    and filter
                    in (
                        MessageFilter.INPUT,
                        MessageFilter.TRANSCRIPT,
                        MessageFilter.IDENTITY,
                    )
                )
                or (
                    message.role == "assistant"
                    and filter
                    in (
                        MessageFilter.OUTPUT,
                        MessageFilter.TRANSCRIPT,
                        MessageFilter.IDENTITY,
                    )
                )
            ]
        )


@dataclass
class ConversationList:
    """A collection of message lists and the filter to use to format them."""

    message_lists: Sequence[MessageList]
    message_filter: MessageFilter = MessageFilter.INPUT
    add_generation_prompt: bool = True

    def __len__(self):
        return len(self.message_lists)

    def __getitem__(self, index: int) -> MessageList:
        return self.message_lists[index]

    def __iter__(self):
        return iter(self.message_lists)


if __name__ == "__main__":
    messages = MessageList(
        [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
    )
    print(messages.to_dicts())
