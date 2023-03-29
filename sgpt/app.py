"""
shell-gpt: An interface to OpenAI's ChatGPT (GPT-3.5) API

This module provides a simple interface for OpenAI's ChatGPT API using Typer
as the command line interface. It supports different modes of output including
shell commands and code, and allows users to specify the desired OpenAI model
and length and other options of the output. Additionally, it supports executing
shell commands directly from the interface.

API Key is stored locally for easy use in future runs.
"""


import os
from typing import Mapping, List

import typer

from rich import print as rich_print
from rich.rule import Rule
# Click is part of typer.
from click import MissingParameter, BadParameter
from sgpt import config, make_prompt, OpenAIClient
from sgpt.utils import (
    echo_chat_ids,
    echo_chat_messages,
    get_edited_prompt,
)


def get_completion(
    messages: List[Mapping[str, str]],
    temperature: float,
    top_p: float,
    caching: bool,
    chat: str,
):
    api_host = config.get("OPENAI_API_HOST")
    api_key = config.get("OPENAI_API_KEY")
    client = OpenAIClient(api_host, api_key)
    return client.get_completion(
        messages=messages,
        model="gpt-3.5-turbo",
        temperature=temperature,
        top_probability=top_p,
        caching=caching,
        chat_id=chat,
    )


def chat_mode(
    prompt: str,
    temperature: float,
    top_p: float,
    chat_id: str,
    shell: bool,
    code: bool,
    cache: bool,
):
    exists = OpenAIClient.chat_cache.exists(chat_id)
    if exists:
        # If exists, it should contain at least one message.
        chat_history = OpenAIClient.chat_cache.get_messages(chat_id)
        rich_print(Rule(title="Chat History", style="bold magenta"))
        echo_chat_messages(chat_id, shell, code)
        rich_print(Rule(style="bold magenta"))
        typer.echo()
        is_shell_chat = chat_history[0].endswith("###\nCommand:")
        is_code_chat = chat_history[0].endswith("###\nCode:")
        if is_shell_chat and code:
            raise BadParameter(
                f"Chat id:{chat_id} was initiated as shell assistant, can be used with --shell only"
            )
        if is_code_chat and shell:
            raise BadParameter(
                f"Chat id:{chat_id} was initiated as code assistant, can be used with --code only"
            )

        shell, code = is_shell_chat, is_code_chat

    if not prompt:
        prompt = typer.prompt("Enter your prompt", err=True)
    else:
        typer.echo(prompt)

    prompt = make_prompt.chat_mode(prompt, shell, code) \
        if exists else make_prompt.initial(prompt, shell, code)

    while True:
        completion = get_completion(
            [{"role": "user", "content": prompt}],
            temperature,
            top_p,
            cache,
            chat_id
        )
        full_completion = ""
        for word in completion:
            typer.secho(word, fg="magenta", bold=True, nl=False)
            full_completion += word
        typer.secho()

        prompt_text = "Revise or [bold magenta][E][/bold magenta]xecute" if shell else "Prompt"
        rich_print(prompt_text, end="")
        prompt = typer.prompt("")

        if shell:
            if prompt == "execute" or prompt == "e":
                typer.echo()
                os.system(full_completion)
                typer.echo()
                rich_print(Rule(style="bold magenta"))
                prompt = typer.prompt("Enter your prompt", err=True)
            #     prompt = make_prompt.chat_mode(prompt, shell, code=False)
            # else:
        prompt = make_prompt.chat_mode(prompt, shell, code)


def main(
    prompt: str = typer.Argument(None, show_default=False, help="The prompt to generate completions for."),
    temperature: float = typer.Option(1.0, min=0.0, max=1.0, help="Randomness of generated output."),
    top_probability: float = typer.Option(1.0, min=0.1, max=1.0, help="Limits highest probable tokens (words)."),
    chat: str = typer.Option(None, help="Follow conversation with id (chat mode)."),
    list_chat: bool = typer.Option(False, help="List all existing chat ids."),
    shell: bool = typer.Option(False, "--shell", "-s", help="Generate and execute shell command."),
    code: bool = typer.Option(False, help="Provide code as output."),
    editor: bool = typer.Option(False, help="Open $EDITOR to provide a prompt."),
    cache: bool = typer.Option(True, help="Cache completion results."),
) -> None:
    if list_chat:
        echo_chat_ids()
        return

    if editor:
        prompt = get_edited_prompt()

    if chat:
        chat_mode(prompt, temperature, top_probability, chat, shell, code, cache)

    if not prompt and not editor:
        raise MissingParameter(param_hint="PROMPT", param_type="string")

    prompt = make_prompt.initial(prompt, shell, code)

    completion = get_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_probability,
        caching=cache,
        chat=chat,
    )

    full_completion = ""
    for word in completion:
        typer.secho(word, fg="magenta", bold=True, nl=False)
        full_completion += word
    typer.secho()
    if not code and shell and typer.confirm("Execute shell command?"):
        os.system(full_completion)


def entry_point() -> None:
    # Python package entry point defined in setup.py
    typer.run(main)


if __name__ == "__main__":
    entry_point()
