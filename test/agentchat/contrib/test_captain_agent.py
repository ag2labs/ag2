import os
import re
import sys

import pytest

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.captain_agent import CaptainAgent
from autogen.agentchat.contrib.captain_user_proxy_agent import CaptainUserProxyAgent
from autogen.agentchat.contrib.tool_retriever import ToolBuilder

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from conftest import MOCK_OPEN_AI_API_KEY, reason, skip_openai  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from test_assistant_agent import KEY_LOC, OAI_CONFIG_LIST  # noqa: E402

try:
    import chromadb
    import huggingface_hub
except ImportError:
    skip = True
else:
    skip = False


@pytest.mark.skipif(
    skip_openai,
    reason=reason,
)
def test_captain_agent_from_scratch():
    config_list = config_list_from_json(
        OAI_CONFIG_LIST,
        file_location=KEY_LOC,
        filter_dict={
            "tags": ["gpt-4"],
        },
    )
    general_llm_config = {
        "temperature": 0,
        "config_list": config_list,
    }
    nested_mode_config = {
        "autobuild_init_config": {
            "config_file_or_env": os.path.join(KEY_LOC, OAI_CONFIG_LIST),
            "builder_model": "gpt-4-1106-preview",
            "agent_model": "gpt-4-1106-preview",
        },
        "autobuild_build_config": {
            "default_llm_config": {"temperature": 1, "top_p": 0.95, "max_tokens": 1500, "seed": 52},
            "code_execution_config": {"timeout": 300, "work_dir": "groupchat", "last_n_messages": 1},
            "coding": True,
        },
        "group_chat_config": {"max_round": 10},
        "group_chat_llm_config": general_llm_config.copy(),
    }
    captain_agent = CaptainAgent(name="captain_agent", llm_config=general_llm_config, nested_mode="autobuild")
    captain_user_proxy = CaptainUserProxyAgent(
        name="captain_user_proxy",
        nested_mode_config=nested_mode_config,
        code_execution_config={"use_docker": False},
        agent_config_save_path=None,
    )
    task = (
        "Find a paper on arxiv by programming, and analyze its application in some domain. "
        "For example, find a recent paper about gpt-4 on arxiv "
        "and find its potential applications in software."
    )

    result = captain_user_proxy.initiate_chat(captain_agent, message=task, max_turns=4)
    print(result)


@pytest.mark.skipif(
    skip_openai or skip,
    reason=reason,
)
def test_captain_agent_with_library():

    config_list = config_list_from_json(
        OAI_CONFIG_LIST,
        file_location=KEY_LOC,
        filter_dict={
            "tags": ["gpt-4"],
        },
    )
    general_llm_config = {
        "temperature": 0,
        "config_list": config_list,
    }
    nested_mode_config = {
        "autobuild_init_config": {
            "config_file_or_env": os.path.join(KEY_LOC, OAI_CONFIG_LIST),
            "builder_model": "gpt-4-1106-preview",
            "agent_model": "gpt-4-1106-preview",
        },
        "autobuild_build_config": {
            "default_llm_config": {"temperature": 1, "top_p": 0.95, "max_tokens": 1500, "seed": 52},
            "code_execution_config": {"timeout": 300, "work_dir": "groupchat", "last_n_messages": 1},
            "coding": True,
            "library_path_or_json": "notebook/captainagent_expert_library.json",
        },
        "autobuild_tool_config": {
            "tool_root": "default",
            "retriever": "all-mpnet-base-v2",
        },
        "group_chat_config": {"max_round": 10},
        "group_chat_llm_config": general_llm_config.copy(),
    }
    captain_agent = CaptainAgent(name="captain_agent", llm_config=general_llm_config, nested_mode="autobuild")
    captain_user_proxy = CaptainUserProxyAgent(
        name="captain_user_proxy",
        nested_mode_config=nested_mode_config,
        code_execution_config={"use_docker": False},
        agent_config_save_path=None,
    )
    task = (
        "Find a paper on arxiv by programming, and analyze its application in some domain. "
        "For example, find a recent paper about gpt-4 on arxiv "
        "and find its potential applications in software."
    )

    result = captain_user_proxy.initiate_chat(captain_agent, message=task, max_turns=4)
    print(result)


def test_tool_builder():
    builder = ToolBuilder(corpus_path="autogen/agentchat/contrib/captainagent/tools/tool_description.csv")

    # test retrieve
    query = "How to find the square root of 4?"
    results = builder.retrieve(query, top_k=3)
    assert len(results) == 3

    # test bind
    assistant = AssistantAgent(
        "assistant",
        system_message="You are a helpful assistant.",
        llm_config={
            "cache_seed": 42,
            "config_list": [
                {
                    "model": "gpt-4-1106-preview",
                    "api_key": MOCK_OPEN_AI_API_KEY,
                }
            ],
        },
    )
    builder.bind(assistant, functions="foo, bar")
    assert "foo" in assistant.system_message

    # test bind user proxy
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": "coding",
            "timeout": 60,
        },
        max_consecutive_auto_reply=3,
    )
    user_proxy = builder.bind_user_proxy(user_proxy, tool_root="autogen/agentchat/contrib/captainagent/tools")


if __name__ == "__main__":
    test_captain_agent_from_scratch()
    test_captain_agent_with_library()
    test_tool_builder()
