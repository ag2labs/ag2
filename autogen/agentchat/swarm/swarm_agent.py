import json
from inspect import signature
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from autogen.agentchat import Agent, ConversableAgent
from autogen.function_utils import get_function_schema, remove_parameter_from_function_schema
from autogen.oai import OpenAIWrapper


def parse_json_object(response: str) -> dict:
    return json.loads(response)


# Parameter name for context variables
# Use the value in functions and they will be substituted with the context variables:
# e.g. def my_function(context_variables: Dict[str, Any], my_other_parameters: Any) -> Any:
__CONTEXT_VARIABLES_PARAM_NAME__ = "context_variables"


class SwarmResult:
    """
    Encapsulates the possible return values for a swarm agent function.

    arguments:
        values (str): The result values as a string. Can be many due to multiple tool calls.
        agent (SwarmAgent): The swarm agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    values: List[str] = []
    agent: Optional["SwarmAgent"] = None
    context_variables: dict = {}

    def __init__(
        self,
        values: str,  # Text response, could be the next agent name as well
        next_agent: str = None,  # The name of the next agent if known
        context_variables: dict = {},
    ) -> None:
        self.values = values
        self.next_agent = next_agent
        self.context_variables = context_variables


class SwarmAgent(ConversableAgent):
    def __init__(
        self,
        name: str,
        system_message: Optional[str] = "You are a helpful AI Assistant.",
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        functions: Union[List[Callable], Callable] = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "NEVER",
        description: Optional[str] = None,
        context_variables: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            llm_config=llm_config,
            description=description,
            **kwargs,
        )

        if isinstance(functions, list):
            self.add_functions(functions)
        elif isinstance(functions, Callable):
            self.add_single_function(functions)

        self._reply_func_list.clear()
        self.register_reply([Agent, None], SwarmAgent.generate_reply_with_tool_calls)
        self.context_variables = context_variables or {}

    def update_context_variables(self, context_variables: Dict[str, Any]) -> None:
        pass

    def generate_reply_with_tool_calls(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, SwarmResult]:

        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]
        response = self._generate_oai_reply_from_client(client, self._oai_system_message + messages, self.client_cache)

        if isinstance(response, str):
            return True, SwarmResult(
                values=[response],
                next_agent=self.name,
            )
        elif isinstance(response, dict):
            # Tool calls, inject context_variables back in to the response before executing the tools
            if "tool_calls" in response:
                for tool_call in response["tool_calls"]:
                    if tool_call["type"] == "function":
                        function_name = tool_call["function"]["name"]

                        # Check if this function exists in our function map
                        if function_name in self._function_map:
                            func = self._function_map[function_name]  # Get the original function

                            # Check if function has context_variables parameter
                            sig = signature(func)
                            needs_context = __CONTEXT_VARIABLES_PARAM_NAME__ in sig.parameters

                            if needs_context:
                                # Parse existing arguments
                                try:
                                    current_args = json.loads(tool_call["function"]["arguments"])
                                except json.JSONDecodeError:
                                    current_args = {}

                                # Inject context_variables
                                updated_args = {"context_variables": self.context_variables, **current_args}

                                # Update the tool call with new arguments
                                tool_call["function"]["arguments"] = json.dumps(updated_args)

            _, func_response = self.generate_tool_calls_reply([response])

            return_values = []
            for response in func_response["tool_responses"]:
                return_values.append(response["content"])

            return True, SwarmResult(
                values=return_values,
                next_agent=None,
            )
        else:
            raise ValueError("Invalid response type:", type(response))

    def add_single_function(self, func: Callable, description=""):
        func._name = func.__name__

        if description:
            func._description = description
        else:
            # Use function's docstring, strip whitespace, fall back to empty string
            func._description = (func.__doc__ or "").strip()

        f = get_function_schema(func, name=func._name, description=func._description)

        # Remove the context_variable parameter from the function signature stored in self.llm_config["tools"]
        # This is done to prevent the context_variable parameter from being passed to the function when it is called
        # by the LLM
        f_no_context_variable = remove_parameter_from_function_schema(f, __CONTEXT_VARIABLES_PARAM_NAME__)
        self.update_tool_signature(f_no_context_variable, is_remove=False)

        self.register_function({func._name: self._wrap_function(func)})

    def add_functions(self, func_list: List[Callable]):
        for func in func_list:
            self.add_single_function(func)
