from typing import Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel


class SingleAgentState(TypedDict):
    input: str
    output: Any
    log: list[BaseMessage]


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item if isinstance(item, str) else item.get("text", "")
            for item in content
            if isinstance(item, str) or isinstance(item, dict)
        )
    return str(content)


def _response_to_output(response: Any) -> Any:
    if isinstance(response, BaseModel):
        return response.model_dump()
    if isinstance(response, dict) and isinstance(response.get("parsed"), BaseModel):
        return response["parsed"].model_dump()
    if isinstance(response, BaseMessage):
        return _message_content_to_text(response.content)
    content = getattr(response, "content", None)
    if content is not None:
        return _message_content_to_text(content)
    return response


def build_single_agent_graph(
    llm: BaseChatModel,
    output_schema: type[BaseModel] | None = None,
) -> CompiledStateGraph:
    """Build a LangGraph baseline with one LLM node."""
    model = (
        llm.with_structured_output(output_schema.model_json_schema())
        if output_schema is not None
        else llm
    )

    def single_agent_node(state: SingleAgentState) -> SingleAgentState:
        messages = [HumanMessage(content=state["input"])]
        response = model.invoke(messages)
        output = _response_to_output(response)
        if output_schema is not None:
            output = output_schema.model_validate(output).model_dump()
        return {
            "input": state["input"],
            "output": output,
            "log": [*state.get("log", []), *messages],
        }

    graph = StateGraph(SingleAgentState)
    graph.add_node("single_agent", single_agent_node)
    graph.add_edge(START, "single_agent")
    graph.add_edge("single_agent", END)
    return graph.compile()


__all__ = ["SingleAgentState", "build_single_agent_graph"]
