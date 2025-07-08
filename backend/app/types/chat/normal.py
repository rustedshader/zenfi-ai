from pydantic import BaseModel, Field
from typing import List, Optional, Annotated, TypedDict
from langgraph.graph.message import add_messages
from enum import Enum


class ClientAttachment(BaseModel):
    name: str
    contentType: str
    url: str


class ToolInvocation(BaseModel):
    toolCallId: str
    toolName: str
    args: dict
    result: dict


class SearchQuery(BaseModel):
    search_query: Optional[str] = Field(None, description="Query for web search.")


class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )


# Python Code Generation Models
class PythonSearchNeed(BaseModel):
    needs_python_code: Optional[bool] = Field(
        default=False,
    )


class PythonCode(BaseModel):
    code: Optional[str] = Field(
        None, description="Generated Python code to answer the user's query."
    )


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    NOT_EXECUTED = "not_executed"


class PythonExecutionResult(BaseModel):
    status: ExecutionStatus = Field(
        default=ExecutionStatus.NOT_EXECUTED,
        description="Status of the Python code execution.",
    )
    result: Optional[str] = Field(
        None, description="The result or output from executing the Python code."
    )
    error: Optional[str] = Field(None, description="Error message if execution failed.")
    execution_time: Optional[float] = Field(
        None, description="Time taken to execute the code in seconds."
    )


# Main Application State
class AppState(TypedDict):
    messages: Annotated[list, add_messages]
    needs_python_code: Optional[bool]
    needs_web_search: Optional[bool]
    search_queries: list[SearchQuery]
    search_sufficient: Optional[bool]
    summary: Optional[str]
    python_code: Optional[str]
    execution_result: Optional[str]
    source_str: Optional[str] = None
    search_iterations: int
    python_code: Optional[str]
    execution_result: Optional[PythonExecutionResult]
    tools_response: Optional[str]
