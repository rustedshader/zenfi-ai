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


class AdditionalToolsNeed(BaseModel):
    needs_additional_tools: Optional[bool] = Field(
        default=False,
        description="Whether additional tool calls are needed to provide a complete answer.",
    )
    reasoning: Optional[str] = Field(
        None,
        description="Explanation of why additional tools are needed or not needed.",
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


# File Processing Models
class FileProcessingNeed(BaseModel):
    has_files: Optional[bool] = Field(
        default=False,
        description="Whether the user message contains files that need processing.",
    )


class FileAnalysis(BaseModel):
    file_type: str = Field(description="Type of file (pdf, image, csv, text)")
    content_summary: str = Field(description="Summary of the file content")
    key_data: Optional[str] = Field(
        None, description="Key data extracted from the file"
    )
    insights: Optional[str] = Field(
        None, description="Important insights from the file"
    )
    extracted_text: Optional[str] = Field(
        None, description="Raw text extracted from the file"
    )


class FileProcessingResult(BaseModel):
    status: ExecutionStatus = Field(
        default=ExecutionStatus.NOT_EXECUTED, description="Status of file processing"
    )
    analyses: List[FileAnalysis] = Field(
        default=[], description="Analysis results for each file"
    )
    context_for_query: Optional[str] = Field(
        None, description="Formatted context to be used with user query"
    )
    error: Optional[str] = Field(None, description="Error message if processing failed")


# Main Application State
class AppState(TypedDict):
    messages: Annotated[list, add_messages]
    has_files: Optional[bool]
    file_processing_result: Optional[FileProcessingResult]
    file_context: Optional[str]
    needs_python_code: Optional[bool]
    python_code: Optional[str]
    execution_result: Optional[PythonExecutionResult]
    python_retry_count: int
    max_python_retries: int
    previous_python_error: Optional[str]
    tool_retry_count: int
    max_tool_retries: int
    needs_additional_tools: Optional[bool]
    tool_data_for_python: Optional[str]
