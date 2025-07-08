from pydantic import BaseModel
from typing import List, Optional
from typing import Annotated, List, TypedDict, Literal, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


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


class AppState(TypedDict):
    messages: Annotated[list, add_messages]
    needs_portfolio: Optional[bool]
    needs_knowledge_base: Optional[bool]
    needs_python_code: Optional[bool]
    needs_web_search: Optional[bool]
    search_queries: list[SearchQuery]
    search_sufficient: Optional[bool]
    summary: Optional[str]
    python_code_context: Optional[str]
    python_code: Optional[str]
    execution_result: Optional[str]
    knowledge_base_results: Optional[str] = None
    source_str: Optional[str] = None
    search_iterations: int
    portfolio_data: Optional[str] = None


class PythonSearchNeed(BaseModel):
    needs_python_code: Optional[bool] = Field(
        False,
        description="Indicates if Python code is needed for the query. Only For Calculations or Data Analysis. If the query can be answered without code, this should be False.",
    )


class PythonCodeContext(BaseModel):
    python_code_context: Optional[str] = Field(
        None, description="Context for the Python code generation."
    )


class PythonCode(BaseModel):
    code: Optional[str] = Field(
        None, description="Generated Python code to answer the user's query."
    )
