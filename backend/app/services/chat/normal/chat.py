from typing import List
import time
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from app.settings.config import GEMINI_API_KEY
from app.utils.tools.finance.stock_tools import (
    get_stock_currency,
    get_stock_day_high,
    get_stock_day_low,
    get_stock_exchange,
    get_stock_fifty_day_average,
    get_stock_history,
    get_stock_income_statement,
    get_stock_info,
    get_stock_last_price,
    get_stock_last_volume,
    get_stock_market_cap,
    get_stock_open,
    get_stock_options_chain,
    get_stock_previous_close,
    get_stock_quote_type,
    get_stock_regular_market_previous_close,
    get_stock_shares,
    get_stock_ten_day_average_volume,
    get_stock_three_month_average_volume,
    get_stock_timezone,
    get_stock_two_hundred_day_average,
    get_stock_year_change,
    get_stock_year_high,
    get_stock_year_low,
    get_stock_point_change,
    get_stock_percentage_change,
    get_stock_price_change,
)
from app.utils.tools.web_search.google_search import google_search
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
)
from .prompt import (
    SYSTEM_INSTRUCTIONS,
    python_code_generation_prompt,
    python_code_needed_decision_prompt,
    python_code_retry_prompt,
    tool_executor_system_instructions,
    additional_tools_decision_prompt,
    file_processing_check_prompt,
    file_analysis_prompt,
    file_context_preparation_prompt,
)
from langgraph.graph import StateGraph, END, START
from app.types.chat.normal import (
    AppState,
    PythonCode,
    PythonSearchNeed,
    PythonExecutionResult,
    ExecutionStatus,
    AdditionalToolsNeed,
    FileProcessingNeed,
    FileAnalysis,
    FileProcessingResult,
)
from app.types.api import ClientMessage
from langgraph.prebuilt import ToolNode
from langchain_sandbox import PyodideSandbox


class ChatService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY
        )
        self.tools = [
            get_stock_currency,
            get_stock_day_high,
            get_stock_day_low,
            get_stock_exchange,
            get_stock_fifty_day_average,
            get_stock_last_price,
            get_stock_last_volume,
            get_stock_market_cap,
            get_stock_open,
            get_stock_previous_close,
            get_stock_quote_type,
            get_stock_regular_market_previous_close,
            get_stock_shares,
            get_stock_ten_day_average_volume,
            get_stock_three_month_average_volume,
            get_stock_timezone,
            get_stock_two_hundred_day_average,
            get_stock_year_change,
            get_stock_year_high,
            get_stock_year_low,
            get_stock_history,
            get_stock_income_statement,
            get_stock_info,
            get_stock_options_chain,
            get_stock_point_change,
            get_stock_percentage_change,
            get_stock_price_change,
            google_search,
        ]
        self.tool_executor = ToolNode(self.tools)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def check_python_code_needed(self, state: AppState):
        print(
            "----------------Checking if Python code generation is needed------------"
        )
        try:
            last_message = next(
                (
                    msg
                    for msg in reversed(state["messages"])
                    if isinstance(msg, HumanMessage)
                ),
                None,
            )
            if last_message is None:
                raise ValueError("No HumanMessage found in messages.")

            tool_data_context = "Available tools can provide: stock prices, volumes, market cap, financial ratios, historical data, income statements, options data, and web search capabilities."

            # Get file context from state
            file_context = state.get("file_context", "")

            print(
                f"-------------------File Context in Python Decision: {len(file_context)} chars-------------------"
            )

            structured_llm = self.llm.with_structured_output(PythonSearchNeed)
            formatted_prompt = python_code_needed_decision_prompt.format(
                user_query=last_message.content,
                file_context=file_context,
                tool_data_context=tool_data_context,
            )
            response: PythonSearchNeed = await structured_llm.ainvoke(
                [
                    SystemMessage(content=formatted_prompt),
                    HumanMessage(content="Does Python Code Generation needed ? "),
                ]
            )
            print("-------------------Python Code Needed Check-------------------")
            print(response.needs_python_code)
            print("-----------------------------------------------------")

            needs_python_code = getattr(response, "needs_python_code", False)

            if needs_python_code:
                return {
                    "needs_python_code": needs_python_code,
                }

            return {
                "needs_python_code": needs_python_code,
            }
        except Exception as e:
            print(f"Error in check_python_code_needed: {e}")
            return {"needs_python_code": False, "python_subgraph_state": None}

    async def generate_python_code(self, state: AppState):
        print("---------------Generating Python Code---------------")
        try:
            last_message = next(
                (
                    msg
                    for msg in reversed(state["messages"])
                    if isinstance(msg, HumanMessage)
                ),
                None,
            )
            if last_message is None:
                raise ValueError("No HumanMessage found in messages.")

            # Fetch tool data for Python code generation
            tool_data_result = await self.fetch_tool_data_for_python(state)
            tool_data = tool_data_result.get("tool_data_for_python", "")

            # Get file context from state
            file_context = state.get("file_context", "")

            print(
                f"-------------------File Context in Python Generation: {len(file_context)} chars-------------------"
            )
            print(
                f"-------------------File Context Preview: {file_context[:150]}...-------------------"
            )

            structured_llm = self.llm.with_structured_output(PythonCode)
            formatted_prompt = python_code_generation_prompt.format(
                user_query=last_message.content,
                file_context=file_context,
                tool_data=tool_data,
            )

            response = await structured_llm.ainvoke(
                [
                    SystemMessage(content=formatted_prompt),
                    HumanMessage(content="Generate Python code."),
                ]
            )

            python_code = getattr(response, "code", None)
            print("-----------------------Python Code------------------")
            print(python_code)
            print("-----------------------------------------------------")

            return {
                "python_code": python_code,
                "python_retry_count": 0,
                "previous_python_error": None,
                "tool_data_for_python": tool_data,
            }

        except Exception as e:
            print(f"Error in generate_python_code: {e}")
            return {
                "python_code": None,
                "python_retry_count": 0,
                "previous_python_error": None,
                "tool_data_for_python": "",
            }

    async def regenerate_python_code(self, state: AppState):
        print(
            "--------------- Regenerating Python Code with Error Context ------------"
        )
        try:
            last_message = next(
                (
                    msg
                    for msg in reversed(state["messages"])
                    if isinstance(msg, HumanMessage)
                ),
                None,
            )
            if last_message is None:
                raise ValueError("No HumanMessage found in messages.")

            previous_code = state.get("python_code", "")
            previous_error = state.get("previous_python_error", "")
            retry_count = state.get("python_retry_count", 0)

            # Use existing tool data or fetch new data if not available
            tool_data = state.get("tool_data_for_python", "")
            if not tool_data:
                tool_data_result = await self.fetch_tool_data_for_python(state)
                tool_data = tool_data_result.get("tool_data_for_python", "")

            # Get file context from state
            file_context = state.get("file_context", "")

            structured_llm = self.llm.with_structured_output(PythonCode)
            formatted_prompt = python_code_retry_prompt.format(
                user_query=last_message.content,
                file_context=file_context,
                previous_code=previous_code,
                previous_error=previous_error,
                tool_data=tool_data,
            )

            response = await structured_llm.ainvoke(
                [
                    SystemMessage(content=formatted_prompt),
                    HumanMessage(
                        content="Fix the Python code based on the previous error."
                    ),
                ]
            )

            python_code = getattr(response, "code", None)
            print("-----------------------Regenerated Python Code------------------")
            print(python_code)
            print("----------------------------------------------------------------")

            return {
                "python_code": python_code,
                "python_retry_count": retry_count + 1,
                "tool_data_for_python": tool_data,
            }

        except Exception as e:
            print(f"Error in regenerate_python_code: {e}")
            return {
                "python_code": None,
                "python_retry_count": state.get("python_retry_count", 0) + 1,
            }

    def _validate_python_syntax(self, code: str) -> tuple[bool, str]:
        """Validate Python code syntax and check for common string literal errors."""
        try:
            # Basic syntax check using compile
            compile(code, "<string>", "exec")

            # Additional checks for common string literal issues
            lines = code.split("\n")
            in_triple_double = False
            in_triple_single = False

            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                # Check for triple quotes (multi-line strings)
                if '"""' in stripped:
                    triple_count = stripped.count('"""')
                    if triple_count % 2 == 1:
                        in_triple_double = not in_triple_double

                if "'''" in stripped:
                    triple_count = stripped.count("'''")
                    if triple_count % 2 == 1:
                        in_triple_single = not in_triple_single

                # Skip quote checking if we're inside a triple-quoted string
                if in_triple_double or in_triple_single:
                    continue

                # Check for unmatched single and double quotes (excluding triple quotes)
                line_without_triple = stripped.replace('"""', "").replace("'''", "")

                single_quotes = line_without_triple.count(
                    "'"
                ) - line_without_triple.count("\\'")
                double_quotes = line_without_triple.count(
                    '"'
                ) - line_without_triple.count('\\"')

                if single_quotes % 2 != 0:
                    return False, f"Line {i}: Unterminated single quote in: {line}"
                if double_quotes % 2 != 0:
                    return False, f"Line {i}: Unterminated double quote in: {line}"

            return True, ""

        except SyntaxError as e:
            return False, f"Syntax Error: {str(e)}"
        except Exception as e:
            return False, f"Validation Error: {str(e)}"

    async def execute_python_code(self, state: AppState):
        try:
            python_code = state.get("python_code")
            if not python_code:
                print("LOG: No code provided for execution.")
                return {
                    "execution_result": PythonExecutionResult(
                        status=ExecutionStatus.ERROR, error="No code provided"
                    )
                }

            # Validate syntax before execution
            is_valid, validation_error = self._validate_python_syntax(python_code)
            if not is_valid:
                print(f"LOG: Python code validation failed: {validation_error}")
                return {
                    "execution_result": PythonExecutionResult(
                        status=ExecutionStatus.ERROR,
                        error=f"Syntax validation failed: {validation_error}",
                    ),
                    "previous_python_error": validation_error,
                }

            start_time = time.time()

            sandbox = PyodideSandbox(
                allow_net=True,
                allow_env=True,
                allow_run=True,
                allow_write=True,
                allow_read=True,
                allow_ffi=True,
            )
            result = await sandbox.execute(python_code)
            execution_time = time.time() - start_time

            print("---------------PYTHON CODE EXECUTION RESULT---------------")
            print(result)
            print("----------------------------------------------------------")

            if hasattr(result, "status") and result.status == "error":
                error_message = (
                    result.stderr if hasattr(result, "stderr") else str(result)
                )
                return {
                    "execution_result": PythonExecutionResult(
                        status=ExecutionStatus.ERROR,
                        error=error_message,
                        execution_time=execution_time,
                    ),
                    "previous_python_error": error_message,
                }
            else:
                result_output = (
                    result.stdout if hasattr(result, "stdout") else str(result)
                )
                return {
                    "execution_result": PythonExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        result=result_output,
                        execution_time=execution_time,
                    )
                }
        except Exception as e:
            execution_time = (
                time.time() - start_time if "start_time" in locals() else None
            )
            print(f"LOG: Error during Python code execution: {e}")
            return {
                "execution_result": PythonExecutionResult(
                    status=ExecutionStatus.ERROR,
                    error=str(e),
                    execution_time=execution_time,
                ),
                "previous_python_error": str(e),
            }

    def should_retry_python_code(self, state: AppState) -> str:
        execution_result = state.get("execution_result")
        retry_count = state.get("python_retry_count", 0)
        max_retries = state.get("max_python_retries", 4)

        print(
            f"LOG: Checking retry condition - Status: {execution_result.status if execution_result else 'None'}, Retry count: {retry_count}, Max retries: {max_retries}"
        )

        if (
            execution_result
            and execution_result.status == ExecutionStatus.ERROR
            and retry_count < max_retries
        ):
            print("LOG: Retrying Python code generation")
            return "regenerate_python_code"

        print("LOG: Proceeding to tools")
        return "call_tools"

    async def call_tools(self, state: AppState):
        print("----------------------Checking Tool Calls---------------------")
        messages = state["messages"]
        last_message = next(
            (msg for msg in reversed(messages) if isinstance(msg, HumanMessage)),
            None,
        )

        formatted_system_prompt = tool_executor_system_instructions.format(
            user_query=last_message.content
        )
        response = await self.llm_with_tools.ainvoke(
            messages
            + [
                SystemMessage(content=formatted_system_prompt),
                HumanMessage(content="Generate response with tools"),
            ],
        )
        response.content = ""
        print("---------------------Tool Calls Response---------------------")
        print(response)
        print("-----------------------------------------------------")
        return {"messages": messages + [response]}

    def should_call_tools(self, state: AppState) -> str:
        print("LOG: Checking if the last message contains tool calls...")
        last_message = state["messages"][-1]

        if (
            isinstance(last_message, AIMessage)
            and hasattr(last_message, "tool_calls")
            and len(last_message.tool_calls) > 0
        ):
            print("LOG: Routing to tool node.")
            return "tool_node"

        print("LOG: No tool calls found, routing to final model.")
        return "call_model"

    async def call_model(self, state: AppState):
        print("LOG: Calling the model with current state...")
        last_message = next(
            (
                msg
                for msg in reversed(state["messages"])
                if isinstance(msg, HumanMessage)
            ),
            None,
        )

        tools_response = next(
            (
                msg
                for msg in reversed(state["messages"])
                if isinstance(msg, ToolMessage)
            ),
            None,
        )

        python_code = state.get("python_code", "")
        python_execution_result = state.get("execution_result", "")
        tools_response_content = tools_response.content if tools_response else ""
        file_context = state.get("file_context", "")

        formatted_system_prompt = SYSTEM_INSTRUCTIONS.format(
            user_query=last_message.content,
            file_context=file_context,
            python_code=python_code,
            python_execution_result=python_execution_result,
            tools_response=tools_response_content,
        )

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=formatted_system_prompt),
                HumanMessage(content=last_message.content),
            ]
        )

        return {"messages": [response]}

    async def final_response(self, state: AppState):
        """Generate the final streaming response after all tool calls are complete."""
        print("LOG: Generating final streaming response...")
        last_message = next(
            (
                msg
                for msg in reversed(state["messages"])
                if isinstance(msg, HumanMessage)
            ),
            None,
        )

        tools_response = next(
            (
                msg
                for msg in reversed(state["messages"])
                if isinstance(msg, ToolMessage)
            ),
            None,
        )

        python_code = state.get("python_code", "")
        python_execution_result = state.get("execution_result", "")
        tools_response_content = tools_response.content if tools_response else ""
        file_context = state.get("file_context", "")

        formatted_system_prompt = SYSTEM_INSTRUCTIONS.format(
            user_query=last_message.content,
            file_context=file_context,
            python_code=python_code,
            python_execution_result=python_execution_result,
            tools_response=tools_response_content,
        )

        response_stream = self.llm.astream(
            [
                SystemMessage(content=formatted_system_prompt),
                HumanMessage(content=last_message.content),
            ]
        )
        async for response in response_stream:
            yield {"messages": [response]}

    async def check_additional_tools_needed(self, state: AppState):
        """Check if additional tool calls are needed based on current tool responses and AI response."""
        try:
            print(
                "---------------Checking if Additional Tools are Needed---------------"
            )

            last_message = next(
                (
                    msg
                    for msg in reversed(state["messages"])
                    if isinstance(msg, HumanMessage)
                ),
                None,
            )

            if last_message is None:
                return {"needs_additional_tools": False}

            # Get all tool responses
            tool_responses = [
                msg for msg in state["messages"] if isinstance(msg, ToolMessage)
            ]

            # Get the most recent AI response
            current_ai_response = next(
                (
                    msg
                    for msg in reversed(state["messages"])
                    if isinstance(msg, AIMessage)
                ),
                None,
            )

            tool_responses_content = (
                "\n".join(
                    [
                        f"Tool: {msg.name if hasattr(msg, 'name') else 'Unknown'}\nResponse: {msg.content}"
                        for msg in tool_responses[
                            -3:
                        ]  # Only check last 3 tool responses to avoid token limits
                    ]
                )
                if tool_responses
                else "No tool responses available"
            )

            current_response_content = (
                current_ai_response.content
                if current_ai_response
                else "No AI response available"
            )

            structured_llm = self.llm.with_structured_output(AdditionalToolsNeed)
            formatted_prompt = additional_tools_decision_prompt.format(
                user_query=last_message.content,
                tool_responses=tool_responses_content,
                current_response=current_response_content,
            )

            response: AdditionalToolsNeed = await structured_llm.ainvoke(
                [
                    SystemMessage(content=formatted_prompt),
                    HumanMessage(content="Do we need additional tool calls?"),
                ]
            )

            needs_additional_tools = getattr(response, "needs_additional_tools", False)
            reasoning = getattr(response, "reasoning", "No reasoning provided")

            print("-------------------Additional Tools Needed--------------------")
            print(needs_additional_tools)
            print("-----------------------------------------------------")
            print("----------Additional Tools Reasoning----------")
            print(reasoning)
            print("-----------------------------------------------------")

            return {
                "needs_additional_tools": needs_additional_tools,
            }

        except Exception as e:
            print(f"Error in check_additional_tools_needed: {e}")
            return {"needs_additional_tools": False}

    def should_retry_tools(self, state: AppState) -> str:
        """Determine if we should call tools again or finish."""
        needs_additional_tools = state.get("needs_additional_tools", False)
        tool_retry_count = state.get("tool_retry_count", 0)
        max_tool_retries = state.get("max_tool_retries", 3)

        print(
            f"LOG: Tool retry check - Needs additional: {needs_additional_tools}, "
            f"Retry count: {tool_retry_count}, Max retries: {max_tool_retries}"
        )

        if needs_additional_tools and tool_retry_count < max_tool_retries:
            print("LOG: Calling tools again")
            return "call_tools_retry"

        print("LOG: Finishing response")
        return "finish"

    async def call_tools_retry(self, state: AppState):
        print("----------------Calling Tools Retry----------------")

        messages = state["messages"]
        last_message = next(
            (msg for msg in reversed(messages) if isinstance(msg, HumanMessage)),
            None,
        )

        # Enhanced system prompt for retry calls
        retry_system_prompt = f"""
        {tool_executor_system_instructions.format(user_query=last_message.content)}
        
        IMPORTANT: Previous tool calls have been made but additional information is needed. 
        Consider:
        1. Searching for more recent or comprehensive information
        2. Using different search terms or approaches
        3. Looking for additional context that might be missing
        4. Exploring related topics that could enhance the answer
        """

        response = await self.llm_with_tools.ainvoke(
            messages
            + [
                SystemMessage(content=retry_system_prompt),
                HumanMessage(
                    content="Generate additional tool calls for more comprehensive information"
                ),
            ],
        )

        response.content = ""
        print("LOG: Additional tool calls generated")

        # Increment retry count
        current_retry_count = state.get("tool_retry_count", 0)

        return {
            "messages": messages + [response],
            "tool_retry_count": current_retry_count + 1,
        }

    def build_graph(self, checkpointer):
        builder = StateGraph(AppState)

        # Add all nodes
        builder.add_node("call_model", self.call_model)
        builder.add_node(
            "check_file_processing_needed", self.check_file_processing_needed
        )
        builder.add_node("process_files", self.process_files)
        builder.add_node("check_python_code_needed", self.check_python_code_needed)
        builder.add_node("generate_python_code", self.generate_python_code)
        builder.add_node("regenerate_python_code", self.regenerate_python_code)
        builder.add_node("execute_python_code", self.execute_python_code)
        builder.add_node("call_tools", self.call_tools)
        builder.add_node("call_tools_retry", self.call_tools_retry)
        builder.add_node("tool_node", self.tool_executor)
        builder.add_node(
            "check_additional_tools_needed", self.check_additional_tools_needed
        )
        builder.add_node("final_response", self.final_response)

        # Start with file processing check
        builder.add_edge(START, "check_file_processing_needed")

        # File processing flow
        builder.add_conditional_edges(
            "check_file_processing_needed",
            self.should_process_files,
            {
                "process_files": "process_files",
                "check_python_code_needed": "check_python_code_needed",
            },
        )

        # After file processing, go to python check
        builder.add_edge("process_files", "check_python_code_needed")

        def after_python_check(state):
            return (
                "generate_python_code"
                if state.get("needs_python_code", False)
                else "call_tools"
            )

        # Python code flow
        builder.add_conditional_edges("check_python_code_needed", after_python_check)
        builder.add_edge("generate_python_code", "execute_python_code")
        builder.add_edge("regenerate_python_code", "execute_python_code")
        builder.add_conditional_edges(
            "execute_python_code",
            self.should_retry_python_code,
            {
                "regenerate_python_code": "regenerate_python_code",
                "call_tools": "call_tools",
            },
        )

        # Tools flow
        builder.add_conditional_edges(
            "call_tools",
            self.should_call_tools,
            {
                "tool_node": "tool_node",
                "call_model": "call_model",
            },
        )

        # Tool retry flow
        builder.add_conditional_edges(
            "call_tools_retry",
            self.should_call_tools,
            {
                "tool_node": "tool_node",
                "call_model": "call_model",
            },
        )

        # After tool execution, go to model
        builder.add_edge("tool_node", "call_model")

        # After model response, check if additional tools are needed
        builder.add_edge("call_model", "check_additional_tools_needed")

        # Conditional routing based on additional tools check
        builder.add_conditional_edges(
            "check_additional_tools_needed",
            self.should_retry_tools,
            {
                "call_tools_retry": "call_tools_retry",
                "finish": "final_response",
            },
        )

        # Final response ends the graph
        builder.add_edge("final_response", END)

        self.graph = builder.compile(checkpointer=checkpointer)

    def _convert_message_to_langchain_format(self, messages: List[ClientMessage]):
        langchain_messages = []
        for message in messages:
            parts = message.parts
            role = message.role

            if role == "user":
                content = []
                for part in parts:
                    if part.type == "text" and part.text:
                        content.append({"type": "text", "text": part.text})
                    elif part.type == "file" and part.url:
                        if part.mediaType and part.mediaType.startswith("image/"):
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": part.url},
                                }
                            )
                        elif part.mediaType == "application/pdf":
                            base64_data = part.url.split(",")[1]
                            content.append(
                                {
                                    "type": "file",
                                    "source_type": "base64",
                                    "mime_type": "application/pdf",
                                    "data": base64_data,
                                }
                            )
                        elif part.mediaType == "text/csv":
                            base64_data = part.url.split(",")[1]
                            content.append(
                                {
                                    "type": "file",
                                    "source_type": "base64",
                                    "mime_type": "text/csv",
                                    "data": base64_data,
                                }
                            )
                        elif part.mediaType == "text/plain":
                            base64_data = part.url.split(",")[1]
                            content.append(
                                {
                                    "type": "file",
                                    "source_type": "base64",
                                    "mime_type": "text/plain",
                                    "data": base64_data,
                                }
                            )

                if content:
                    if len(content) == 1 and content[0]["type"] == "text":
                        langchain_messages.append(
                            HumanMessage(content=content[0]["text"])
                        )
                    else:
                        langchain_messages.append(HumanMessage(content=content))

            elif role == "assistant":
                text_parts = []
                for part in parts:
                    if part.type == "text" and part.text:
                        text_parts.append(part.text)

                if text_parts:
                    langchain_messages.append(AIMessage(content=" ".join(text_parts)))

        return langchain_messages

    async def stream_response(
        self, messages: List[ClientMessage], protocol: str = "data"
    ):
        print(
            "--------------------------------------------STARTING STREAM--------------------------------------"
        )
        if protocol == "text":
            langchain_messages = self._convert_message_to_langchain_format(messages)
            self.build_graph(checkpointer=None)
            state = AppState(
                messages=langchain_messages,
                has_files=False,
                file_processing_result=None,
                file_context="",
                needs_python_code=False,
                python_code=None,
                execution_result=None,
                python_retry_count=0,
                max_python_retries=4,
                previous_python_error=None,
                tool_retry_count=0,
                max_tool_retries=3,
                needs_additional_tools=False,
                tool_data_for_python="",
            )
            last_yielded_content = None
            async for chunk in self.graph.astream_events(state):
                if chunk.get("event") == "on_chain_stream":
                    if (
                        chunk["data"].get("chunk")
                        and "messages" in chunk["data"]["chunk"]
                    ):
                        if len(chunk["data"]["chunk"]["messages"]) > 0:
                            message_chunk = chunk["data"]["chunk"]["messages"][0]
                            if isinstance(message_chunk, AIMessageChunk):
                                content = message_chunk.content
                                if content != last_yielded_content:
                                    yield content
                                    last_yielded_content = content

    async def fetch_tool_data_for_python(self, state: AppState):
        print("-----------------Fetching Tool Data for Python-----------------")
        try:
            last_message = next(
                (
                    msg
                    for msg in reversed(state["messages"])
                    if isinstance(msg, HumanMessage)
                ),
                None,
            )
            if last_message is None:
                return {"tool_data_for_python": ""}

            # Create a system prompt to decide which tools to call for data gathering
            data_gathering_prompt = f"""
            <User Query>
            {last_message.content}
            </User Query>
            
            <Instructions>
            You are a data gathering assistant for Python-based financial analysis. Based on the user query, determine which financial tools should be called to gather numerical data that will be useful for calculations, analysis, or modeling.
            
            Focus on gathering:
            1. Stock prices (current, high, low, open, close)
            2. Volume data
            3. Financial ratios and metrics
            4. Historical data for time series analysis
            5. Market capitalization and valuation metrics
            6. Any other numerical data relevant to the analysis
            
            Consider the type of analysis mentioned:
            - Comparison analysis: Get data for multiple stocks
            - Trend analysis: Get historical data
            - Valuation analysis: Get financial metrics and ratios
            - Risk analysis: Get volatility and price data
            
            If specific stocks are mentioned, use the correct format (e.g., "RELIANCE.NS" for Indian stocks).
            Call multiple tools to gather comprehensive data for the analysis.
            
            Be strategic about which tools to call - gather the data that would be most useful for Python-based calculations and analysis.
            </Instructions>
            """

            # Call tools to gather data
            response = await self.llm_with_tools.ainvoke(
                [
                    SystemMessage(content=data_gathering_prompt),
                    HumanMessage(
                        content="Gather the necessary financial data for Python analysis"
                    ),
                ]
            )

            # If tools were called, execute them
            if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
                print(
                    f"LOG: Executing {len(response.tool_calls)} tools for data gathering..."
                )
                temp_messages = [response]
                tool_results = await self.tool_executor.ainvoke(
                    {"messages": temp_messages}
                )

                # Extract tool data
                tool_data = []
                for msg in tool_results["messages"]:
                    if isinstance(msg, ToolMessage):
                        tool_data.append(
                            {
                                "tool_name": msg.name
                                if hasattr(msg, "name")
                                else "unknown",
                                "data": msg.content,
                            }
                        )

                print(f"LOG: Successfully gathered data from {len(tool_data)} tools")

                # Format tool data for Python code
                formatted_tool_data = self._format_tool_data_for_python(tool_data)
                return {"tool_data_for_python": formatted_tool_data}

            print("LOG: No tools called for data gathering")
            return {"tool_data_for_python": ""}

        except Exception as e:
            print(f"Error in fetch_tool_data_for_python: {e}")
            return {"tool_data_for_python": ""}

    def _format_tool_data_for_python(self, tool_data):
        """Format tool data in a way that can be easily used in Python code."""
        if not tool_data:
            return ""

        formatted_data = "# Available Tool Data for Analysis:\n"
        data_variables = []

        for i, data_item in enumerate(tool_data):
            tool_name = data_item.get("tool_name", f"tool_{i}")
            data_content = data_item.get("data", "")

            # Clean tool name for variable naming
            clean_tool_name = (
                tool_name.replace("get_stock_", "").replace("_", "").lower()
            )

            # Try to extract numerical data and create Python variables
            if self._is_numerical_data(data_content):
                try:
                    # Extract numbers from the data
                    import re

                    numbers = re.findall(r"-?\d+\.?\d*", str(data_content))
                    if numbers:
                        # Take the first number found (usually the main value)
                        value = numbers[0]
                        # Try to convert to appropriate type
                        if "." in value:
                            python_var = f"{clean_tool_name} = {float(value)}"
                        else:
                            python_var = f"{clean_tool_name} = {int(value)}"

                        data_variables.append(python_var)
                        formatted_data += f"# {tool_name}: {data_content}\n"
                        formatted_data += f"{python_var}\n"
                    else:
                        formatted_data += f"# {tool_name}: {data_content}\n"
                except Exception:
                    formatted_data += f"# {tool_name}: {data_content}\n"
            else:
                # For non-numerical data, add as comments with truncation
                content_preview = (
                    data_content[:200] + "..."
                    if len(data_content) > 200
                    else data_content
                )
                formatted_data += f"# {tool_name}: {content_preview}\n"

        # Add a summary of available variables
        if data_variables:
            formatted_data += "\n# Available variables for analysis:\n"
            for var in data_variables:
                formatted_data += f"# {var}\n"
            formatted_data += "\n"

        return formatted_data

    def _is_numerical_data(self, data):
        """Check if data contains numerical values that could be useful for Python analysis."""
        try:
            # Simple check for numbers in the data
            import re

            numbers = re.findall(r"\d+\.?\d*", str(data))
            return len(numbers) > 0
        except Exception:
            return False

    async def check_file_processing_needed(self, state: AppState):
        """Check if the user message contains files that need processing."""
        print("----------------Checking if File Processing is Needed----------------")
        try:
            last_message = next(
                (
                    msg
                    for msg in reversed(state["messages"])
                    if isinstance(msg, HumanMessage)
                ),
                None,
            )
            if last_message is None:
                return {"has_files": False}

            # Check if the message content contains files
            has_files = False
            if isinstance(last_message.content, list):
                for content_item in last_message.content:
                    if (
                        isinstance(content_item, dict)
                        and content_item.get("type") == "file"
                    ):
                        has_files = True
                        break

            # You could also use LLM to make this decision more sophisticated
            if not has_files:
                # Also check using LLM for more complex cases
                user_message_text = ""
                if isinstance(last_message.content, list):
                    for content_item in last_message.content:
                        if (
                            isinstance(content_item, dict)
                            and content_item.get("type") == "text"
                        ):
                            user_message_text = content_item.get("text", "")
                            break
                elif isinstance(last_message.content, str):
                    user_message_text = last_message.content

                structured_llm = self.llm.with_structured_output(FileProcessingNeed)
                formatted_prompt = file_processing_check_prompt.format(
                    user_message=user_message_text
                )

                response = await structured_llm.ainvoke(
                    [
                        SystemMessage(content=formatted_prompt),
                        HumanMessage(content="Check if files need processing"),
                    ]
                )

                has_files = getattr(response, "has_files", False)

            print(
                f"-------------------File Processing Needed: {has_files}-------------------"
            )
            return {"has_files": has_files}

        except Exception as e:
            print(f"Error in check_file_processing_needed: {e}")
            return {"has_files": False}

    async def process_files(self, state: AppState):
        """Process uploaded files and extract content and insights."""
        print("----------------Processing Files----------------")
        try:
            last_message = next(
                (
                    msg
                    for msg in reversed(state["messages"])
                    if isinstance(msg, HumanMessage)
                ),
                None,
            )
            if last_message is None:
                return {
                    "file_processing_result": FileProcessingResult(
                        status=ExecutionStatus.ERROR, error="No message found"
                    )
                }

            analyses = []

            if isinstance(last_message.content, list):
                for content_item in last_message.content:
                    if (
                        isinstance(content_item, dict)
                        and content_item.get("type") == "file"
                    ):
                        try:
                            analysis = await self._analyze_single_file(content_item)
                            if analysis:
                                analyses.append(analysis)
                        except Exception as e:
                            print(f"Error processing file: {e}")
                            continue

            if not analyses:
                return {
                    "file_processing_result": FileProcessingResult(
                        status=ExecutionStatus.SUCCESS, analyses=[]
                    )
                }

            # Create context for the user query
            user_query = ""
            if isinstance(last_message.content, list):
                for content_item in last_message.content:
                    if (
                        isinstance(content_item, dict)
                        and content_item.get("type") == "text"
                    ):
                        user_query = content_item.get("text", "")
                        break
            elif isinstance(last_message.content, str):
                user_query = last_message.content

            context = await self._prepare_file_context(user_query, analyses)

            result = FileProcessingResult(
                status=ExecutionStatus.SUCCESS,
                analyses=analyses,
                context_for_query=context,
            )

            print(
                f"-------------------File Processing Complete: {len(analyses)} files processed-------------------"
            )
            print(
                f"-------------------Generated File Context Length: {len(context)}-------------------"
            )
            print(
                f"-------------------File Context Preview: {context[:200]}...-------------------"
            )
            return {"file_processing_result": result, "file_context": context}

        except Exception as e:
            print(f"Error in process_files: {e}")
            return {
                "file_processing_result": FileProcessingResult(
                    status=ExecutionStatus.ERROR, error=str(e)
                )
            }

    async def _analyze_single_file(self, file_content):
        """Analyze a single file and extract insights."""
        try:
            mime_type = file_content.get("mime_type", "")
            data = file_content.get("data", "")

            # Determine file type
            file_type = "unknown"
            if mime_type == "application/pdf":
                file_type = "pdf"
            elif mime_type.startswith("image/"):
                file_type = "image"
            elif mime_type == "text/csv":
                file_type = "csv"
            elif mime_type == "text/plain":
                file_type = "text"

            # For now, we'll pass the base64 data directly to the LLM
            # In a production environment, you might want to decode and process files differently
            structured_llm = self.llm.with_structured_output(FileAnalysis)
            formatted_prompt = file_analysis_prompt.format(
                file_type=file_type,
                media_type=mime_type,
                filename="uploaded_file",
                file_content=f"[Base64 encoded {file_type} file data]",
            )

            # For files that can be processed directly by the LLM (like images and PDFs),
            # we include them in the content
            if file_type == "image":
                response = await structured_llm.ainvoke(
                    [
                        SystemMessage(content=formatted_prompt),
                        HumanMessage(
                            content=[
                                {"type": "text", "text": "Analyze this file"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{data}"
                                    },
                                },
                            ]
                        ),
                    ]
                )
            elif file_type == "pdf":
                # For PDFs, pass the base64 data directly to the LLM as it can process PDFs
                response = await structured_llm.ainvoke(
                    [
                        SystemMessage(content=formatted_prompt),
                        HumanMessage(
                            content=[
                                {
                                    "type": "text",
                                    "text": "Analyze this PDF file and extract all financial data, numbers, and text content",
                                },
                                {
                                    "type": "file",
                                    "source_type": "base64",
                                    "mime_type": mime_type,
                                    "data": data,
                                },
                            ]
                        ),
                    ]
                )
            else:
                # For other file types, we need to decode and process them
                if file_type == "text" or file_type == "csv":
                    try:
                        decoded_content = base64.b64decode(data).decode("utf-8")
                        formatted_prompt = file_analysis_prompt.format(
                            file_type=file_type,
                            media_type=mime_type,
                            filename="uploaded_file",
                            file_content=decoded_content,
                        )
                    except Exception as e:
                        print(f"Error decoding file: {e}")
                        formatted_prompt = file_analysis_prompt.format(
                            file_type=file_type,
                            media_type=mime_type,
                            filename="uploaded_file",
                            file_content="[Unable to decode file content]",
                        )

                response = await structured_llm.ainvoke(
                    [
                        SystemMessage(content=formatted_prompt),
                        HumanMessage(content="Analyze this file content"),
                    ]
                )

            return response

        except Exception as e:
            print(f"Error in _analyze_single_file: {e}")
            return None

    async def _prepare_file_context(self, user_query, analyses):
        """Prepare formatted context from file analyses."""
        try:
            if not analyses:
                return ""

            # Format analyses for context preparation
            formatted_analyses = []
            for i, analysis in enumerate(analyses):
                formatted_analysis = f"""
File {i + 1} ({analysis.file_type}):
- Summary: {analysis.content_summary}
- Key Data: {analysis.key_data or "None"}
- Insights: {analysis.insights or "None"}
- Extracted Text: {analysis.extracted_text or "None"}
"""
                formatted_analyses.append(formatted_analysis)

            analyses_text = "\n".join(formatted_analyses)

            # Use LLM to create contextual summary
            formatted_prompt = file_context_preparation_prompt.format(
                user_query=user_query, file_analyses=analyses_text
            )

            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=formatted_prompt),
                    HumanMessage(content="Prepare context for the user query"),
                ]
            )

            return response.content

        except Exception as e:
            print(f"Error in _prepare_file_context: {e}")
            return ""

    def should_process_files(self, state: AppState) -> str:
        """Determine if files should be processed or skip to python check."""
        has_files = state.get("has_files", False)

        if has_files:
            print("LOG: Files detected, processing files")
            return "process_files"

        print("LOG: No files detected, checking python code needs")
        return "check_python_code_needed"
