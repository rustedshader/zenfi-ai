from typing import List
import time
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
    tool_exectutor_system_instructions,
)
from langgraph.graph import StateGraph, END, START
from app.types.chat.normal import (
    AppState,
    PythonCode,
    PythonSearchNeed,
    PythonExecutionResult,
    ExecutionStatus,
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
        ]
        self.tool_executor = ToolNode(self.tools)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def check_python_code_needed(self, state: AppState):
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
            print("LOG: Last message content:", last_message.content)
            structured_llm = self.llm.with_structured_output(PythonSearchNeed)
            formatted_prompt = python_code_needed_decision_prompt.format(
                user_query=last_message
            )
            response = await structured_llm.ainvoke(
                [
                    SystemMessage(content=formatted_prompt),
                    HumanMessage(content="Does Python Code Generation needed ? "),
                ]
            )
            print("LOG:", response.needs_python_code)

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
        print("LOG: Generating Python Code...")
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
            structured_llm = self.llm.with_structured_output(PythonCode)
            formatted_prompt = python_code_generation_prompt.format(
                user_query=last_message.content
            )

            response = await structured_llm.ainvoke(
                [
                    SystemMessage(content=formatted_prompt),
                    HumanMessage(content="Generate Python code."),
                ]
            )

            python_code = getattr(response, "code", None)
            print("LOG: Generated Python Code:", python_code)

            return {
                "python_code": python_code,
            }

        except Exception as e:
            print(f"Error in generate_python_code: {e}")
            return {
                "python_code": None,
            }

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

            print("LOG: Python code execution result:", result)

            return {
                "execution_result": PythonExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    result=str(result),
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
                )
            }

    async def call_tools(self, state: AppState):
        """
        This node invokes the LLM with tools. If the LLM decides to call a tool,
        it will return an AIMessage with tool_calls. We append this message to
        our state.
        """
        print("LOG: Checking for potential tool calls...")
        messages = state["messages"]

        # The llm_with_tools will automatically decide if a tool should be called
        response = await self.llm_with_tools.ainvoke(messages)
        print("LOG: Tool calling LLM response:", response)

        # Append the response to the history of messages
        # This is the crucial step.
        return {"messages": messages + [response]}

    def should_call_tools(self, state: AppState) -> str:
        """
        Determines whether to call tools or end the chain.
        """
        print("LOG: Checking if the last message contains tool calls...")
        last_message = state["messages"][-1]

        # If the last message is an AIMessage and has tool_calls, route to tool_node
        if (
            isinstance(last_message, AIMessage)
            and hasattr(last_message, "tool_calls")
            and len(last_message.tool_calls) > 0
        ):
            print("LOG: Routing to tool node.")
            return "tool_node"

        # Otherwise, route to the final model call
        print("LOG: No tool calls found, routing to final model.")
        return "call_model"

    async def call_model(self, state: AppState):
        print("LOG: Calling the model with current state...")
        print("App State", state)
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
        print("Tools Response:", tools_response_content)

        formatted_system_prompt = SYSTEM_INSTRUCTIONS.format(
            user_query=last_message.content,
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

    def build_graph(self, checkpointer):
        builder = StateGraph(AppState)

        builder.add_node("call_model", self.call_model)
        builder.add_node("check_python_code_needed", self.check_python_code_needed)
        builder.add_node("generate_python_code", self.generate_python_code)
        builder.add_node("execute_python_code", self.execute_python_code)

        # This node now correctly appends the AIMessage to the state
        builder.add_node("call_tools", self.call_tools)

        # This is the prebuilt ToolNode
        builder.add_node("tool_node", self.tool_executor)

        builder.add_edge(START, "check_python_code_needed")

        def after_python_check(state):
            return (
                "generate_python_code"
                if state.get("needs_python_code", False)
                else "call_tools"
            )

        builder.add_conditional_edges("check_python_code_needed", after_python_check)
        builder.add_edge("generate_python_code", "execute_python_code")
        builder.add_edge("execute_python_code", "call_tools")

        # *** REVISED TOOL EXECUTION FLOW ***
        # After calling the tool-enabled LLM, we use our new conditional function
        builder.add_conditional_edges(
            "call_tools",
            self.should_call_tools,  # Use the new function here
            {
                "tool_node": "tool_node",
                "call_model": "call_model",
            },
        )

        # After the ToolNode executes, the ToolMessages are added to the state.
        # We can now pass this full context to the final model to generate a response.
        builder.add_edge("tool_node", "call_model")

        # The final model call ends the graph run
        builder.add_edge("call_model", END)

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
        if protocol == "text":
            langchain_messages = self._convert_message_to_langchain_format(messages)
            self.build_graph(checkpointer=None)
            state = AppState(
                messages=langchain_messages,
                needs_portfolio=False,
                needs_knowledge_base=False,
                needs_python_code=False,
                needs_web_search=False,
                search_queries=[],
                search_sufficient=False,
                summary=None,
                python_code=None,
                execution_result=None,
                knowledge_base_results=None,
                source_str=None,
                search_iterations=0,
                portfolio_data=None,
                has_tool_calls=False,
                tools_executed=False,
                tools_response=None,
            )
            async for chunk in self.graph.astream_events(state):
                if chunk.get("event") == "on_chain_stream":
                    if (
                        chunk["data"].get("chunk")
                        and "messages" in chunk["data"]["chunk"]
                    ):
                        if len(chunk["data"]["chunk"]["messages"]) > 0:
                            if isinstance(
                                chunk["data"]["chunk"]["messages"][0], AIMessageChunk
                            ):
                                print(chunk["data"]["chunk"]["messages"][0])
                                yield chunk["data"]["chunk"]["messages"][0].content
