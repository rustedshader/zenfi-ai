import { GeistMono } from "geist/font/mono";
import Link from "next/link";
import { ReactNode } from "react";

const Code = ({ children }: { children: ReactNode }) => {
  return (
    <code
      className={`${GeistMono.className} text-xs bg-gray-100 text-red-600 px-2 py-1 rounded border`}
    >
      {children}
    </code>
  );
};

export const Card = ({
  type,
  onQuerySelect,
}: {
  type: string;
  onQuerySelect?: (query: string) => void;
}) => {
  const handleQueryClick = (query: string) => {
    if (onQuerySelect) {
      onQuerySelect(query);
    }
  };

  return type === "chat-text" ? (
    <div className="self-center w-full fixed bottom-20 px-8 py-6">
      <div className="p-4 border rounded-lg flex flex-col gap-2 w-full">
        <div className="text font-semibold text-zinc-800">
          Stream Chat Completions
        </div>
        <div className="text-zinc-500 text-sm leading-6 flex flex-col gap-4">
          <p>
            The <Code>useChat</Code> hook can be integrated with a Python
            FastAPI backend to stream chat completions in real-time. The most
            basic setup involves streaming plain text chunks by setting the{" "}
            <Code>streamProtocol</Code> to <Code>text</Code>.
          </p>

          <p>
            To make your responses streamable, you will have to use the{" "}
            <Code>StreamingResponse</Code> class provided by FastAPI.
          </p>
        </div>
      </div>
    </div>
  ) : type === "chat-data" ? (
    <div className="self-center w-full fixed bottom-20 px-8 py-6">
      <div className="p-4 border rounded-lg flex flex-col gap-2 w-full">
        <div className="text font-semibold text-zinc-800">
          Stream Chat Completions with Tools
        </div>
        <div className="text-zinc-500 text-sm leading-6 flex flex-col gap-4">
          <p>
            The <Code>useChat</Code> hook can be integrated with a Python
            FastAPI backend to stream chat completions in real-time. However,
            the most basic setup that involves streaming plain text chunks by
            setting the <Code>streamProtocol</Code> to <Code>text</Code> is
            limited.
          </p>

          <p>
            As a result, setting the streamProtocol to <Code>data</Code> allows
            you to stream chunks that include information about tool calls and
            results.
          </p>

          <p>
            To make your responses streamable, you will have to use the{" "}
            <Code>StreamingResponse</Code> class provided by FastAPI. You will
            also have to ensure that your chunks follow the{" "}
            <Link
              target="_blank"
              className="text-blue-500 hover:underline"
              href="https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol#data-stream-protocol"
            >
              data stream protocol
            </Link>{" "}
            and that the response has <Code>x-vercel-ai-data-stream</Code>{" "}
            header set to <Code>v1</Code>.
          </p>
        </div>
      </div>
    </div>
  ) : type === "chat-attachments" ? (
    <div className="w-full max-w-2xl mx-auto">
      <div className="p-6 border border-gray-200 rounded-xl bg-white shadow-sm flex flex-col gap-4">
        <div className="text-xl font-semibold text-gray-800 flex items-center gap-2">
          <span className="text-2xl">🤖</span>
          Welcome to Zenfi AI
        </div>
        <div className="text-gray-600 text-sm leading-6 flex flex-col gap-4">
          <p>
            <strong>Advanced AI Financial Assistant</strong> powered by Python
            code execution, robust RAG system, and real-time market data
            integration. Ask complex financial questions and get comprehensive
            analysis with charts, calculations, and insights.
          </p>

          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="font-semibold text-gray-800 mb-2">
              💬 Try these example queries:
            </h4>
            <div className="space-y-2 text-xs">
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() =>
                    handleQueryClick("What is stock price of Reliance?")
                  }
                  className="bg-blue-100 text-blue-800 px-2 py-1 rounded hover:bg-blue-200 transition-colors"
                >
                  What is stock price of Reliance?
                </button>
                <button
                  onClick={() =>
                    handleQueryClick(
                      "Get the latest Bitcoin price and create a trend analysis"
                    )
                  }
                  className="bg-green-100 text-green-800 px-2 py-1 rounded hover:bg-green-200 transition-colors"
                >
                  Get latest Bitcoin price and trend analysis
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() =>
                    handleQueryClick(
                      "Analyze Apple stock performance over the last 1 month"
                    )
                  }
                  className="bg-purple-100 text-purple-800 px-2 py-1 rounded hover:bg-purple-200 transition-colors"
                >
                  Analyze Apple stock performance over last 1 month
                </button>
                <button
                  onClick={() =>
                    handleQueryClick(
                      "Calculate compound interest for $1000 at 5% for 10 years"
                    )
                  }
                  className="bg-orange-100 text-orange-800 px-2 py-1 rounded hover:bg-orange-200 transition-colors"
                >
                  Calculate compound interest for $1000 at 5% for 10 years
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() =>
                    handleQueryClick(
                      "Compare unemployment rates between different countries"
                    )
                  }
                  className="bg-red-100 text-red-800 px-2 py-1 rounded hover:bg-red-200 transition-colors"
                >
                  Compare unemployment rates between countries
                </button>
                <button
                  onClick={() =>
                    handleQueryClick("What is the latest financial news?")
                  }
                  className="bg-indigo-100 text-indigo-800 px-2 py-1 rounded hover:bg-indigo-200 transition-colors"
                >
                  What's the latest financial news?
                </button>
              </div>
            </div>
          </div>

          <div className="bg-amber-50 p-4 rounded-lg border border-amber-200">
            <h4 className="font-semibold text-amber-800 mb-2">
              🚀 Advanced Analysis Examples:
            </h4>
            <ul className="text-xs text-amber-700 space-y-1">
              <li>• Generate linear regression models with custom data</li>
              <li>• Comprehensive Tesla financial analysis (FY 2024)</li>
              <li>
                • Backtest Apple momentum trading strategies for WWDC events
              </li>
              <li>• Analyze Middle East events impact on portfolios</li>
              <li>
                • Compare ordinary least squares vs maximum likelihood estimator
              </li>
            </ul>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
            <h4 className="font-semibold text-blue-800 mb-2">
              📊 Data Processing & Analysis:
            </h4>
            <p className="text-xs text-blue-700">
              Upload CSV files, financial documents, bank statements, or paste
              data directly. Analyze your personal finances, trading patterns,
              and investment performance. Example: Trading volume vs closing
              price analysis with predictive modeling.
            </p>
          </div>

          <div className="bg-green-50 p-4 rounded-lg border border-green-200">
            <h4 className="font-semibold text-green-800 mb-2">
              🏦 Bank Statement Analysis:
            </h4>
            <p className="text-xs text-green-700">
              Upload your bank statements for comprehensive financial analysis.
              Track spending patterns, categorize transactions, identify trends,
              and get insights into your financial health and budgeting
              opportunities.
            </p>
          </div>

          <p className="text-xs text-gray-500">
            Powered by advanced AI with Python code execution and real-time data
            processing. All file uploads are processed securely with
            comprehensive financial analysis capabilities.
          </p>
        </div>
      </div>
    </div>
  ) : null;
};
