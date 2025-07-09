"use client";

import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import "katex/dist/katex.min.css";

interface MessageProps {
  role: "user" | "assistant";
  content: string;
  parts?: Array<{
    type: string;
    text?: string;
    mediaType?: string;
    url?: string;
  }>;
}

export function Message({ role, content, parts }: MessageProps) {
  const renderMarkdown = (text: string) => {
    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeRaw]}
        components={{
          code({ node, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            const language = match ? match[1] : "";
            const isInline = !className;

            if (isInline) {
              return <code>{children}</code>;
            }

            return (
              <div className="my-4">
                <div className="bg-gray-100 text-gray-700 px-3 py-2 text-sm rounded-t-lg border-b border-gray-200 font-medium">
                  {language || "code"}
                </div>
                <SyntaxHighlighter
                  style={oneLight as any}
                  language={language || "text"}
                  PreTag="div"
                  className="rounded-t-none rounded-b-lg !mt-0"
                  showLineNumbers={true}
                >
                  {String(children).replace(/\n$/, "")}
                </SyntaxHighlighter>
              </div>
            );
          },
          table({ children }) {
            return <table>{children}</table>;
          },
          thead({ children }) {
            return <thead>{children}</thead>;
          },
          th({ children }) {
            return <th>{children}</th>;
          },
          td({ children }) {
            return <td>{children}</td>;
          },
          blockquote({ children }) {
            return <blockquote>{children}</blockquote>;
          },
          h1({ children }) {
            return <h1>{children}</h1>;
          },
          h2({ children }) {
            return <h2>{children}</h2>;
          },
          h3({ children }) {
            return <h3>{children}</h3>;
          },
          p({ children }) {
            return <p>{children}</p>;
          },
          ul({ children }) {
            return <ul>{children}</ul>;
          },
          ol({ children }) {
            return <ol>{children}</ol>;
          },
          li({ children }) {
            return <li>{children}</li>;
          },
          a({ href, children }) {
            return (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            );
          },
          strong({ children }) {
            return <strong>{children}</strong>;
          },
          em({ children }) {
            return <em>{children}</em>;
          },
        }}
      >
        {text}
      </ReactMarkdown>
    );
  };

  return (
    <div className="w-full max-w-none">
      <div className="flex flex-row gap-4 mb-8">
        <div className="flex-shrink-0 w-20 text-sm font-medium text-gray-600 capitalize">
          {role}:
        </div>
        <div className="flex-1 min-w-0">
          {parts && parts.length > 0 ? (
            <div className="flex flex-col gap-3">
              {parts.map((part, index) => {
                if (part.type === "text" && part.text) {
                  return (
                    <div
                      key={index}
                      className="prose prose-sm max-w-none markdown-content"
                    >
                      {renderMarkdown(part.text)}
                    </div>
                  );
                }
                if (
                  part.type === "file" &&
                  part.mediaType?.startsWith("image/")
                ) {
                  return (
                    <div key={index} className="max-w-md">
                      <img
                        className="rounded-lg shadow-md"
                        src={part.url}
                        alt="Uploaded image"
                      />
                    </div>
                  );
                }
                return null;
              })}
            </div>
          ) : (
            <div className="prose prose-sm max-w-none markdown-content">
              {renderMarkdown(content)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
