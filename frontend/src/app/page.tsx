"use client";

import { Card } from "@/components/card";
import { Message } from "@/components/message";
import { FileAttachment } from "@/components/file-attachment";
import { Navbar } from "@/components/navbar";
import { useAuth } from "@/contexts/AuthContext";
/* eslint-disable @next/next/no-img-element */
import { useChat } from "@ai-sdk/react";
import { TextStreamChatTransport } from "ai";
import { useRef, useState, useEffect } from "react";

export default function Page() {
  const { isAuthenticated, isLoading } = useAuth();
  const [input, setInput] = useState("");
  const { messages, sendMessage, status } = useChat({
    transport: new TextStreamChatTransport({
      api: "http://127.0.0.1:8000/api/chat/stream?protocol=text",
    }),
  });

  const [files, setFiles] = useState<FileList | undefined>(undefined);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (isLoading) {
    return (
      <div className="flex flex-col min-h-screen bg-white">
        <Navbar />
        <div className="flex-1 flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-orange-500"></div>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    // This should be handled by middleware, but just in case
    return null;
  }

  return (
    <div className="flex flex-col min-h-screen bg-white">
      <Navbar />
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center min-h-full pb-24">
            <div className="max-w-2xl mx-auto p-6">
              <Card type="chat-attachments" />
            </div>
          </div>
        ) : (
          <div className="flex flex-col">
            <div className="max-w-3xl mx-auto w-full px-6 py-8 space-y-6">
              {messages.map((message) => (
                <Message
                  key={message.id}
                  role={message.role as "user" | "assistant"}
                  content={
                    message.parts?.find((p) => p.type === "text")?.text || ""
                  }
                  parts={message.parts}
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
            <div className="pb-24" /> {/* Spacer for floating input */}
          </div>
        )}
      </div>

      <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 w-full max-w-2xl px-4">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            sendMessage({ text: input, files });
            setInput("");
            setFiles(undefined);

            if (fileInputRef.current) {
              fileInputRef.current.value = "";
            }
          }}
          className="bg-white border border-gray-200 shadow-xl rounded-2xl backdrop-blur-sm"
        >
          {/* File attachments preview */}
          {files && files.length > 0 && (
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 rounded-t-2xl">
              <div className="flex flex-wrap gap-2">
                {Array.from(files).map((file, index) => (
                  <FileAttachment
                    key={`${file.name}-${index}`}
                    file={file}
                    onRemove={() => {
                      const newFiles = Array.from(files).filter(
                        (_, i) => i !== index
                      );
                      const dataTransfer = new DataTransfer();
                      newFiles.forEach((f) => dataTransfer.items.add(f));
                      setFiles(dataTransfer.files);
                    }}
                  />
                ))}
              </div>
            </div>
          )}

          <div className="p-4">
            <div className="flex items-end gap-3">
              <label
                htmlFor="file-upload"
                className="flex-shrink-0 p-2 text-gray-400 hover:text-gray-600 cursor-pointer rounded-lg hover:bg-gray-100 transition-colors"
                title="Attach files"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
                  />
                </svg>
              </label>

              <input
                id="file-upload"
                type="file"
                onChange={(event) => {
                  if (event.target.files) {
                    setFiles(event.target.files);
                  }
                }}
                multiple
                ref={fileInputRef}
                className="hidden"
                accept="image/*,text/*,.pdf,.json,.csv,.xml"
              />

              <textarea
                value={input}
                placeholder="Ask about finance, analyze data, or get insights... (Shift+Enter for new line)"
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    if (input.trim() || files?.length) {
                      const form = e.currentTarget.form;
                      if (form) {
                        const submitEvent = new Event("submit", {
                          bubbles: true,
                          cancelable: true,
                        });
                        form.dispatchEvent(submitEvent);
                      }
                    }
                  }
                }}
                className="flex-1 bg-transparent outline-none text-gray-900 placeholder-gray-400 py-2 px-0 resize-none min-h-[20px] max-h-32 overflow-y-auto"
                disabled={status !== "ready"}
                rows={1}
                style={{
                  height: "auto",
                  minHeight: "20px",
                }}
                onInput={(e) => {
                  const target = e.target as HTMLTextAreaElement;
                  target.style.height = "auto";
                  target.style.height =
                    Math.min(target.scrollHeight, 128) + "px";
                }}
              />

              <button
                type="submit"
                disabled={
                  status !== "ready" || (!input.trim() && !files?.length)
                }
                className="flex-shrink-0 bg-orange-500 text-white p-2 rounded-xl hover:bg-orange-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                {status === "streaming" ? (
                  <svg
                    className="w-5 h-5 animate-spin"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                ) : (
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                    />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
