"use client";

import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Send, Bot, HelpCircle, Home } from "lucide-react";
import { useState } from "react";
import { useRouter } from "next/navigation";

interface Message {
  type: "bot" | "user";
  content: string;
}

const initialMessages: Message[] = [
  {
    type: "bot",
    content:
      "Hello! I'll help you input your company's financial data. Let's start with your revenue. What was your total revenue for the last fiscal year?",
  },
];

export default function Page() {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState("");
  const router = useRouter();

  const handleSend = () => {
    if (!input.trim()) return;

    setMessages([...messages, { type: "user", content: input }]);
    setInput("");

    // Simulate bot response
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          type: "bot",
          content:
            "Thank you. Now, what was your total operating expenses for the same period?",
        },
      ]);
    }, 1000);
  };

  const navigateToMetrics = () => {
    router.push("/additional-metrics");
  };

  const navigateToDashboard = () => {
    router.push("/dashboard");
  };

  return (
    <div className="flex flex-col lg:flex-row h-screen bg-background">
      {/* Chat Interface */}
      <div className="flex-grow lg:max-w-[70%] p-4 lg:p-6">
        <Card className="h-full flex flex-col shadow-lg">
          <div className="p-4 border-b flex justify-between items-center">
            <h2 className="text-2xl font-semibold flex items-center gap-2">
              <Bot className="w-6 h-6" />
              Financial Data Assistant
            </h2>
            <div className="flex gap-2">
              <Button onClick={navigateToDashboard} variant="ghost">
                <Home className="w-5 h-5 mr-1" /> Dashboard
              </Button>
              <Button onClick={navigateToMetrics} variant="ghost">
                Additional Metrics
              </Button>
            </div>
          </div>

          <ScrollArea className="flex-grow p-4">
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${
                    message.type === "bot" ? "justify-start" : "justify-end"
                  }`}
                >
                  <div
                    className={`max-w-[80%] p-3 rounded-xl text-sm ${
                      message.type === "bot"
                        ? "bg-gray-100 text-gray-800"
                        : "bg-blue-600 text-white"
                    }`}
                  >
                    {message.content}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>

          <div className="p-4 border-t">
            <div className="flex gap-2">
              <Input
                placeholder="Type your response..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
              />
              <Button onClick={handleSend}>
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </Card>
      </div>

      {/* Guide Section */}
      <div className="w-full lg:w-1/3 p-4 lg:p-6 bg-gray-50">
        <Card className="h-full shadow-md">
          <div className="p-4 border-b">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <HelpCircle className="w-5 h-5" />
              Input Guide
            </h2>
          </div>
          <ScrollArea className="h-[calc(100%-4rem)] p-4">
            <div className="space-y-4">
              <div>
                <h3 className="font-medium mb-2 text-gray-800">
                  Financial Data Required:
                </h3>
                <ul className="list-disc list-inside space-y-2 text-sm text-gray-600">
                  <li>Annual Revenue</li>
                  <li>Operating Expenses</li>
                  <li>Gross Profit Margin</li>
                  <li>Net Income</li>
                  <li>Total Assets</li>
                  <li>Total Liabilities</li>
                  <li>Cash Flow from Operations</li>
                  <li>Capital Expenditure</li>
                </ul>
              </div>

              <div>
                <h3 className="font-medium mb-2 text-gray-800">
                  Format Guidelines:
                </h3>
                <ul className="list-disc list-inside space-y-2 text-sm text-gray-600">
                  <li>Use numerical values only</li>
                  <li>Round to nearest whole number</li>
                  <li>Use USD currency</li>
                  <li>Separate thousands with commas</li>
                  <li>Example: 1,234,567</li>
                </ul>
              </div>
            </div>
          </ScrollArea>
        </Card>
      </div>
    </div>
  );
}


