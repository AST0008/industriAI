"use client";

import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Send, Bot, HelpCircle } from "lucide-react";
import { useState } from "react";
import { useRouter } from "next/navigation";

interface Message {
  type: 'bot' | 'user';
  content: string;
}

const initialMessages: Message[] = [
  {
    type: 'bot',
    content: 'Hello! I\'ll help you input your company\'s financial data. Let\'s start with your revenue. What was your total revenue for the last fiscal year?'
  }
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState('');
  const router = useRouter();

  const handleSend = () => {
    if (!input.trim()) return;

    setMessages([...messages, { type: 'user', content: input }]);
    setInput('');

    // Simulate bot response
    setTimeout(() => {
      setMessages(prev => [...prev, {
        type: 'bot',
        content: 'Thank you. Now, what was your total operating expenses for the same period?'
      }]);
    }, 1000);
  };

  const navigateToMetrics = () => {
    router.push("/additional-metrics");
  };
  const navigateToDashboard = () => {
    router.push("/dashboard");
  };

  return (
    <div className="flex h-screen bg-background">
      {/* Chat Interface */}
      <div className="flex-grow max-w-[75%] p-6">
        <Card className="h-full flex flex-col">
          <div className="p-4 border-b flex justify-between items-center">
            <h2 className="text-2xl font-semibold flex items-center gap-2">
              <Bot className="w-6 h-6" />
              Financial Data Assistant
            </h2>
            <Button onClick={navigateToDashboard} className="ml-4" variant="secondary">
              Dashboard
            </Button>
            <Button onClick={navigateToMetrics} variant="secondary">
              Additional Metrics
            </Button>
          </div>
          
          <ScrollArea className="flex-grow p-4">
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${
                    message.type === 'bot' ? 'justify-start' : 'justify-end'
                  }`}
                >
                  <div
                    className={`max-w-[80%] p-3 rounded-lg ${
                      message.type === 'bot'
                        ? 'bg-secondary text-secondary-foreground'
                        : 'bg-primary text-primary-foreground'
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
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
              />
              <Button onClick={handleSend}>
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </Card>
      </div>

      {/* Guide Section */}
      <div className="w-1/4 p-6">
        <Card className="h-full">
          <div className="p-4 border-b">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <HelpCircle className="w-5 h-5" />
              Input Guide
            </h2>
          </div>
          <ScrollArea className="h-[calc(100%-4rem)] p-4">
            <div className="space-y-4">
              <div>
                <h3 className="font-medium mb-2">Financial Data Required:</h3>
                <ul className="list-disc list-inside space-y-2 text-sm">
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
                <h3 className="font-medium mb-2">Format Guidelines:</h3>
                <ul className="list-disc list-inside space-y-2 text-sm">
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
