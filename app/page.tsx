// app/chat/page.tsx
'use client';

import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Send, Bot, HelpCircle, Home } from 'lucide-react';
import { useState, ChangeEvent, KeyboardEvent } from 'react';
import { useRouter } from 'next/navigation';

interface Message {
  type: 'bot' | 'user';
  content: string;
}

const initialMessages: Message[] = [
  {
    type: 'bot',
    content:
      "Hello! I'll help you input your company's financial data. Let's start with your revenue. What was your total revenue for the last fiscal year?",
  },
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState('');
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const router = useRouter();

  const handleSend = async () => {
    if (!input.trim()) return;

    // Update chat with user message
    setMessages([...messages, { type: 'user', content: input }]);
    setInput('');

    // Prepare data to send
    const data = { message: input };

    try {
      const response = await fetch('/api/process-input', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error('Failed to process input');
      }

      const result = await response.json();
      console.log("Result from Flask:", result);

      // Optionally handle the result from Flask and Supabase
      setMessages((prev) => [
        ...prev,
        {
          type: 'bot',
          content: result.message.response,
        },
      ]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        {
          type: 'bot',
          content: 'An error occurred while processing your input. Please try again.',
        },
      ]);
    }

    // Simulate next bot response
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          type: 'bot',
          content:
            'Thank you.',
        },
      ]);
    }, 1000);
  };

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setCsvFile(file);

    // Prepare FormData for file upload
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/process-input', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process CSV file');
      }

      const result = await response.json();

      // Update chat with confirmation
      setMessages((prev) => [
        ...prev,
        {
          type: 'user',
          content: `Uploaded CSV file: ${file.name}`,
        },
        {
          type: 'bot',
          content: 'Your CSV file has been processed and stored successfully!',
        },
      ]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        {
          type: 'bot',
          content: 'There was an error processing your CSV file. Please try again.',
        },
      ]);
    }
  };

  const navigateToMetrics = () => {
    router.push('/additional-metrics');
  };

  const navigateToDashboard = () => {
    router.push('/dashboard');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSend();
    }
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
                    message.type === 'bot' ? 'justify-start' : 'justify-end'
                  }`}
                >
                  <div
                    className={`max-w-[80%] p-3 rounded-xl text-sm ${
                      message.type === 'bot'
                        ? 'bg-gray-100 text-gray-800'
                        : 'bg-blue-600 text-white'
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
                onKeyDown={handleKeyDown}
              />
              <Button onClick={handleSend}>
                <Send className="w-4 h-4" />
              </Button>
            </div>
            {/* CSV Upload Section */}
            <div className="mt-4">
              <label className="flex items-center justify-center w-full px-4 py-2 bg-gray-200 text-gray-700 rounded-lg shadow-md tracking-wide uppercase border border-gray-300 cursor-pointer hover:bg-gray-300">
                <span className="mr-2">Upload CSV</span>
                <input
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={handleFileUpload}
                />
              </label>
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
                  <li>Net Assets Rate</li>
                  <li>Fossil Fuel Grade</li>
                  <li>Deforestation Grade</li>
                  <li>Relative Carbon Footprint (1-100)</li>
                  <li>Civilian Firearms Grade</li>
                  <li>Military Weapons Grade</li>
                  <li>Tobacco Grade</li>
                  <li>Prison Grade</li>
                </ul>
              </div>

              <div>
                <h3 className="font-medium mb-2 text-gray-800">
                  Format Guidelines:
                </h3>
                <ul className="list-disc list-inside space-y-2 text-sm text-gray-600">
                  <li>Use numerical values and Grades between A-E</li>
                  <li>Round to the nearest whole number</li>
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
