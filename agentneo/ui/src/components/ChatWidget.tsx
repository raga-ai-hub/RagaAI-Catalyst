import React, { useState, useEffect, useRef } from 'react';
import { MessageCircle, X, Send, Loader2, Trash2, AlertCircle, User, Bot } from 'lucide-react';
import { ChatMessage, ChatState } from '../types/type';

const ChatWidget = () => {
  const [state, setState] = useState<ChatState>({
    messages: [],
    isOpen: false,
    hasNewMessages: false,
    isLoading: false,
    error: null as string | null,
  });
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const savedMessages = localStorage.getItem('chatHistory');
    if (savedMessages) {
      const parsedMessages = JSON.parse(savedMessages);
      setState(prev => ({
        ...prev,
        messages: parsedMessages.slice(-25)
      }));
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('chatHistory', JSON.stringify(state.messages));
  }, [state.messages]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [state.messages]);

  const toggleChat = () => {
    setState(prev => ({
      ...prev,
      isOpen: !prev.isOpen,
      hasNewMessages: false,
      error: null // Clear any errors when closing/opening chat
    }));
  };

  const clearChat = () => {
    setState(prev => ({
      ...prev,
      messages: [],
      error: null // Clear any errors when clearing chat
    }));
    localStorage.removeItem('chatHistory');
  };

  const handleSend = async () => {
    if (!input.trim() || state.isLoading) return;

    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      content: input,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    setState(prev => ({
      ...prev,
      messages: [...prev.messages.slice(-24), newMessage],
      isLoading: true,
      error: null // Clear any previous errors
    }));
    setInput('');

    try {
      const response = await fetch(import.meta.env.VITE_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: input,
          history: state.messages.slice(-6)
        })
      });

      if (!response.ok) {
        throw new Error(response.status === 500 
          ? 'Server error occurred. Please try again later.'
          : 'Failed to send message. Please try again.');
      }

      const data = await response.json();

      if (data.success) {
        const botMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          content: data.message,
          sender: 'bot',
          timestamp: new Date().toISOString()
        };

        setState(prev => ({
          ...prev,
          messages: [...prev.messages, botMessage].slice(-25),
          isLoading: false,
          error: null
        }));
      } else {
        throw new Error(data.message || 'Failed to get response from server');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setState(prev => ({ 
        ...prev, 
        isLoading: false,
        error: error instanceof Error ? error.message : 'An unexpected error occurred'
      }));
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {/* Chat Button */}
      <button
        onClick={toggleChat}
        className="relative p-4 bg-blue-600 text-white rounded-full shadow-lg hover:bg-blue-700 transition-colors"
      >
        <MessageCircle size={24} />
        {state.hasNewMessages && (
          <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full border-2 border-white" />
        )}
      </button>

      {/* Chat Window */}
      {state.isOpen && (
        <div
          ref={chatContainerRef}
          className="absolute bottom-16 right-0 w-96 h-[500px] bg-white rounded-lg shadow-2xl flex flex-col"
        >
          {/* Header */}
          <div className="p-4 border-b flex justify-between items-center bg-blue-600 text-white rounded-t-lg">
            <div className="flex items-center space-x-2">
              <Bot size={20} />
              <h3 className="font-semibold">Chat Assistant</h3>
            </div>
            <div className="flex items-center space-x-2">
              <button 
                onClick={clearChat}
                className="hover:bg-blue-700 p-1.5 rounded-lg transition-colors"
                title="Clear chat"
              >
                <Trash2 size={18} />
              </button>
              <button 
                onClick={toggleChat} 
                className="hover:bg-blue-700 p-1.5 rounded-lg transition-colors"
                title="Close chat"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {/* Error Message */}
            {state.error && (
              <div className="flex items-center p-3 mb-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
                <AlertCircle className="flex-shrink-0 w-5 h-5 mr-2" />
                <span>{state.error}</span>
              </div>
            )}

            {/* No Messages State */}
            {state.messages.length === 0 && !state.error && (
              <div className="flex flex-col items-center justify-center h-full text-gray-500 space-y-2">
                <MessageCircle size={40} className="text-gray-400" />
                <p>No messages yet. Start a conversation!</p>
              </div>
            )}

            {/* Message List */}
            {state.messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.sender === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`flex items-start space-x-2 max-w-[80%] ${
                    message.sender === 'user' ? 'flex-row-reverse space-x-reverse' : 'flex-row'
                  }`}
                >
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      message.sender === 'user' 
                        ? 'bg-blue-100 text-blue-600' 
                        : 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    {message.sender === 'user' 
                      ? <User size={16} /> 
                      : <Bot size={16} />
                    }
                  </div>
                  <div className={`space-y-1 ${message.sender === 'user' ? 'items-end' : 'items-start'}`}>
                    <div
                      className={`p-3 rounded-2xl ${
                        message.sender === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      {message.content}
                    </div>
                    <div className={`text-xs text-gray-500 px-1 ${
                      message.sender === 'user' ? 'text-right' : 'text-left'
                    }`}>
                      {new Date(message.timestamp).toLocaleTimeString([], { 
                        hour: '2-digit', 
                        minute: '2-digit'
                      })}
                    </div>
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 border-t bg-gray-50">
            <div className="flex items-center space-x-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message..."
                className="flex-1 p-2.5 border rounded-xl focus:outline-none focus:border-blue-600 focus:ring-1 focus:ring-blue-600"
                disabled={state.isLoading}
              />
              <button
                onClick={handleSend}
                disabled={state.isLoading}
                className="p-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {state.isLoading ? (
                  <Loader2 className="animate-spin" size={20} />
                ) : (
                  <Send size={20} />
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatWidget;