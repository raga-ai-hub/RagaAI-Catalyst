export interface ChatMessage {
    id: string;
    content: string;
    sender: 'user' | 'bot';
    timestamp: string;
  }
  
  export interface ChatState {
    messages: ChatMessage[];
    isOpen: boolean;
    hasNewMessages: boolean;
    isLoading: boolean;
    error: string | null;
  }
  
  export interface ChatResponse {
    message: string;
    success: boolean;
  }