export interface TimelineData {
  name: string;
  duration: number;
  startTime: string;
  endTime: string;
  color: string;
  type: string;
  row: string;
  details?: {
    agent?: string;
    function?: string;
    input?: string;
    output?: string;
    model?: string;
    token_usage?: number;
    cost?: number;
    error_message?: string;
    interaction_type?: string;
    content?: string;
  };
}