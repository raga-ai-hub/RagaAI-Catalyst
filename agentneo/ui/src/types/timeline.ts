export interface TimelineData {
  name: string;
  duration: number;
  startTime: string;
  endTime: string;
  color: string;
  type: string;
  row: string;
  isDot?: boolean;
  details?: {
    name?: string,
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
    parentName?: string;
  };
  counts?: {
    llms: number;
    tools: number;
    interactions: number;
    errors: number;
    agents: number;
  };
}