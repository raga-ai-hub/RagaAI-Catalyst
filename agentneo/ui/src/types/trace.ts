export interface Project {
    id: number;
    name: string;
}

export interface TraceHistoryItem {
    id: string;
    start_time: string;
    end_time: string;
    duration: number;
    total_llm_calls: number;
    total_tool_calls: number;
    total_agent_calls: number;
    total_errors: number;
}

export interface TraceComponent {
    name: string;
    type: 'agent' | 'tool' | 'llm' | 'network' | 'user';
    duration: string;
    status: 'success' | 'error' | 'warning';
    details: string;
    input?: any;
    output?: any;
    children?: TraceComponent[];
}

export interface DetailedTraceComponents {
    id: string;
    duration: string;
    errors: number;
    components: TraceComponent[];
}