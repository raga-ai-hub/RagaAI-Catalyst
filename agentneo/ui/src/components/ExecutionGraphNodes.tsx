import React, { useState } from 'react';
import { Handle, Position } from 'reactflow';

const formatTimestamp = (timestamp: string) => {
  if (!timestamp) return 'N/A';
  return new Date(timestamp).toLocaleTimeString();
};

const truncateText = (text: string, maxLength: number = 50) => {
  if (!text) return '';
  return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
};

const calculateDuration = (start_time: string, end_time: string) => {
  if (!start_time || !end_time) return 'N/A';
  const start = new Date(start_time).getTime();
  const end = new Date(end_time).getTime();
  const durationMs = end - start;
  return (durationMs / 1000).toFixed(2) + 's';
};

const formatValue = (value: any, defaultValue: string = 'N/A') => {
  if (value === null || value === undefined) return defaultValue;
  return value;
};

const formatNumber = (value: any) => {
  if (value === null || value === undefined) return 0;
  return value;
};

const formatJson = (value: any) => {
  if (!value) return 'No data';
  try {
    return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  } catch (e) {
    return String(value);
  }
};


const formatNetworkCalls = (networkCalls: any[]) => {
  if (!networkCalls || networkCalls.length === 0) return 'No network calls';
  
  return networkCalls.map((call: any, index: number) => {
    const status = call.status_code ? 
      (call.status_code >= 200 && call.status_code < 300 ? '✓' : '✗') : 
      '?';
    
    // Parse response body if it's a JSON string
    let responseBody = {};
    try {
      responseBody = typeof call.response_body === 'string' ? 
        JSON.parse(call.response_body) : call.response_body;
    } catch (e) {
      responseBody = call.response_body;
    }

    return `Call ${index + 1}:
    ${status} ${call.method?.toUpperCase() || 'Unknown'} ${call.status_code || ''}
    Duration: ${formatValue(call.duration?.toFixed(2), '0')}s
    URL: ${call.url || 'N/A'}
    Response: ${JSON.stringify(responseBody, null, 2)}
    Error: ${call.error || 'None'}`;
  }).join('\n\n');
};

const getNetworkCallsSummary = (networkCalls: any[]) => {
  if (!networkCalls || networkCalls.length === 0) return '0 calls';
  const successCalls = networkCalls.filter(call => call.status_code >= 200 && call.status_code < 300).length;
  const failedCalls = networkCalls.filter(call => call.status_code >= 400).length;
  return `${networkCalls.length} calls (${successCalls} ✓, ${failedCalls} ✗)`;
};

const getNodeIcon = (type: string) => {
  switch (type) {
    case 'trace':
      return '🔍';
    case 'agent':
      return '🤖';
    case 'llm':
      return '🧠';
    case 'tool':
      return '🛠️';
    case 'interaction':
      return '💬';
    case 'error':
      return '⚠️';
    default:
      return '📌';
  }
};

export const CustomNode = ({ data }: { data: any }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  const getCollapsedContent = () => {
    switch (data.type) {
      case 'trace':
        const counts = data.metadata?.totalCounts || {};
        
        // Get all nodes including nested ones within agent calls
        const allNodes = [
          ...(data.metadata?.agent_calls || []).flatMap(agent => [
            agent,
            ...(agent.llm_calls || []),
            ...(agent.tool_calls || [])
          ]),
          ...(data.metadata?.llm_calls || []),
          ...(data.metadata?.tool_calls || [])
        ];

        // Find earliest start time and latest end time
        const startTimes = allNodes
          .map(node => node.start_time)
          .filter(time => time)
          .map(time => new Date(time).getTime());
        
        const endTimes = allNodes
          .map(node => node.end_time)
          .filter(time => time)
          .map(time => new Date(time).getTime());

        const traceStartTime = startTimes.length ? Math.min(...startTimes) : data.startTime;
        const traceEndTime = endTimes.length ? Math.max(...endTimes) : data.endTime;
        const duration = calculateDuration(
          new Date(traceStartTime).toISOString(),
          new Date(traceEndTime).toISOString()
        );
        
        // Add direct calls to the stats
        const directLLMCalls = data.metadata?.llm_calls?.length || 0;
        const directToolCalls = data.metadata?.tool_calls?.length || 0;
        const totalLLMCalls = formatNumber(counts.llmCount || directLLMCalls);
        const totalToolCalls = formatNumber(counts.toolCount || directToolCalls);
        
        return (
          <div className="node-content">
            <div className="node-header">
              <span className="node-icon">{getNodeIcon(data.type)}</span>
              <span className="node-title">Trace: {formatValue(data.name)}</span>
            </div>
            <div className="node-body">
              <div className="stat-grid">
                <div className="stat-item">
                  <span className="stat-label">Duration</span>
                  <span className="stat-value">{duration}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Agents</span>
                  <span className="stat-value">{formatNumber(data.metadata?.agent_calls?.length)}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">LLM Calls</span>
                  <span className="stat-value">{totalLLMCalls}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Tool Calls</span>
                  <span className="stat-value">{totalToolCalls}</span>
                </div>
              </div>
              <div className="text-xs mt-3 space-y-1">
                <div className="truncate">
                  <span className="text-gray-500">Start:</span> {formatTimestamp(new Date(traceStartTime).toISOString())}
                </div>
                <div className="truncate">
                  <span className="text-gray-500">End:</span> {formatTimestamp(new Date(traceEndTime).toISOString())}
                </div>
              </div>
            </div>
          </div>
        );

        case 'agent':
          return (
            <div className="node-content">
              <div className="node-header">
                <span className="node-icon">{getNodeIcon(data.type)}</span>
                <span className="node-title">Agent: {formatValue(data.name)}</span>
              </div>
              <div className="node-body">
                <div className="stat-grid">
                  <div className="stat-item">
                    <span className="stat-label">Duration</span>
                    <span className="stat-value">{formatValue(data.duration)}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">LLM Calls</span>
                    <span className="stat-value">{formatNumber(data.metadata?.llm_calls?.length)}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Tool Calls</span>
                    <span className="stat-value">{formatNumber(data.metadata?.tool_calls?.length)}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Status</span>
                    <span className="stat-value">{data.metadata?.errors?.length ? '❌ Error' : '✅ Success'}</span>
                  </div>
                </div>
                <div className="text-xs mt-3 space-y-1">
                  <div className="truncate">
                    <span className="text-gray-500">Start:</span> {formatTimestamp(data.metadata?.start_time)}
                  </div>
                  <div className="truncate">
                    <span className="text-gray-500">End:</span> {formatTimestamp(data.metadata?.end_time)}
                  </div>
                </div>
              </div>
            </div>
          );

      case 'llm':
        return (
          <div className="node-content">
            <div className="node-header">
              <span className="node-icon">{getNodeIcon(data.type)}</span>
              <span className="node-title">LLM: {formatValue(data.metadata?.model)}</span>
            </div>
            <div className="node-body">
              <div className="stat-grid">
                <div className="stat-item">
                  <span className="stat-label">Duration</span>
                  <span className="stat-value">{formatValue(data.duration)}</span>
                </div>
                {/* <div className="stat-item">
                    <span className="stat-label">Status</span>
                    <span className="stat-value">{data.metadata?.errors?.length ? '❌ Error' : '✅ Success'}</span>
                </div> */}
              </div>
              <div className="text-xs mt-2 space-y-1">
                <div className="truncate"><span className="text-gray-500">In:</span> {truncateText(formatValue(data.metadata?.input_prompt, ''), 40)}</div>
                <div className="truncate"><span className="text-gray-500">Out:</span> {truncateText(formatValue(data.metadata?.output, ''), 40)}</div>
              </div>
            </div>
          </div>
        );

      case 'tool':
        return (
          <div className="node-content">
            <div className="node-header">
              <span className="node-icon">{getNodeIcon(data.type)}</span>
              <span className="node-title">Tool: {formatValue(data.name)}</span>
            </div>
            <div className="node-body">
              <div className="stat-grid">
                <div className="stat-item">
                  <span className="stat-label">Duration</span>
                  <span className="stat-value">{formatValue(data.duration)}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Network</span>
                  <span className="stat-value">{getNetworkCallsSummary(data.metadata?.network_calls)}</span>
                </div>
              </div>
              <div className="text-xs mt-2 space-y-1">
                <div className="truncate"><span className="text-gray-500">In:</span> {truncateText(formatJson(data.metadata?.input_parameters), 40)}</div>
                <div className="truncate"><span className="text-gray-500">Out:</span> {truncateText(formatValue(data.metadata?.output, ''), 40)}</div>
              </div>
            </div>
          </div>
        );

      case 'interaction':
        return (
          <div className="node-content">
            <div className="node-header">
              <span className="node-icon">{getNodeIcon(data.type)}</span>
              <span className="node-title">Interaction: {formatValue(data.metadata?.interaction_type)}</span>
            </div>
            <div className="node-body">
              <div className="text-xs mt-1">
                <div className="truncate">{truncateText(formatValue(data.metadata?.content, ''), 60)}</div>
              </div>
            </div>
          </div>
        );

      case 'error':
        return (
          <div className="node-content">
            <div className="node-header">
              <span className="node-icon">{getNodeIcon(data.type)}</span>
              <span className="node-title text-red-600">Error: {formatValue(data.metadata?.error_type)}</span>
            </div>
            <div className="node-body">
              <div className="text-xs text-red-500 mt-1">
                <div className="truncate">{truncateText(formatValue(data.metadata?.error_message, ''), 60)}</div>
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  const getExpandedContent = () => {
    switch (data.type) {
      case 'trace':
        const counts = data.metadata?.totalCounts || {};
        const directLLMCalls = data.metadata?.llm_calls || [];
        const directToolCalls = data.metadata?.tool_calls || [];

        // Get all nodes including nested ones within agent calls
        const allNodes = [
          ...(data.metadata?.agent_calls || []).flatMap(agent => [
            agent,
            ...(agent.llm_calls || []),
            ...(agent.tool_calls || [])
          ]),
          ...(data.metadata?.llm_calls || []),
          ...(data.metadata?.tool_calls || [])
        ];

        // Find earliest start time and latest end time
        const startTimes = allNodes
          .map(node => node.start_time)
          .filter(time => time)
          .map(time => new Date(time).getTime());
        
        const endTimes = allNodes
          .map(node => node.end_time)
          .filter(time => time)
          .map(time => new Date(time).getTime());

        const traceStartTime = startTimes.length ? Math.min(...startTimes) : data.startTime;
        const traceEndTime = endTimes.length ? Math.max(...endTimes) : data.endTime;
        const traceDuration = calculateDuration(
          new Date(traceStartTime).toISOString(),
          new Date(traceEndTime).toISOString()
        );

        return (
          <div className="text-sm">
            <div className="font-bold mb-2">Trace Details</div>
            <div className="grid grid-cols-2 gap-2 mb-2">
              <div>Name: {formatValue(data.name)}</div>
              <div>Duration: {traceDuration}</div>
              <div>Start: {formatTimestamp(new Date(traceStartTime).toISOString())}</div>
              <div>End: {formatTimestamp(new Date(traceEndTime).toISOString())}</div>
            </div>

            {/* Show agent calls if present */}
            {data.metadata?.agent_calls?.length > 0 && (
              <div className="mt-4">
                <div className="font-semibold mb-2">Agent Calls ({data.metadata?.agent_calls?.length})</div>
                <div className="space-y-2 max-h-60 overflow-auto bg-gray-50 p-2 rounded">
                  {data.metadata?.agent_calls?.map((call: any, index: number) => (
                    <div key={index} className="text-xs border-b pb-2">
                      <div>Agent: {call.name}</div>
                      <div>Duration: {calculateDuration(call.start_time, call.end_time)}</div>
                      <div>LLM Calls: {call.llm_calls?.length || 0}</div>
                      <div>Tool Calls: {call.tool_calls?.length || 0}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Show direct LLM calls if present */}
            {directLLMCalls.length > 0 && (
              <div className="mt-4">
                <div className="font-semibold mb-2">Direct LLM Calls ({directLLMCalls.length})</div>
                <div className="space-y-2 max-h-60 overflow-auto bg-gray-50 p-2 rounded">
                  {directLLMCalls.map((call: any, index: number) => (
                    <div key={index} className="text-xs border-b pb-2">
                      <div>Model: {call.model}</div>
                      <div>Duration: {call.duration}s</div>
                      <div className="truncate">Input: {truncateText(call.input_prompt, 100)}</div>
                      <div className="truncate">Output: {truncateText(call.output, 100)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Show direct tool calls if present */}
            {directToolCalls.length > 0 && (
              <div className="mt-4">
                <div className="font-semibold mb-2">Direct Tool Calls ({directToolCalls.length})</div>
                <div className="space-y-2 max-h-60 overflow-auto bg-gray-50 p-2 rounded">
                  {directToolCalls.map((call: any, index: number) => (
                    <div key={index} className="text-xs border-b pb-2">
                      <div>Tool: {call.name}</div>
                      <div>Duration: {call.duration}s</div>
                      <div className="truncate">Input: {truncateText(formatJson(call.input_parameters), 100)}</div>
                      <div className="truncate">Output: {truncateText(call.output, 100)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="mt-4">
              <div className="font-semibold mb-2">Statistics</div>
              <div className="grid grid-cols-2 gap-2 bg-gray-50 p-2 rounded">
                <div>Total LLM Calls: {counts.llmCount || directLLMCalls.length}</div>
                <div>Total Tool Calls: {counts.toolCount || directToolCalls.length}</div>
                <div>Total Agents: {data.metadata?.agent_calls?.length || 0}</div>
              </div>
            </div>

            {data.metadata?.errors?.length > 0 && (
              <div className="mt-4">
                <div className="font-semibold mb-2 text-red-600">Errors</div>
                <div className="space-y-2 max-h-40 overflow-auto bg-red-50 p-2 rounded">
                  {data.metadata.errors.map((error: any, index: number) => (
                    <div key={index} className="text-xs text-red-600">
                      {error.message || JSON.stringify(error)}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 'llm':
        return (
          <div className="text-sm">
            <div className="font-bold mb-2">Tool Call Details</div>
            <div>Model: {formatValue(data.metadata?.model)}</div>
            <div>Start: {formatValue(formatTimestamp(data.startTime))}</div>
            <div>End: {formatValue(formatTimestamp(data.endTime))}</div>
            <div>Duration: {formatValue(data.duration)}</div>
            <div>Memory: {formatValue(data.metadata?.memory_used, '0 MB')}</div>

            <div className="mt-2">
              <div className="font-semibold">Cost:</div>
              <div className="bg-gray-50 p-2 rounded text-xs">
                {formatJson(data.metadata?.cost)}
              </div>
            </div>

            <div className="mt-2">
              <div className="font-semibold">Token Usage:</div>
              <div className="bg-gray-50 p-2 rounded text-xs">
                {formatJson(data.metadata?.token_usage)}
              </div>
            </div>

            <div className="mt-2">
              <div className="font-semibold">Input Prompt:</div>
              <div className="whitespace-pre-wrap overflow-auto max-h-40 bg-gray-50 p-2 rounded text-xs">
                {formatValue(data.metadata?.input_prompt, 'No input')}
              </div>
              
              <div className="font-semibold mt-2">Output:</div>
              <div className="whitespace-pre-wrap overflow-auto max-h-40 bg-gray-50 p-2 rounded text-xs">
                {formatValue(data.metadata?.output, 'No output')}
              </div>
            </div>
          </div>
        );
      
        case 'agent':
          return (
            <div className="text-sm">
              <div className="font-bold mb-2">Agent Details</div>
              <div className="grid grid-cols-2 gap-2 mb-2">
                <div>Name: {formatValue(data.name)}</div>
                <div>Duration: {formatValue(data.duration)}</div>
                <div>Start: {formatTimestamp(data.metadata?.start_time)}</div>
                <div>End: {formatTimestamp(data.metadata?.end_time)}</div>
              </div>
        
              <div className="mt-4">
                <div className="font-semibold mb-2">LLM Calls ({data.metadata?.llm_calls?.length || 0})</div>
                <div className="space-y-2 max-h-60 overflow-auto bg-gray-50 p-2 rounded">
                  {data.metadata?.llm_calls?.map((call: any, index: number) => (
                    <div key={index} className="text-xs border-b pb-2">
                      <div>Model: {call.model}</div>
                      <div>Duration: {call.duration}s</div>
                      <div className="truncate">Input: {truncateText(call.input_prompt, 100)}</div>
                      <div className="truncate">Output: {truncateText(call.output, 100)}</div>
                    </div>
                  ))}
                </div>
              </div>
        
              <div className="mt-4">
                <div className="font-semibold mb-2">Tool Calls ({data.metadata?.tool_calls?.length || 0})</div>
                <div className="space-y-2 max-h-60 overflow-auto bg-gray-50 p-2 rounded">
                  {data.metadata?.tool_calls?.map((call: any, index: number) => (
                    <div key={index} className="text-xs border-b pb-2">
                      <div>Tool: {call.name}</div>
                      <div>Duration: {call.duration}s</div>
                      <div className="truncate">Input: {truncateText(formatJson(call.input_parameters), 100)}</div>
                      <div className="truncate">Output: {truncateText(call.output, 100)}</div>
                    </div>
                  ))}
                </div>
              </div>
        
              {data.metadata?.errors?.length > 0 && (
                <div className="mt-4">
                  <div className="font-semibold mb-2 text-red-600">Errors</div>
                  <div className="space-y-2 max-h-40 overflow-auto bg-red-50 p-2 rounded">
                    {data.metadata.errors.map((error: any, index: number) => (
                      <div key={index} className="text-xs text-red-600">
                        {error.message || JSON.stringify(error)}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );

      case 'tool':
        return (
          <div className="text-sm">
            <div className="font-bold mb-2">Tool Call Details</div>
            <div>Name: {formatValue(data.name)}</div>
            <div>Start: {formatValue(formatTimestamp(data.startTime))}</div>
            <div>End: {formatValue(formatTimestamp(data.endTime))}</div>
            <div>Duration: {formatValue(data.duration)}</div>
            <div>Memory: {formatValue(data.metadata?.memory_used, '0 MB')}</div>
            
            <div className="mt-2">
              <div className="font-semibold">Network Calls:</div>
              <div className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-2 rounded">
                {formatNetworkCalls(data.metadata?.network_calls)}
              </div>
            </div>

            <div className="mt-2">
              <div className="font-semibold">Input Parameters:</div>
              <div className="whitespace-pre-wrap overflow-auto max-h-40 bg-gray-50 p-2 rounded text-xs">
                {formatJson(data.metadata?.input_parameters)}
              </div>
              
              <div className="font-semibold mt-2">Output:</div>
              <div className="whitespace-pre-wrap overflow-auto max-h-40 bg-gray-50 p-2 rounded text-xs">
                {formatValue(data.metadata?.output, 'No output')}
              </div>
            </div>
          </div>
        );

      case 'interaction':
        return (
          <div className="text-sm">
            <div className="font-bold mb-2">Interaction Details</div>
            <div>Type: {formatValue(data.metadata?.interaction_type)}</div>
            <div>Time: {formatValue(formatTimestamp(data.metadata?.timestamp))}</div>
            <div className="mt-2">
              <div className="font-semibold">Content:</div>
              <div className="whitespace-pre-wrap bg-gray-50 p-2 rounded">
                {formatValue(data.metadata?.content, 'No content')}
              </div>
            </div>
          </div>
        );

      case 'error':
        return (
          <div className="text-sm">
            <div className="font-bold mb-2 text-red-600">Error Details</div>
            <div>Type: {formatValue(data.metadata?.error_type)}</div>
            <div>Time: {formatValue(formatTimestamp(data.metadata?.timestamp))}</div>
            <div className="mt-2">
              <div className="font-semibold">Message:</div>
              <div className="whitespace-pre-wrap text-red-600 bg-gray-50 p-2 rounded">
                {formatValue(data.metadata?.error_message, 'No error message')}
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div
      className={`node-container ${isExpanded ? 'expanded' : ''} ${isHovered ? 'hover' : ''}`}
      onClick={() => setIsExpanded(!isExpanded)}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        backgroundColor: nodeStylesByType[data.type]?.backgroundColor || '#fff',
        width: isExpanded ? '500px' : '300px', 
        minHeight: isExpanded ? '300px' : '160px', 
        maxHeight: isExpanded ? '800px' : '160px',
      }}
      data-type={data.type}
    >
      <Handle type="target" position={Position.Top} />
      <div className={`${isExpanded ? 'overflow-auto' : ''}`}>
        {isExpanded ? getExpandedContent() : getCollapsedContent()}
      </div>
      {!isExpanded && (
        <div className="expand-hint">
          Click to expand {isHovered ? '👆' : ''}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
};

export const nodeStylesByType = {
  trace: {
    backgroundColor: '#e3f2fd',
    className: 'border-blue-200',
  },
  agent: {
    backgroundColor: '#f3e5f5',
    className: 'border-purple-200',
  },
  llm: {
    backgroundColor: '#e8f5e9',
    className: 'border-green-200',
  },
  tool: {
    backgroundColor: '#fff3e0',
    className: 'border-orange-200',
  },
  interaction: {
    backgroundColor: '#f3e8ff',
    className: 'border-purple-200',
  },
  error: {
    backgroundColor: '#fee2e2',
    className: 'border-red-200',
  }
};

export const nodeTypes = {
  custom: CustomNode,
};
