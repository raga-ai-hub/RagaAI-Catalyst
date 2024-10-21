import React from 'react';
import { Handle, Position } from 'reactflow';

const truncateString = (str: string, maxLength: number) => {
  if (!str) return '';
  return str.length <= maxLength ? str : str.slice(0, maxLength) + '...';
};

const formatDuration = (ms: number) => {
  const seconds = ms / 1000;
  return seconds < 1 ? `${ms}ms` : `${seconds.toFixed(2)}s`;
};

const formatTimestamp = (timestamp: string) => {
  return new Date(timestamp).toLocaleTimeString();
};

export const CustomNode = ({ data }: { data: any }) => {
  const renderNodeContent = () => {
    switch (data.type) {
      case 'agent':
        return (
          <>
            <div className="node-title">{data.name}</div>
            <div className="node-detail">Duration: {formatDuration(data.duration)}</div>
          </>
        );
      case 'llm':
        return (
          <>
            <div className="node-title">{data.name}</div>
            <div className="node-detail">Model: {data.model}</div>
            <div className="node-detail">Duration: {formatDuration(data.duration)}</div>
            <div className="node-detail">Tokens: {data.token_usage?.input + data.token_usage?.completion}</div>
            <div className="node-detail">Input: {truncateString(data.input_prompt, 30)}</div>
            <div className="node-detail">Output: {truncateString(data.output, 30)}</div>
          </>
        );
      case 'tool':
        return (
          <>
            <div className="node-title">{data.name}</div>
            <div className="node-detail">Duration: {formatDuration(data.duration)}</div>
            <div className="node-detail">Network Calls: {data.network_calls || 'None'}</div>
            <div className="node-detail">Input: {truncateString(data.input_parameters, 30)}</div>
            <div className="node-detail">Output: {truncateString(data.output, 30)}</div>
          </>
        );
      case 'user_interaction':
        return (
          <>
            <div className="node-title">User Interaction</div>
            <div className="node-chat">
              {Array.isArray(data.interactions) ? (
                data.interactions.map((interaction: any, index: number) => (
                  <div key={index} className={`chat-message ${interaction.interaction_type}`}>
                    <div className="chat-timestamp">{formatTimestamp(interaction.timestamp)}</div>
                    <div className="chat-content">{truncateString(interaction.content, 50)}</div>
                  </div>
                ))
              ) : (
                <div className="chat-message">
                  <div className="chat-timestamp">{formatTimestamp(data.timestamp)}</div>
                  <div className="chat-content">{truncateString(data.content, 50)}</div>
                </div>
              )}
            </div>
          </>
        );
      case 'error':
        return (
          <>
            <div className="node-title">Error: {data.name}</div>
            <div className="node-detail">Message: {truncateString(data.error_message, 50)}</div>
          </>
        );
      default:
        return <div>Unknown node type</div>;
    }
  };

  return (
    <div className={`custom-node ${data.type}`}>
      <Handle type="target" position={Position.Top} />
      {renderNodeContent()}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
};

export const nodeTypes = {
  custom: CustomNode,
};

export const nodeStylesByType = {
  agent: { backgroundColor: '#E6F3FF', borderColor: '#2196F3' },
  llm: { backgroundColor: '#E8F5E9', borderColor: '#4CAF50' },
  tool: { backgroundColor: '#FFF3E0', borderColor: '#FF9800' },
  user_interaction: { backgroundColor: '#F3E5F5', borderColor: '#9C27B0' },
  error: { backgroundColor: '#FFEBEE', borderColor: '#F44336' },
};