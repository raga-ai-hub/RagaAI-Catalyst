import React, { useState, useMemo } from 'react';
import {
  ChevronRight,
  Clock,
  AlertCircle,
  Bot,
  Wrench,
  MessageSquare,
  ArrowDownCircle,
  ArrowUpCircle,
  X,
  Code,
  Cpu,
  CircleDollarSign,
  Database,
  User,
  AlertTriangle,
} from 'lucide-react';

const utils = {
  formatDuration: (duration) => {
    if (typeof duration === 'string') return duration;
    if (typeof duration === 'number') {
      // Convert to seconds if in milliseconds
      const seconds = duration > 1000 ? duration / 1000 : duration;
      return `${seconds.toFixed(3)}s`;
    }
    return 'N/A';
  },

  formatTime: (timestamp) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  },

  parseJSON: (str) => {
    try {
      return JSON.parse(str);
    } catch (e) {
      return null;
    }
  }
};

const CallDetailsSection = ({ title, children }) => (
  <div className="bg-gray-50 p-3 rounded mb-4">
    <h4 className="text-sm font-medium mb-2">{title}</h4>
    <div className="space-y-2">
      {children}
    </div>
  </div>
);

const UserInteractionMessage = ({ interaction, type }) => {
  const isUser = type === 'input';

  return (
    <div className={`flex gap-3 mb-3 ${isUser ? 'justify-start' : 'justify-end'}`}>
      <div className={`
        max-w-[80%] rounded-lg p-3
        ${isUser ? 'bg-gray-100' : 'bg-blue-50'}
      `}>
        <div className="flex items-center gap-2 mb-1">
          {isUser ? (
            <User className="w-4 h-4 text-gray-600" />
          ) : (
            <Bot className="w-4 h-4 text-blue-600" />
          )}
          <span className="text-xs text-gray-500">
            {utils.formatTime(interaction.timestamp)}
          </span>
        </div>
        <div className="text-sm whitespace-pre-wrap">
          {interaction.content}
        </div>
      </div>
    </div>
  );
};

const TimelineSegment = ({
  call,
  timelineStart,
  timelineDuration,
  depth,
  isSelected,
  onSelect
}) => {
  const startOffset = (((call.startTime - timelineStart) / 1000) / timelineDuration) * 100;
  const duration = (call.duration / timelineDuration) * 100;

  let color;
  switch (call.type) {
    case 'agent': color = 'violet'; break;
    case 'llm': color = 'green'; break;
    case 'tool': color = 'blue'; break;
    default: color = 'gray';
  }

  return (
    <div
      className={`
        absolute h-6 group cursor-pointer
        ${isSelected ? 'z-20' : 'z-10'}
      `}
      style={{
        left: `${startOffset}%`,
        width: `${duration}%`,
        top: `${depth * 28}px`
      }}
      onClick={() => onSelect(call)}
    >
      <div className={`
        absolute -left-1 top-1/2 w-2 h-2 rounded-full bg-${color}-600
        transform -translate-y-1/2
        ${isSelected ? 'ring-2 ring-offset-1 ring-blue-400' : ''}
      `} />

      <div className={`
        h-2 bg-${color}-100 border border-${color}-600
        relative top-1/2 transform -translate-y-1/2
        ${isSelected ? 'ring-1 ring-blue-400' : ''}
      `} />

      <div className={`
        absolute -right-1 top-1/2 w-2 h-2 rounded-full bg-${color}-600
        transform -translate-y-1/2
        ${isSelected ? 'ring-2 ring-offset-1 ring-blue-400' : ''}
      `} />

      <div className={`
        absolute -top-6 left-0 p-1 rounded bg-gray-800 text-white text-xs
        whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity z-30
      `}>
        {call.name || `${call.type} ${call.id}`} ({utils.formatDuration(call.duration)})
      </div>
    </div>
  );
};

const TraceDetailsPanel = ({ isOpen, onClose, traceData }) => {
  const [expandedSections, setExpandedSections] = useState(new Set());
  const [selectedSection, setSelectedSection] = useState(null);

  const chronologicalCalls = useMemo(() => {
    if (!traceData) return [];

    const allCalls = [];
    const processCall = (call, type, parentId = null, depth = 0) => {
      const startTime = new Date(call.start_time).getTime();
      const endTime = new Date(call.end_time).getTime();

      // Convert duration to seconds
      const duration = (endTime - startTime) / 1000;

      allCalls.push({
        ...call,
        type,
        startTime,
        endTime,
        duration,
        parentId,
        depth
      });

      // Process nested calls
      if (call.llm_calls) {
        call.llm_calls.forEach(llmCall =>
          processCall(llmCall, 'llm', call.id, depth + 1)
        );
      }
      if (call.tool_calls) {
        call.tool_calls.forEach(toolCall =>
          processCall(toolCall, 'tool', call.id, depth + 1)
        );
      }
    };

    // Process agent calls
    traceData.agent_calls?.forEach(call => processCall(call, 'agent'));

    // Process standalone LLM calls
    traceData.llm_calls?.forEach(call => processCall(call, 'llm'));

    // Process standalone tool calls
    traceData.tool_calls?.forEach(call => processCall(call, 'tool'));

    return allCalls.sort((a, b) => a.startTime - b.startTime);
  }, [traceData]);

  const timelineStart = useMemo(() => {
    if (chronologicalCalls.length === 0) return 0;
    return Math.min(...chronologicalCalls.map(call => call.startTime));
  }, [chronologicalCalls]);

  const timelineDuration = useMemo(() => {
    if (chronologicalCalls.length === 0) return 0;
    const endTime = Math.max(...chronologicalCalls.map(call => call.endTime));
    return (endTime - timelineStart) / 1000; // Convert to seconds
  }, [chronologicalCalls, timelineStart]);

  const renderErrors = (errors) => {
    if (!errors || errors.length === 0) return null;

    return (
      <CallDetailsSection title={
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-red-500" />
          Errors
        </div>
      }>
        {errors.map((error, index) => (
          <div key={index} className="bg-red-50 border border-red-200 rounded p-2">
            <div className="text-sm font-medium text-red-700">{error.error_type}</div>
            <div className="text-sm text-red-600">{error.error_message}</div>
            <div className="text-xs text-red-500 mt-1">
              {new Date(error.timestamp).toLocaleString()}
            </div>
          </div>
        ))}
      </CallDetailsSection>
    );
  };

  const renderCallDetails = (call) => {
    return (
      <div className="p-4 border-t space-y-4">
        <div className="grid grid-cols-2 gap-2">
          {call.model && (
            <div className="bg-gray-50 p-2 rounded flex items-start gap-2">
              <Cpu className="w-4 h-4 text-gray-400 mt-0.5" />
              <div>
                <div className="text-xs text-gray-500">Model</div>
                <div className="text-sm font-medium">{call.model}</div>
              </div>
            </div>
          )}
          {typeof call.memory_used === 'number' && (
            <div className="bg-gray-50 p-2 rounded flex items-start gap-2">
              <Database className="w-4 h-4 text-gray-400 mt-0.5" />
              <div>
                <div className="text-xs text-gray-500">Memory Used</div>
                <div className="text-sm font-medium">
                  {(call.memory_used / 1024 / 1024).toFixed(2)} MB
                </div>
              </div>
            </div>
          )}
        </div>

        {/* User Interactions */}
        {call.user_interactions && call.user_interactions.length > 0 && (
          <CallDetailsSection title={
            <div className="flex items-center gap-2">
              <User className="w-4 h-4 text-gray-400" />
              Conversation
            </div>
          }>
            <div className="space-y-2">
              {call.user_interactions.map((interaction, index) => (
                <UserInteractionMessage
                  key={index}
                  interaction={interaction}
                  type={interaction.interaction_type.toLowerCase()}
                />
              ))}
            </div>
          </CallDetailsSection>
        )}

        {/* Input/Output */}
        {(call.input_prompt || call.input_parameters) && (
          <CallDetailsSection title={
            <div className="flex items-center gap-2">
              <ArrowDownCircle className="w-4 h-4 text-gray-400" />
              Input
            </div>
          }>
            <div className="bg-gray-800 text-gray-100 rounded-md p-3 text-xs overflow-auto">
              {call.input_prompt && (
                <div className="mb-2">
                  <div className="text-gray-400 mb-1">Prompt:</div>
                  <pre className="whitespace-pre-wrap">
                    {typeof call.input_prompt === 'string'
                      ? call.input_prompt
                      : JSON.stringify(call.input_prompt, null, 2)}
                  </pre>
                </div>
              )}
              {call.input_parameters && (
                <div>
                  <div className="text-gray-400 mb-1">Parameters:</div>
                  <pre className="whitespace-pre-wrap">
                    {JSON.stringify(utils.parseJSON(call.input_parameters), null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </CallDetailsSection>
        )}

        {call.output && (
          <CallDetailsSection title={
            <div className="flex items-center gap-2">
              <ArrowUpCircle className="w-4 h-4 text-gray-400" />
              Output
            </div>
          }>
            <div className="bg-gray-800 text-gray-100 rounded-md p-3 text-xs overflow-auto">
              <pre className="whitespace-pre-wrap">
                {typeof call.output === 'string'
                  ? call.output
                  : JSON.stringify(call.output, null, 2)}
              </pre>
            </div>
          </CallDetailsSection>
        )}

        {/* Token Usage */}
        {call.token_usage && (
          <CallDetailsSection title={
            <div className="flex items-center gap-2">
              <MessageSquare className="w-4 h-4 text-gray-400" />
              Token Usage
            </div>
          }>
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(utils.parseJSON(call.token_usage) || {}).map(([key, value]) => (
                <div key={key} className="bg-white p-2 rounded border">
                  <div className="text-xs text-gray-500">{key}</div>
                  <div className="text-sm font-medium">{String(value)}</div>
                </div>
              ))}
            </div>
          </CallDetailsSection>
        )}

        {/* Cost */}
        {call.cost && (
          <CallDetailsSection title={
            <div className="flex items-center gap-2">
              <CircleDollarSign className="w-4 h-4 text-gray-400" />
              Cost
            </div>
          }>
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(utils.parseJSON(call.cost) || {}).map(([key, value]) => (
                <div key={key} className="bg-white p-2 rounded border">
                  <div className="text-xs text-gray-500">{key}</div>
                  <div className="text-sm font-medium">${Number(value).toFixed(6)}</div>
                </div>
              ))}
            </div>
          </CallDetailsSection>
        )}

        {/* Errors */}
        {renderErrors(call.errors)}
      </div>
    );
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case 'agent': return { icon: Bot, color: 'text-violet-600', bgColor: 'bg-violet-100' };
      case 'llm': return { icon: MessageSquare, color: 'text-green-600', bgColor: 'bg-green-100' };
      case 'tool': return { icon: Wrench, color: 'text-blue-600', bgColor: 'bg-blue-100' };
      default: return { icon: Code, color: 'text-gray-600', bgColor: 'bg-gray-100' };
    }
  };

  const renderCall = (call) => {
    const { icon: Icon, color, bgColor } = getTypeIcon(call.type);
    const callId = `${call.type}-${call.id}`;
    const isExpanded = expandedSections.has(callId);
    const isSelected = selectedSection?.id === call.id;

    return (
      <div key={callId} style={{ marginLeft: `${call.depth * 16}px` }}>
        <div
          className={`
            rounded-lg border transition-all duration-200 mb-2
            ${isSelected ? 'border-blue-300 bg-blue-50' : 'border-gray-200 bg-white'}
          `}
        >
          <div
            className="flex items-center gap-2 p-3 cursor-pointer hover:bg-gray-50"
            onClick={() => {
              setSelectedSection(isSelected ? null : call);
              setExpandedSections(prev => {
                const next = new Set(prev);
                if (next.has(callId)) {
                  next.delete(callId);
                } else {
                  next.add(callId);
                }
                return next;
              });
            }}
          >
            <Icon className={`w-4 h-4 ${color}`} />
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-medium">
                    {call.name || `${call.type.charAt(0).toUpperCase() + call.type.slice(1)} ${call.id}`}
                  </span>
                  <span className={`px-2 py-0.5 rounded-full text-xs ${bgColor} ${color}`}>
                    {call.type}
                  </span>
                </div>
                <span className="text-sm text-gray-500">{utils.formatDuration(call.duration)}</span>
              </div>
              <div className="text-xs text-gray-500 mt-1 flex items-center gap-2">
                <Clock className="w-3 h-3" />
                {utils.formatTime(call.startTime)}
              </div>
            </div>
          </div>

          {isSelected && renderCallDetails(call)}
        </div>
      </div>
    );
  };

  return (
    <div className={`fixed inset-y-0 right-0 w-[600px] bg-white shadow-xl transform transition-transform
      ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}>
      <div className="h-full flex flex-col">
        {/* Header */}
        <div className="p-4 border-b flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Bot className="w-5 h-5 text-purple-600" />
              Trace Details
            </h2>
            {traceData && (
              <div className="text-sm text-gray-500 mt-1">
                ID: {traceData.id}
              </div>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-full"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {traceData && (
          <>
            {/* Summary Section */}
            <div className="p-4 space-y-4 border-b">
              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center gap-2 bg-white p-3 rounded-lg border">
                  <Clock className="w-5 h-5 text-blue-600" />
                  <div>
                    <div className="text-sm text-gray-500">Duration</div>
                    <div className="font-medium">{utils.formatDuration(traceData.duration)}</div>
                  </div>
                </div>
                <div className="flex items-center gap-2 bg-white p-3 rounded-lg border">
                  <AlertCircle className="w-5 h-5 text-red-600" />
                  <div>
                    <div className="text-sm text-gray-500">Errors</div>
                    <div className="font-medium">{Array.isArray(traceData.errors) ? traceData.errors.length : 0}</div>
                  </div>
                </div>
              </div>

              {/* Timeline */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="text-sm font-medium">Timeline</div>
                  <div className="text-xs text-gray-500">
                    {utils.formatTime(timelineStart)} - {utils.formatTime(timelineStart + timelineDuration * 1000)}
                  </div>
                </div>

                <div
                  className="relative bg-gray-100 rounded overflow-hidden"
                  style={{
                    height: `${Math.max(...chronologicalCalls.map(c => c.depth + 1)) * 28}px`,
                    minHeight: '56px'
                  }}
                >
                  {chronologicalCalls.map(call => (
                    <TimelineSegment
                      key={`${call.type}-${call.id}`}
                      call={call}
                      timelineStart={timelineStart}
                      timelineDuration={timelineDuration}
                      depth={call.depth}
                      isSelected={selectedSection?.id === call.id}
                      onSelect={setSelectedSection}
                    />
                  ))}
                </div>

                {/* Timeline Legend */}
                <div className="mt-2 flex flex-wrap gap-3 text-xs text-gray-500">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-violet-600" />
                    Agent Calls
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-green-600" />
                    LLM Calls
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-blue-600" />
                    Tool Calls
                  </div>
                </div>
              </div>
            </div>

            {/* Call List */}
            <div className="flex-1 overflow-y-auto p-4">
              <div className="space-y-1">
                {chronologicalCalls.map((call, index) => (
                  <React.Fragment key={`${call.type}-${call.id}`}>
                    {index === 0 ||
                      utils.formatTime(call.startTime).split(':')[1] !==
                      utils.formatTime(chronologicalCalls[index - 1].startTime).split(':')[1] ? (
                      <div className="text-xs text-gray-400 mt-4 mb-2 sticky top-0 bg-white py-1">
                        {utils.formatTime(call.startTime)}
                      </div>
                    ) : null}
                    {renderCall(call)}
                  </React.Fragment>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default TraceDetailsPanel;