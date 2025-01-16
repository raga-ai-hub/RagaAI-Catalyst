import React, { useState, useMemo } from 'react'
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
  AlertTriangle
} from 'lucide-react'
import { useTheme } from '@/theme/ThemeProvider'

const utils = {
  formatDuration: duration => {
    if (typeof duration === 'string') return duration
    if (typeof duration === 'number') {
      // Convert to seconds if in milliseconds
      const seconds = duration > 1000 ? duration / 1000 : duration
      return `${seconds.toFixed(3)}s`
    }
    return 'N/A'
  },

  formatTime: timestamp => {
    if (!timestamp) return 'N/A'
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  },

  parseJSON: str => {
    try {
      return JSON.parse(str)
    } catch (e) {
      return null
    }
  }
}

const CallDetailsSection = ({ title, children }) => {
  const { theme } = useTheme()

  return (
    <div
      className={`${
        theme === 'dark'
          ? 'bg-gray-800 border-gray-700'
          : 'bg-gray-50 border-gray-200'
      } 
      p-3 rounded mb-4 border`}
    >
      <h4
        className={`text-sm font-medium mb-2 
        ${theme === 'dark' ? 'text-gray-100' : 'text-gray-900'}`}
      >
        {title}
      </h4>
      <div className='space-y-2'>{children}</div>
    </div>
  )
}

const UserInteractionMessage = ({ interaction, type }) => {
  const { theme } = useTheme()
  const isUser = type === 'input'

  const getBgColor = () => {
    if (theme === 'dark') {
      return isUser ? 'bg-gray-900' : 'bg-blue-900'
    }
    return isUser ? 'bg-gray-100' : 'bg-blue-50'
  }

  const getBorderColor = () => {
    if (theme === 'dark') {
      return isUser ? 'border-gray-700' : 'border-blue-800'
    }
    return isUser ? 'border-gray-200' : 'border-blue-100'
  }

  const getIconColor = () => {
    if (theme === 'dark') {
      return isUser ? 'text-gray-400' : 'text-blue-400'
    }
    return isUser ? 'text-gray-600' : 'text-blue-600'
  }

  const getTextColor = () => {
    if (theme === 'dark') {
      return 'text-gray-100'
    }
    return 'text-gray-900'
  }

  const getTimestampColor = () => {
    if (theme === 'dark') {
      return 'text-gray-400'
    }
    return 'text-gray-500'
  }

  return (
    <div
      className={`flex gap-3 mb-3 ${isUser ? 'justify-start' : 'justify-end'}`}
    >
      <div
        className={`
        max-w-[80%] rounded-lg p-3 border
        ${getBgColor()}
        ${getBorderColor()}
      `}
      >
        <div className='flex items-center gap-2 mb-1'>
          {isUser ? (
            <User className={`w-4 h-4 ${getIconColor()}`} />
          ) : (
            <Bot className={`w-4 h-4 ${getIconColor()}`} />
          )}
          <span className={`text-xs ${getTimestampColor()}`}>
            {utils.formatTime(interaction.timestamp)}
          </span>
        </div>
        <div className={`text-sm whitespace-pre-wrap ${getTextColor()}`}>
          {interaction.content}
        </div>
      </div>
    </div>
  )
}

const TimelineSegment = ({
  call,
  timelineStart,
  timelineDuration,
  depth,
  isSelected,
  onSelect
}) => {
  const startOffset =
    ((call.startTime - timelineStart) / 1000 / timelineDuration) * 100
  const duration = (call.duration / timelineDuration) * 100

  let color
  switch (call.type) {
    case 'agent':
      color = 'violet'
      break
    case 'llm':
      color = 'green'
      break
    case 'tool':
      color = 'blue'
      break
    default:
      color = 'gray'
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
      <div
        className={`
        absolute -left-1 top-1/2 w-2 h-2 rounded-full bg-${color}-600
        transform -translate-y-1/2
        ${isSelected ? 'ring-2 ring-offset-1 ring-blue-400' : ''}
      `}
      />

      <div
        className={`
        h-2 bg-${color}-100 border border-${color}-600
        relative top-1/2 transform -translate-y-1/2
        ${isSelected ? 'ring-1 ring-blue-400' : ''}
      `}
      />

      <div
        className={`
        absolute -right-1 top-1/2 w-2 h-2 rounded-full bg-${color}-600
        transform -translate-y-1/2
        ${isSelected ? 'ring-2 ring-offset-1 ring-blue-400' : ''}
      `}
      />

      <div
        className={`
        absolute -top-6 left-0 p-1 rounded bg-gray-800 text-white text-xs
        whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity z-30
      `}
      >
        {call.name || `${call.type} ${call.id}`} (
        {utils.formatDuration(call.duration)})
      </div>
    </div>
  )
}

const TraceDetailsPanel = ({ isOpen, onClose, traceData }) => {
  const { theme } = useTheme()
  const [expandedSections, setExpandedSections] = useState(new Set())
  const [selectedSection, setSelectedSection] = useState(null)

  const chronologicalCalls = useMemo(() => {
    if (!traceData) return []

    const allCalls = []
    const processCall = (call, type, parentId = null, depth = 0) => {
      const startTime = new Date(call.start_time).getTime()
      const endTime = new Date(call.end_time).getTime()

      // Convert duration to seconds
      const duration = (endTime - startTime) / 1000

      allCalls.push({
        ...call,
        type,
        startTime,
        endTime,
        duration,
        parentId,
        depth
      })

      // Process nested calls
      if (call.llm_calls) {
        call.llm_calls.forEach(llmCall =>
          processCall(llmCall, 'llm', call.id, depth + 1)
        )
      }
      if (call.tool_calls) {
        call.tool_calls.forEach(toolCall =>
          processCall(toolCall, 'tool', call.id, depth + 1)
        )
      }
    }

    // Process agent calls
    traceData.agent_calls?.forEach(call => processCall(call, 'agent'))

    // Process standalone LLM calls
    traceData.llm_calls?.forEach(call => processCall(call, 'llm'))

    // Process standalone tool calls
    traceData.tool_calls?.forEach(call => processCall(call, 'tool'))

    return allCalls.sort((a, b) => a.startTime - b.startTime)
  }, [traceData])

  const timelineStart = useMemo(() => {
    if (chronologicalCalls.length === 0) return 0
    return Math.min(...chronologicalCalls.map(call => call.startTime))
  }, [chronologicalCalls])

  const timelineDuration = useMemo(() => {
    if (chronologicalCalls.length === 0) return 0
    const endTime = Math.max(...chronologicalCalls.map(call => call.endTime))
    return (endTime - timelineStart) / 1000 // Convert to seconds
  }, [chronologicalCalls, timelineStart])

  const renderErrors = errors => {
    if (!errors || errors.length === 0) return null

    return (
      <CallDetailsSection
        title={
          <div className='flex items-center gap-2'>
            <AlertTriangle className='w-4 h-4 text-red-500' />
            Errors
          </div>
        }
      >
        {errors.map((error, index) => (
          <div
            key={index}
            className='bg-red-50 border border-red-200 rounded p-2'
          >
            <div className='text-sm font-medium text-red-700'>
              {error.error_type}
            </div>
            <div className='text-sm text-red-600'>{error.error_message}</div>
            <div className='text-xs text-red-500 mt-1'>
              {new Date(error.timestamp).toLocaleString()}
            </div>
          </div>
        ))}
      </CallDetailsSection>
    )
  }

  const renderCallDetails = call => {
    const { theme } = useTheme()

    return (
      <div
        className={`p-4 border-t ${
          theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
        } space-y-4`}
      >
        <div className='grid grid-cols-2 gap-2'>
          {call.model && (
            <div
              className={`${theme === 'dark' ? 'bg-gray-800' : 'bg-gray-50'} 
              p-2 rounded flex items-start gap-2 
              ${
                theme === 'dark'
                  ? 'border border-gray-700'
                  : 'border border-gray-200'
              }`}
            >
              <Cpu
                className={`w-4 h-4 ${
                  theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
                } mt-0.5`}
              />
              <div>
                <div
                  className={`text-xs ${
                    theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                  }`}
                >
                  Model
                </div>
                <div
                  className={`text-sm font-medium ${
                    theme === 'dark' ? 'text-gray-100' : 'text-gray-900'
                  }`}
                >
                  {call.model}
                </div>
              </div>
            </div>
          )}
          {typeof call.memory_used === 'number' && (
            <div
              className={`${theme === 'dark' ? 'bg-gray-800' : 'bg-gray-50'} 
              p-2 rounded flex items-start gap-2
              ${
                theme === 'dark'
                  ? 'border border-gray-700'
                  : 'border border-gray-200'
              }`}
            >
              <Database
                className={`w-4 h-4 ${
                  theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
                } mt-0.5`}
              />
              <div>
                <div
                  className={`text-xs ${
                    theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                  }`}
                >
                  Memory Used
                </div>
                <div
                  className={`text-sm font-medium ${
                    theme === 'dark' ? 'text-gray-100' : 'text-gray-900'
                  }`}
                >
                  {(call.memory_used / 1024 / 1024).toFixed(2)} MB
                </div>
              </div>
            </div>
          )}
        </div>

        {/* User Interactions */}
        {call.user_interactions && call.user_interactions.length > 0 && (
          <CallDetailsSection
            title={
              <div className='flex items-center gap-2'>
                <User
                  className={`w-4 h-4 ${
                    theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
                  }`}
                />
                <span
                  className={
                    theme === 'dark' ? 'text-gray-100' : 'text-gray-900'
                  }
                >
                  Conversation
                </span>
              </div>
            }
          >
            <div className='space-y-2'>
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
          <CallDetailsSection
            title={
              <div className='flex items-center gap-2'>
                <ArrowDownCircle
                  className={`w-4 h-4 ${
                    theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
                  }`}
                />
                <span
                  className={
                    theme === 'dark' ? 'text-gray-100' : 'text-gray-900'
                  }
                >
                  Input
                </span>
              </div>
            }
          >
            <div
              className={`${theme === 'dark' ? 'bg-gray-900' : 'bg-gray-800'} 
              ${
                theme === 'dark'
                  ? 'border border-gray-700'
                  : 'border border-gray-600'
              }
              text-gray-100 rounded-md p-3 text-xs overflow-auto`}
            >
              {call.input_prompt && (
                <div className='mb-2'>
                  <div className='text-gray-400 mb-1'>Prompt:</div>
                  <pre className='whitespace-pre-wrap'>
                    {typeof call.input_prompt === 'string'
                      ? call.input_prompt
                      : JSON.stringify(call.input_prompt, null, 2)}
                  </pre>
                </div>
              )}
              {call.input_parameters && (
                <div>
                  <div className='text-gray-400 mb-1'>Parameters:</div>
                  <pre className='whitespace-pre-wrap'>
                    {JSON.stringify(
                      utils.parseJSON(call.input_parameters),
                      null,
                      2
                    )}
                  </pre>
                </div>
              )}
            </div>
          </CallDetailsSection>
        )}

        {call.output && (
          <CallDetailsSection
            title={
              <div className='flex items-center gap-2'>
                <ArrowUpCircle
                  className={`w-4 h-4 ${
                    theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
                  }`}
                />
                <span
                  className={
                    theme === 'dark' ? 'text-gray-100' : 'text-gray-900'
                  }
                >
                  Output
                </span>
              </div>
            }
          >
            <div
              className={`${theme === 'dark' ? 'bg-gray-900' : 'bg-gray-800'} 
              ${
                theme === 'dark'
                  ? 'border border-gray-700'
                  : 'border border-gray-600'
              }
              text-gray-100 rounded-md p-3 text-xs overflow-auto`}
            >
              <pre className='whitespace-pre-wrap'>
                {typeof call.output === 'string'
                  ? call.output
                  : JSON.stringify(call.output, null, 2)}
              </pre>
            </div>
          </CallDetailsSection>
        )}

        {/* Token Usage */}
        {call.token_usage && (
          <CallDetailsSection
            title={
              <div className='flex items-center gap-2'>
                <MessageSquare
                  className={`w-4 h-4 ${
                    theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
                  }`}
                />
                <span
                  className={
                    theme === 'dark' ? 'text-gray-100' : 'text-gray-900'
                  }
                >
                  Token Usage
                </span>
              </div>
            }
          >
            <div className='grid grid-cols-3 gap-2'>
              {Object.entries(utils.parseJSON(call.token_usage) || {}).map(
                ([key, value]) => (
                  <div
                    key={key}
                    className={`${
                      theme === 'dark' ? 'bg-gray-800' : 'bg-white'
                    } 
                  p-2 rounded 
                  ${
                    theme === 'dark'
                      ? 'border border-gray-700'
                      : 'border border-gray-200'
                  }`}
                  >
                    <div
                      className={`text-xs ${
                        theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                      }`}
                    >
                      {key}
                    </div>
                    <div
                      className={`text-sm font-medium ${
                        theme === 'dark' ? 'text-gray-100' : 'text-gray-900'
                      }`}
                    >
                      {String(value)}
                    </div>
                  </div>
                )
              )}
            </div>
          </CallDetailsSection>
        )}

        {/* Cost */}
        {call.cost && (
          <CallDetailsSection
            title={
              <div className='flex items-center gap-2'>
                <CircleDollarSign
                  className={`w-4 h-4 ${
                    theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
                  }`}
                />
                <span
                  className={
                    theme === 'dark' ? 'text-gray-100' : 'text-gray-900'
                  }
                >
                  Cost
                </span>
              </div>
            }
          >
            <div className='grid grid-cols-3 gap-2'>
              {Object.entries(utils.parseJSON(call.cost) || {}).map(
                ([key, value]) => (
                  <div
                    key={key}
                    className={`${
                      theme === 'dark' ? 'bg-gray-800' : 'bg-white'
                    } 
                  p-2 rounded 
                  ${
                    theme === 'dark'
                      ? 'border border-gray-700'
                      : 'border border-gray-200'
                  }`}
                  >
                    <div
                      className={`text-xs ${
                        theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                      }`}
                    >
                      {key}
                    </div>
                    <div
                      className={`text-sm font-medium ${
                        theme === 'dark' ? 'text-gray-100' : 'text-gray-900'
                      }`}
                    >
                      ${Number(value).toFixed(6)}
                    </div>
                  </div>
                )
              )}
            </div>
          </CallDetailsSection>
        )}

        {/* Errors */}
        {renderErrors(call.errors)}
      </div>
    )
  }

  const getTypeIcon = type => {
    const { theme } = useTheme()

    const getColors = (lightColor, lightBg) => {
      if (theme === 'dark') {
        // Convert light colors to dark theme equivalents
        const darkColor = lightColor.replace('600', '400')
        const darkBg = lightBg.replace('100', '900/30')
        return { color: darkColor, bgColor: darkBg }
      }
      return { color: lightColor, bgColor: lightBg }
    }

    switch (type) {
      case 'agent': {
        const { color, bgColor } = getColors('text-violet-600', 'bg-violet-100')
        return { icon: Bot, color, bgColor }
      }
      case 'llm': {
        const { color, bgColor } = getColors('text-green-600', 'bg-green-100')
        return { icon: MessageSquare, color, bgColor }
      }
      case 'tool': {
        const { color, bgColor } = getColors('text-blue-600', 'bg-blue-100')
        return { icon: Wrench, color, bgColor }
      }
      default: {
        const { color, bgColor } = getColors('text-gray-600', 'bg-gray-100')
        return { icon: Code, color, bgColor }
      }
    }
  }

  const renderCall = call => {
    const { theme } = useTheme()
    const { icon: Icon, color, bgColor } = getTypeIcon(call.type)
    const callId = `${call.type}-${call.id}`
    const isExpanded = expandedSections.has(callId)
    const isSelected = selectedSection?.id === call.id

    const getSelectedStyles = () => {
      if (theme === 'dark') {
        return isSelected
          ? 'border-blue-500 bg-blue-900/30'
          : 'border-gray-700 bg-gray-800'
      }
      return isSelected
        ? 'border-blue-300 bg-blue-50'
        : 'border-gray-200 bg-white'
    }

    const getHoverStyle = () => {
      return theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-50'
    }

    // Adjust type badge colors for dark theme
    const getTypeBadgeStyles = () => {
      const darkThemeColors = {
        bgColor: bgColor.replace('50', '900'),
        color: color.replace('600', '400')
      }

      return theme === 'dark'
        ? `${darkThemeColors.bgColor} ${darkThemeColors.color}`
        : `${bgColor} ${color}`
    }

    return (
      <div key={callId} style={{ marginLeft: `${call.depth * 16}px` }}>
        <div
          className={`
            rounded-lg border transition-all duration-200 mb-2
            ${getSelectedStyles()}
          `}
        >
          <div
            className={`flex items-center gap-2 p-3 cursor-pointer ${getHoverStyle()}`}
            onClick={() => {
              setSelectedSection(isSelected ? null : call)
              setExpandedSections(prev => {
                const next = new Set(prev)
                if (next.has(callId)) {
                  next.delete(callId)
                } else {
                  next.add(callId)
                }
                return next
              })
            }}
          >
            <Icon
              className={`w-4 h-4 ${
                theme === 'dark' ? color.replace('600', '400') : color
              }`}
            />
            <div className='flex-1'>
              <div className='flex items-center justify-between'>
                <div className='flex items-center gap-2'>
                  <span
                    className={`font-medium ${
                      theme === 'dark' ? 'text-gray-100' : 'text-gray-900'
                    }`}
                  >
                    {call.name ||
                      `${
                        call.type.charAt(0).toUpperCase() + call.type.slice(1)
                      } ${call.id}`}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded-full text-xs ${getTypeBadgeStyles()}`}
                  >
                    {call.type}
                  </span>
                </div>
                <span
                  className={`text-sm ${
                    theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                  }`}
                >
                  {utils.formatDuration(call.duration)}
                </span>
              </div>
              <div
                className={`text-xs ${
                  theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                } mt-1 flex items-center gap-2`}
              >
                <Clock className='w-3 h-3' />
                {utils.formatTime(call.startTime)}
              </div>
            </div>
          </div>

          {isSelected && renderCallDetails(call)}
        </div>
      </div>
    )
  }

  return (
    <div
      className={`fixed inset-y-0 right-0 w-[600px] transform transition-transform
        ${isOpen ? 'translate-x-0' : 'translate-x-full'}
        ${
          theme === 'dark'
            ? 'bg-gray-900 border-l border-gray-700'
            : 'bg-white border-l border-gray-200'
        }`}
    >
      <div className='h-full flex flex-col'>
        {/* Header */}
        <div
          className={`p-4 flex items-center justify-between border-b ${
            theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
          }`}
        >
          <div>
            <h2
              className={`text-lg font-semibold flex items-center gap-2 ${
                theme === 'dark' ? 'text-gray-100' : 'text-gray-900'
              }`}
            >
              <Bot className='w-5 h-5 text-purple-600' />
              Trace Details
            </h2>
            {traceData && (
              <div className='text-sm text-gray-500 mt-1'>
                ID: {traceData.id}
              </div>
            )}
          </div>
          <button
            onClick={onClose}
            className={`p-2 rounded-full ${
              theme === 'dark' ? 'hover:bg-gray-800' : 'hover:bg-gray-100'
            }`}
          >
            <X
              className={`w-5 h-5 ${
                theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
              }`}
            />
          </button>
        </div>

        {traceData && (
          <>
            {/* Summary Section */}
            <div
              className={`p-4 space-y-4 border-b ${
                theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
              }`}
            >
              <div className='grid grid-cols-2 gap-4'>
                <div
                  className={`flex items-center gap-2 p-3 rounded-lg border ${
                    theme === 'dark'
                      ? 'bg-gray-800 border-gray-700'
                      : 'bg-white border-gray-200'
                  }`}
                >
                  <Clock className='w-5 h-5 text-blue-600' />
                  <div>
                    <div
                      className={`text-sm ${
                        theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                      }`}
                    >
                      Duration
                    </div>
                    <div
                      className={`font-medium ${
                        theme === 'dark' ? 'text-gray-100' : 'text-gray-900'
                      }`}
                    >
                      {utils.formatDuration(traceData.duration)}
                    </div>
                  </div>
                </div>
                <div
                  className={`flex items-center gap-2 p-3 rounded-lg border ${
                    theme === 'dark'
                      ? 'bg-gray-800 border-gray-700'
                      : 'bg-white border-gray-200'
                  }`}
                >
                  <AlertCircle className='w-5 h-5 text-red-600' />
                  <div>
                    <div
                      className={`text-sm ${
                        theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                      }`}
                    >
                      Errors
                    </div>
                    <div
                      className={`font-medium ${
                        theme === 'dark' ? 'text-gray-100' : 'text-gray-900'
                      }`}
                    >
                      {Array.isArray(traceData.errors)
                        ? traceData.errors.length
                        : 0}
                    </div>
                  </div>
                </div>
              </div>

              {/* Timeline */}
              <div
                className={`p-4 rounded-lg ${
                  theme === 'dark' ? 'bg-gray-800' : 'bg-gray-50'
                }`}
              >
                <div className='flex items-center justify-between mb-2'>
                  <div
                    className={`text-sm font-medium ${
                      theme === 'dark' ? 'text-gray-100' : 'text-gray-900'
                    }`}
                  >
                    Timeline
                  </div>
                  <div
                    className={`text-xs ${
                      theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                    }`}
                  >
                    {utils.formatTime(timelineStart)} -{' '}
                    {utils.formatTime(timelineStart + timelineDuration * 1000)}
                  </div>
                </div>

                <div
                  className={`relative rounded overflow-hidden ${
                    theme === 'dark' ? 'bg-gray-700' : 'bg-gray-100'
                  }`}
                  style={{
                    height: `${
                      Math.max(...chronologicalCalls.map(c => c.depth + 1)) * 28
                    }px`,
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
                <div
                  className={`mt-2 flex flex-wrap gap-3 text-xs ${
                    theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                  }`}
                >
                  <div className='flex items-center gap-1'>
                    <div className='w-3 h-3 rounded bg-violet-600' />
                    Agent Calls
                  </div>
                  <div className='flex items-center gap-1'>
                    <div className='w-3 h-3 rounded bg-green-600' />
                    LLM Calls
                  </div>
                  <div className='flex items-center gap-1'>
                    <div className='w-3 h-3 rounded bg-blue-600' />
                    Tool Calls
                  </div>
                </div>
              </div>
            </div>

            {/* Call List */}
            <div
              className={`flex-1 overflow-y-auto p-4 ${
                theme === 'dark' ? 'bg-gray-900' : 'bg-white'
              }`}
            >
              <div className='space-y-1'>
                {chronologicalCalls.map((call, index) => (
                  <React.Fragment key={`${call.type}-${call.id}`}>
                    {index === 0 ||
                    utils.formatTime(call.startTime).split(':')[1] !==
                      utils
                        .formatTime(chronologicalCalls[index - 1].startTime)
                        .split(':')[1] ? (
                      <div
                        className={`text-xs sticky top-0 py-1 ${
                          theme === 'dark'
                            ? 'text-gray-400 bg-gray-900'
                            : 'text-gray-400 bg-white'
                        }`}
                      >
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
  )
}

export default TraceDetailsPanel
