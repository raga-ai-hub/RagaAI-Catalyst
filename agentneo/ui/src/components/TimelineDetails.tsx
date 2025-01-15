import React from 'react'
import { TimelineData } from '../types/timeline'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Clock,
  Calendar,
  Bot,
  Cpu,
  Wrench,
  AlertCircle,
  MessageSquare,
  Hash
} from 'lucide-react'

interface TimelineDetailsProps {
  selectedEvent: TimelineData | null
  overlappingEvents: TimelineData[]
  counts: {
    llms: number
    tools: number
    interactions: number
    errors: number
  }
}

const getEventIcon = (type: string) => {
  switch (type.toLowerCase()) {
    case 'trace':
      return <Bot className='w-5 h-5' />
    case 'agent':
      return <Bot className='w-5 h-5' />
    case 'llm':
      return <Cpu className='w-5 h-5' />
    case 'tool':
      return <Wrench className='w-5 h-5' />
    case 'error':
      return <AlertCircle className='w-5 h-5' />
    case 'interaction':
      return <MessageSquare className='w-5 h-5' />
    default:
      return <Hash className='w-5 h-5' />
  }
}

const getEventColor = (type: string) => {
  switch (type.toLowerCase()) {
    case 'trace':
      return 'bg-blue-500/10 text-blue-700 dark:bg-blue-500/20 dark:text-blue-300'
    case 'agent':
      return 'bg-blue-500/10 text-blue-700 dark:bg-blue-500/20 dark:text-blue-300'
    case 'llm':
      return 'bg-green-500/10 text-green-700 dark:bg-green-500/20 dark:text-green-300'
    case 'tool':
      return 'bg-yellow-500/10 text-yellow-700 dark:bg-yellow-500/20 dark:text-yellow-300'
    case 'error':
      return 'bg-red-500/10 text-red-700 dark:bg-red-500/20 dark:text-red-300'
    case 'interaction':
      return 'bg-purple-500/10 text-purple-700 dark:bg-purple-500/20 dark:text-purple-300'
    default:
      return 'bg-gray-500/10 text-gray-700 dark:bg-gray-500/20 dark:text-gray-300'
  }
}

const TimelineDetails: React.FC<TimelineDetailsProps> = ({
  selectedEvent,
  overlappingEvents,
  counts
}) => {
  if (!selectedEvent) {
    return (
      <div className='h-full flex items-center justify-center text-muted-foreground italic'>
        Select an event to view details
      </div>
    )
  }

  const renderEventDetails = (event: TimelineData) => {
    const startTime = new Date(event.startTime)
    const endTime = new Date(event.endTime)
    const duration = (endTime.getTime() - startTime.getTime()) / 1000

    return (
      <div key={event.name} className='space-y-6'>
        {/* Header Section */}
        <div className='space-y-3'>
          <div className='flex items-center space-x-2'>
            {getEventIcon(event.type)}
            <h3 className='text-lg font-semibold text-foreground'>
              {event.name}
            </h3>
          </div>
          <Badge variant='secondary' className={`${getEventColor(event.type)}`}>
            {event.type}
          </Badge>
        </div>

        {/* Timing Section */}
        <div className='space-y-3 p-3 rounded-lg border bg-card text-card-foreground'>
          <div className='flex items-center space-x-2'>
            <Calendar className='w-4 h-4' />
            <span className='text-sm'>Start: {startTime.toLocaleString()}</span>
          </div>
          <div className='flex items-center space-x-2'>
            <Calendar className='w-4 h-4' />
            <span className='text-sm'>End: {endTime.toLocaleString()}</span>
          </div>
          <div className='flex items-center space-x-2'>
            <Clock className='w-4 h-4' />
            <span className='text-sm'>Duration: {duration.toFixed(2)}s</span>
          </div>
        </div>

        {/* Count Section for Trace and Agent */}
        {event.type.toLowerCase() === 'agent' && (
          <div className='space-y-3 p-3 rounded-lg border bg-muted'>
            <div className='flex items-center space-x-2'>
              <Cpu className='w-4 h-4' />
              <span className='text-sm'>LLMs: {counts.llms}</span>
            </div>
            <div className='flex items-center space-x-2'>
              <Wrench className='w-4 h-4' />
              <span className='text-sm'>Tools: {counts.tools}</span>
            </div>
            <div className='flex items-center space-x-2'>
              <MessageSquare className='w-4 h-4' />
              <span className='text-sm'>
                Interactions: {counts.interactions}
              </span>
            </div>
            <div className='flex items-center space-x-2'>
              <AlertCircle className='w-4 h-4' />
              <span className='text-sm'>Errors: {counts.errors}</span>
            </div>
          </div>
        )}

        {/* Details Section for LLM, Tool, and Interaction */}
        {event.details &&
          (event.type.toLowerCase() === 'llm' ||
            event.type.toLowerCase() === 'tool' ||
            event.type.toLowerCase() === 'interaction') && (
            <div className='space-y-4'>
              {/* Parent Name */}
              <div className='p-3 rounded-lg border bg-muted'>
                <div className='flex items-center space-x-2'>
                  <span className='text-sm'>
                    Parent: {event.details.parentName}
                  </span>
                </div>
              </div>

              {/* LLM Details */}
              {event.type.toLowerCase() === 'llm' && (
                <div className='p-3 rounded-lg border space-y-3 bg-green-500/5 dark:bg-green-500/10'>
                  <div className='space-y-2'>
                    {event.details.model && (
                      <div className='flex items-center space-x-2'>
                        <Cpu className='w-4 h-4' />
                        <span className='text-sm'>
                          Model: {event.details.model}
                        </span>
                      </div>
                    )}
                  </div>
                  {event.details.input && (
                    <div className='mt-3'>
                      <h4 className='font-medium mb-2'>Prompt</h4>
                      <div className='text-sm p-3 rounded-lg border bg-card/50 whitespace-pre-wrap'>
                        {event.details.input}
                      </div>
                    </div>
                  )}
                  {event.details.output && (
                    <div className='mt-3'>
                      <h4 className='font-medium mb-2'>Response</h4>
                      <div className='text-sm p-3 rounded-lg border bg-card/50 whitespace-pre-wrap'>
                        {event.details.output}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Tool Details */}
              {event.type.toLowerCase() === 'tool' && (
                <div className='p-3 rounded-lg border space-y-3 bg-yellow-500/5 dark:bg-yellow-500/10'>
                  {event.details.input && (
                    <div className='space-y-2'>
                      <h4 className='font-medium'>Input</h4>
                      <pre className='text-sm p-3 rounded-lg border bg-card/50 overflow-x-auto'>
                        {event.details.input}
                      </pre>
                    </div>
                  )}
                  {event.details.output && (
                    <div className='space-y-2'>
                      <h4 className='font-medium'>Output</h4>
                      <pre className='text-sm p-3 rounded-lg border bg-card/50 overflow-x-auto'>
                        {event.details.output}
                      </pre>
                    </div>
                  )}
                </div>
              )}

              {/* Interaction Details */}
              {event.type.toLowerCase() === 'interaction' &&
                event.details.content && (
                  <div className='p-3 rounded-lg border space-y-2 bg-purple-500/5 dark:bg-purple-500/10'>
                    <h4 className='font-medium'>Interaction Content</h4>
                    <div className='text-sm p-3 rounded-lg border bg-card/50 whitespace-pre-wrap'>
                      {event.details.content}
                    </div>
                  </div>
                )}
            </div>
          )}
      </div>
    )
  }

  return (
    <ScrollArea className='h-[calc(70vh-2rem)] pr-4'>
      {renderEventDetails(selectedEvent)}
    </ScrollArea>
  )
}

export default TimelineDetails
