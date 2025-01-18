import React from 'react';
import { TimelineData } from '../types/timeline';
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { 
  Clock, 
  Calendar, 
  Bot, 
  Cpu, 
  Wrench, 
  AlertCircle, 
  MessageSquare, 
  Hash
} from 'lucide-react';

interface TimelineDetailsProps {
  selectedEvent: TimelineData | null;
  overlappingEvents: TimelineData[];
  counts: {
    llms: number;
    tools: number;
    interactions: number;
    errors: number;
  };
}

const getEventIcon = (type: string) => {
  switch (type.toLowerCase()) {
    case 'trace': return <Bot className="w-5 h-5" />;
    case 'agent': return <Bot className="w-5 h-5" />;
    case 'llm': return <Cpu className="w-5 h-5" />;
    case 'tool': return <Wrench className="w-5 h-5" />;
    case 'error': return <AlertCircle className="w-5 h-5" />;
    case 'interaction': return <MessageSquare className="w-5 h-5" />;
    default: return <Hash className="w-5 h-5" />;
  }
};

const getEventColor = (type: string) => {
  switch (type.toLowerCase()) {
    case 'trace': return 'bg-blue-100 text-blue-800';
    case 'agent': return 'bg-blue-100 text-blue-800';
    case 'llm': return 'bg-green-100 text-green-800';
    case 'tool': return 'bg-yellow-100 text-yellow-800';
    case 'error': return 'bg-red-100 text-red-800';
    case 'interaction': return 'bg-purple-100 text-purple-800';
    default: return 'bg-gray-100 text-gray-800';
  }
};

const TimelineDetails: React.FC<TimelineDetailsProps> = ({ selectedEvent, overlappingEvents, counts }) => {
  if (!selectedEvent) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500 italic">
        Select an event to view details
      </div>
    );
  }

  const renderEventDetails = (event: TimelineData) => {
    const startTime = new Date(event.startTime);
    const endTime = new Date(event.endTime);
    const duration = (endTime.getTime() - startTime.getTime()) / 1000;

    return (
      <div key={event.name} className="space-y-6">
        {/* Header Section */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            {getEventIcon(event.type)}
            <h3 className="text-lg font-semibold">{event.name}</h3>
          </div>
          <Badge variant="secondary" className={`${getEventColor(event.type)}`}>
            {event.type}
          </Badge>
        </div>

        {/* Timing Section */}
        <div className="space-y-3 bg-gray-50 p-3 rounded-lg">
          <div className="flex items-center space-x-2 text-gray-700">
            <Calendar className="w-4 h-4" />
            <span className="text-sm">Start: {startTime.toLocaleString()}</span>
          </div>
          <div className="flex items-center space-x-2 text-gray-700">
            <Calendar className="w-4 h-4" />
            <span className="text-sm">End: {endTime.toLocaleString()}</span>
          </div>
          <div className="flex items-center space-x-2 text-gray-700">
            <Clock className="w-4 h-4" />
            <span className="text-sm">Duration: {duration.toFixed(2)}s</span>
          </div>
        </div>

        {/* Count Section for Trace and Agent */}
        {(event.type.toLowerCase() === 'agent') && (
          <div className="space-y-3 bg-gray-50 p-3 rounded-lg">
            <div className="flex items-center space-x-2 text-gray-700">
              <Cpu className="w-4 h-4" />
              <span className="text-sm">LLMs: {counts.llms}</span>
            </div>
            <div className="flex items-center space-x-2 text-gray-700">
              <Wrench className="w-4 h-4" />
              <span className="text-sm">Tools: {counts.tools}</span>
            </div>
            <div className="flex items-center space-x-2 text-gray-700">
              <MessageSquare className="w-4 h-4" />
              <span className="text-sm">Interactions: {counts.interactions}</span>
            </div>
            <div className="flex items-center space-x-2 text-gray-700">
              <AlertCircle className="w-4 h-4" />
              <span className="text-sm">Errors: {counts.errors}</span>
            </div>
          </div>
        )}

        {/* Details Section for LLM, Tool, and Interaction */}
        {event.details && (event.type.toLowerCase() === 'llm' || event.type.toLowerCase() === 'tool' || event.type.toLowerCase() === 'interaction') && (
          <div className="space-y-4">
            {/* Parent Name */}
            <div className="bg-gray-50 p-3 rounded-lg">
              <div className="flex items-center space-x-2 text-gray-700">
                <span className="text-sm">Parent: {event.details.parentName}</span>
              </div>
            </div>

            {/* LLM Details */}
            {event.type.toLowerCase() === 'llm' && (
              <div className="bg-green-50 p-3 rounded-lg space-y-3">
                <div className="space-y-2">
                  {event.details.model && (
                    <div className="flex items-center space-x-2 text-green-700">
                      <Cpu className="w-4 h-4" />
                      <span className="text-sm">Model: {event.details.model}</span>
                    </div>
                  )}
                </div>
                {event.details.input && (
                  <div className="mt-3">
                    <h4 className="font-medium text-green-800 mb-2">Prompt</h4>
                    <div className="text-sm bg-white/50 p-3 rounded-lg whitespace-pre-wrap text-green-900">
                      {event.details.input}
                    </div>
                  </div>
                )}
                {event.details.output && (
                  <div className="mt-3">
                    <h4 className="font-medium text-green-800 mb-2">Response</h4>
                    <div className="text-sm bg-white/50 p-3 rounded-lg whitespace-pre-wrap text-green-900">
                      {event.details.output}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Tool Details */}
            {event.type.toLowerCase() === 'tool' && (
              <div className="bg-yellow-50 p-3 rounded-lg space-y-3">
                {event.details.input && (
                  <div className="space-y-2">
                    <h4 className="font-medium text-yellow-800">Input</h4>
                    <pre className="text-sm bg-white/50 p-3 rounded-lg overflow-x-auto text-yellow-900">
                      {event.details.input}
                    </pre>
                  </div>
                )}
                {event.details.output && (
                  <div className="space-y-2">
                    <h4 className="font-medium text-yellow-800">Output</h4>
                    <pre className="text-sm bg-white/50 p-3 rounded-lg overflow-x-auto text-yellow-900">
                      {event.details.output}
                    </pre>
                  </div>
                )}
              </div>
            )}

            {/* Interaction Details */}
            {event.type.toLowerCase() === 'interaction' && event.details.content && (
              <div className="bg-purple-50 p-3 rounded-lg space-y-2">
                <h4 className="font-medium text-purple-800">Interaction Content</h4>
                <div className="text-sm bg-white/50 p-3 rounded-lg whitespace-pre-wrap text-purple-900">
                  {event.details.content}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <ScrollArea className="h-[calc(70vh-2rem)] pr-4">
      {renderEventDetails(selectedEvent)}
    </ScrollArea>
  );
};

export default TimelineDetails;