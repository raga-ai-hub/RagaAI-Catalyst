import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import TimelineControls from './TimelineControls';
import TimelineDetails from './TimelineDetails';
import TimelineLegend from './TimelineLegend';
import TimelineRows from './TimelineRows';
import TimelineContent from './TimelineContent';
import { TimelineData } from '../types/timeline';
import { useProject } from '../contexts/ProjectContext';
import { fetchTraceDetails } from '../utils/api';

const ExecutionTimeline: React.FC = () => {
  const { selectedTraceId } = useProject();
  const [timelineData, setTimelineData] = useState<TimelineData[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<TimelineData | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [zoom, setZoom] = useState(100);
  const timelineRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const loadTimelineData = async () => {
      if (selectedTraceId) {
        try {
          const traceData = await fetchTraceDetails(selectedTraceId);
          const convertedData: TimelineData[] = convertTraceToTimelineData(traceData);
          setTimelineData(convertedData);
        } catch (error) {
          console.error('Error loading timeline data:', error);
        }
      }
    };

    loadTimelineData();
  }, [selectedTraceId]);

  const convertTraceToTimelineData = (traceData: any): TimelineData[] => {
    const timelineEvents: TimelineData[] = [];

    // Add main trace
    timelineEvents.push({
      name: traceData.name || `Trace ${traceData.id}`,
      startTime: traceData.start_time,
      endTime: traceData.end_time,
      duration: (new Date(traceData.end_time).getTime() - new Date(traceData.start_time).getTime()) / 1000,
      color: '#4285F4',
      type: 'Trace',
      row: 'Main',
      details: {
        content: traceData.description || 'Main trace execution',
        parentName: 'Standalone Trace'
      },
      counts: {
        llms: traceData.llm_calls?.length || 0,
        tools: traceData.tool_calls?.length || 0,
        interactions: 0,
        errors: 0,
        agents: traceData.agent_calls?.length || 0, // Add agent count
      }
    });

    // Add standalone LLM calls
    traceData.llm_calls?.forEach((llm: any) => {
      timelineEvents.push({
        name: llm.name || 'LLM Call',
        startTime: llm.start_time,
        endTime: llm.end_time,
        duration: (new Date(llm.end_time).getTime() - new Date(llm.start_time).getTime()) / 1000,
        color: '#34A853',
        type: 'LLM',
        row: 'LLM',
        details: {
          model: llm.model,
          input: llm.input_prompt,
          output: llm.output,
          parentName: 'Standalone LLM Call'
        }
      });
    });

    // Add standalone tool calls
    traceData.tool_calls?.forEach((tool: any) => {
      const duration = (new Date(tool.end_time).getTime() - new Date(tool.start_time).getTime()) / 1000;
      timelineEvents.push({
        name: tool.name,
        startTime: tool.start_time,
        endTime: tool.end_time,
        duration: duration,
        color: '#FBBC05',
        type: 'Tool',
        row: 'Tool', // Ensure it stays in the 'Tool' row
        isDot: duration < 0.1, // Keep this logic for visual representation
        details: {
          name: tool.name,
          input: tool.input_parameters,
          output: tool.output,
          parentName: 'Standalone Tool Call'
        }
      });
    });

    // Process agent calls
    traceData.agent_calls?.forEach((agent: any) => {
      const agentCounts = {
        llms: agent.llm_calls?.length || 0,
        tools: agent.tool_calls?.length || 0,
        interactions: agent.user_interactions?.length || 0,
        errors: agent.errors?.length || 0,
        agents: 0, // No nested agents in this context
      };

      // Add agent
      timelineEvents.push({
        name: agent.name,
        startTime: agent.start_time,
        endTime: agent.end_time,
        duration: (new Date(agent.end_time).getTime() - new Date(agent.start_time).getTime()) / 1000,
        color: '#4285F4',
        type: 'Agent',
        row: 'Agent',
        details: {
          agent: agent.name,
          content: agent.description || 'Agent execution',
          parentName: 'Standalone Agent'
        },
        counts: agentCounts
      });

      // Add LLM calls
      agent.llm_calls?.forEach((llm: any) => {
        timelineEvents.push({
          name: llm.name || 'LLM Call',
          startTime: llm.start_time,
          endTime: llm.end_time,
          duration: (new Date(llm.end_time).getTime() - new Date(llm.start_time).getTime()) / 1000,
          color: '#34A853',
          type: 'LLM',
          row: 'LLM',
          details: {
            model: llm.model,
            input: llm.input_prompt,
            output: llm.output,
            parentName: agent.name
          }
        });
      });

      // Add tool calls
      agent.tool_calls?.forEach((tool: any) => {
        const duration = (new Date(tool.end_time).getTime() - new Date(tool.start_time).getTime()) / 1000;
        timelineEvents.push({
          name: tool.name,
          startTime: tool.start_time,
          endTime: tool.end_time,
          duration: duration,
          color: '#FBBC05',
          type: 'Tool',
          row: 'Tool', // Ensure it stays in the 'Tool' row
          isDot: duration < 0.1,
          details: {
            name: tool.name,
            input: tool.input_parameters,
            output: tool.output,
            parentName: agent.name
          }
        });
      });

      // Add user interactions
      agent.user_interactions?.forEach((interaction: any) => {
        timelineEvents.push({
          name: '', // Empty name for dot representation
          startTime: interaction.timestamp,
          endTime: interaction.timestamp,
          duration: 0,
          color: '#9C27B0',
          type: 'Interaction',
          row: 'Interaction', // Ensure it stays in the 'Interaction' row
          isDot: true,
          details: {
            name: interaction.interaction_type,
            interaction_type: interaction.interaction_type,
            content: interaction.content,
            parentName: agent.name
          }
        });
      });

      // Add errors
      agent.errors?.forEach((error: any) => {
        timelineEvents.push({
          name: error.error_type,
          startTime: error.timestamp,
          endTime: error.timestamp,
          duration: 0,
          color: '#EA4335',
          type: 'Error',
          row: 'Error', // Ensure it stays in the 'Error' row
          details: {
            error_message: error.message,
            parentName: agent.name
          }
        });
      });
    });

    return timelineEvents;
  };

  const totalDuration = Math.max(...(timelineData.map(event => 
    (new Date(event.endTime).getTime() - new Date(timelineData[0].startTime).getTime()) / 1000
  ) || [0]));
  
  const rows = ['Main', 'Agent', 'LLM', 'Tool', 'Interaction', 'Error'];

  const handleEventClick = (event: TimelineData) => {
    setSelectedEvent(event);
    const startTimeMs = new Date(event.startTime).getTime();
    const firstEventMs = new Date(timelineData[0]?.startTime || 0).getTime();
    setCurrentTime((startTimeMs - firstEventMs) / 1000);
  };

  const getOverlappingEvents = (currentEvent: TimelineData) => {
    return timelineData.filter(event => {
      const currentStart = new Date(currentEvent.startTime).getTime();
      const currentEnd = new Date(currentEvent.endTime).getTime();
      const eventStart = new Date(event.startTime).getTime();
      const eventEnd = new Date(event.endTime).getTime();
      return (eventStart < currentEnd && eventEnd > currentStart);
    });
  };

  useEffect(() => {
    if (timelineRef.current) {
      const timelineWidth = timelineRef.current.offsetWidth;
      const scrollPosition = (currentTime / totalDuration) * timelineWidth * (zoom / 100) - timelineWidth / 2;
      timelineRef.current.scrollLeft = scrollPosition;
    }
  }, [currentTime, zoom, totalDuration]);

  return (
    <Card className="mt-8 bg-white shadow-md">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-2xl font-bold">Execution Timeline</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex">
          <div className="w-3/4 pr-4">
            <TimelineControls
              zoom={zoom}
              setZoom={setZoom}
              currentTime={currentTime}
              setCurrentTime={setCurrentTime}
              totalDuration={totalDuration}
            />
            <TimelineLegend timelineData={timelineData} />
            <div className="flex">
              <TimelineRows rows={rows} />
              <TimelineContent
                timelineRef={timelineRef}
                zoom={zoom}
                rows={rows}
                timelineData={timelineData}
                totalDuration={totalDuration}
                currentTime={currentTime}
                handleEventClick={handleEventClick}
              />
            </div>
          </div>
          <div className="w-1/4 pl-4 border-l border-gray-200">
            <TimelineDetails 
              selectedEvent={selectedEvent} 
              overlappingEvents={selectedEvent ? getOverlappingEvents(selectedEvent) : []} 
              counts={selectedEvent?.counts || { llms: 0, tools: 0, interactions: 0, errors: 0, agents: 0 }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ExecutionTimeline;