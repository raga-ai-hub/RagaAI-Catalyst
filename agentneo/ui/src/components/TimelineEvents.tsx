import React from 'react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface TimelineData {
  name: string;
  start: number;
  duration: number;
  color: string;
  type: string;
  row: string;
  details?: {
    agent?: string;
    function?: string;
    input?: string;
    output?: string;
  };
}

interface TimelineEventsProps {
  timelineData: TimelineData[];
  totalDuration: number;
  rows: string[];
  handleEventClick: (event: TimelineData) => void;
}

const TimelineEvents: React.FC<TimelineEventsProps> = ({
  timelineData,
  totalDuration,
  rows,
  handleEventClick,
}) => {
  return (
    <>
      {timelineData.map((event, index) => (
        <TooltipProvider key={index}>
          <Tooltip>
            <TooltipTrigger asChild>
              <div
                className="absolute flex items-center cursor-pointer"
                style={{
                  left: `${(event.start / totalDuration) * 100}%`,
                  width: `${(event.duration / totalDuration) * 100}%`,
                  top: `${rows.indexOf(event.row) * 48}px`,
                  height: '40px',
                }}
                onClick={() => handleEventClick(event)}
              >
                <div className="w-full h-full rounded-sm" style={{ backgroundColor: event.color }}></div>
                <div className="absolute left-2 text-xs text-white truncate">{event.name}</div>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>{event.name}</p>
              <p>Start: {event.start}s</p>
              <p>Duration: {event.duration}s</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      ))}
    </>
  );
};

export default TimelineEvents;