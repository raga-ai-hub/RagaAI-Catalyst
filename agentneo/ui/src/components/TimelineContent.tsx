import React from 'react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { TimelineData } from '../types/timeline';

interface TimelineContentProps {
  timelineRef: React.RefObject<HTMLDivElement>;
  zoom: number;
  rows: string[];
  timelineData: TimelineData[];
  totalDuration: number;
  currentTime: number;
  handleEventClick: (event: TimelineData) => void;
}

const TimelineContent: React.FC<TimelineContentProps> = ({
  timelineRef,
  zoom,
  rows,
  timelineData,
  totalDuration,
  currentTime,
  handleEventClick
}) => {
  const renderTimeTicks = () => {
    const ticks = [];
    const tickCount = 8;
    const tickInterval = totalDuration / tickCount;
    for (let i = 0; i <= tickCount; i++) {
      const time = i * tickInterval;
      ticks.push(
        <div key={i} className="absolute text-xs text-gray-500" style={{ left: `${(time / totalDuration) * 100}%` }}>
          <div className="h-2 border-l border-gray-300"></div>
          <div>{time.toFixed(2)}s</div>
        </div>
      );
    }
    return ticks;
  };

  const getEventColor = (type: string) => {
    switch (type) {
      case 'LLM': return '#34A853';
      case 'Tool': return '#FBBC05';
      case 'Action': return '#4285F4';
      case 'Error': return '#EA4335';
      case 'UserInteraction': return '#9C27B0';
      default: return '#757575';
    }
  };

  const startTime = timelineData.length > 0 ? new Date(timelineData[0].startTime).getTime() : 0;

  return (
    <div className="w-5/6">
      <div className="relative" style={{ height: `${rows.length * 48 + 50}px`, overflowX: 'auto', overflowY: 'hidden' }} ref={timelineRef}>
        <div style={{ width: `${zoom}%`, minWidth: '100%', height: '100%', position: 'relative' }}>
          {timelineData.map((event, index) => {
            const eventStart = (new Date(event.startTime).getTime() - startTime) / 1000;
            const eventDuration = (new Date(event.endTime).getTime() - new Date(event.startTime).getTime()) / 1000;
            return (
              <TooltipProvider key={index}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div
                      className="absolute flex items-center cursor-pointer"
                      style={{
                        left: `${(eventStart / totalDuration) * 100}%`,
                        width: `${(eventDuration / totalDuration) * 100}%`,
                        top: `${rows.indexOf(event.type) * 48}px`,
                        height: '40px',
                      }}
                      onClick={() => handleEventClick(event)}
                    >
                      <div className="w-full h-full rounded-sm" style={{ backgroundColor: getEventColor(event.type) }}></div>
                      <div className="absolute left-2 text-xs text-white truncate">{event.name}</div>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>{event.name}</p>
                    <p>Start: {eventStart.toFixed(2)}s</p>
                    <p>Duration: {eventDuration.toFixed(2)}s</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            );
          })}
          <div
            className="absolute top-0 w-px h-full bg-red-500"
            style={{ left: `${(currentTime / totalDuration) * 100}%` }}
          ></div>
          <div className="absolute bottom-0 left-0 right-0 h-10 border-t border-gray-300">
            {renderTimeTicks()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TimelineContent;