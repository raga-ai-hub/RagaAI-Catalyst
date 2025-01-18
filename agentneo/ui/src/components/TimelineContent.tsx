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

  const startTime = timelineData.length > 0 ? new Date(timelineData[0].startTime).getTime() : 0;

  const getOverlappingEvents = (currentEvent: TimelineData, allEvents: TimelineData[]) => {
    return allEvents.filter(event => {
      const currentStart = new Date(currentEvent.startTime).getTime();
      const currentEnd = new Date(currentEvent.endTime).getTime();
      const eventStart = new Date(event.startTime).getTime();
      const eventEnd = new Date(event.endTime).getTime();
      return (eventStart < currentEnd && eventEnd > currentStart);
    });
  };

  // ... existing code ...

const renderEvent = (event: TimelineData, index: number, overlappingEvents: TimelineData[]) => {
  const eventStart = (new Date(event.startTime).getTime() - startTime) / 1000;
  const eventDuration = (new Date(event.endTime).getTime() - new Date(event.startTime).getTime()) / 1000;
  const borderRadius = 8; // Define the border radius
  const isDot = event.type === 'Interaction' || (event.type === 'Tool' && eventDuration < 0.1);

  return (
    <TooltipProvider key={index}>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={`absolute cursor-pointer flex items-center`}
            style={{
              left: `${(eventStart / totalDuration) * 100}%`,
              width: isDot ? `${borderRadius * 2}px` : `${(eventDuration / totalDuration) * 100}%`,
              top: `${rows.indexOf(event.row) * 48 + 20}px`,
              height: '40px', // Keep the height consistent
              transform: isDot ? 'translate(-50%, 30%)' : 'none',
              backgroundColor: event.color,
              opacity: 0.8, // Add transparency
              transition: 'all 0.3s ease', // Smooth transition for hover effect
              borderRadius: `${borderRadius}px`, // Rounded corners
              boxShadow: overlappingEvents.length > 1 ? '0 0 10px rgba(0, 0, 0, 0.5)' : 'none', // Shadow for overlapping
            }}
            onClick={() => handleEventClick(event)}
            onMouseEnter={(e) => {
              e.currentTarget.style.opacity = '1'; // Full opacity on hover
              e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.8)'; // Change color on hover
              const eventNameElement = e.currentTarget.querySelector('.event-name') as HTMLElement;
              if (eventNameElement) {
                eventNameElement.style.color = 'black'; // Change text color to black
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.opacity = '0.8'; // Revert opacity
              e.currentTarget.style.backgroundColor = event.color; // Revert color
              const eventNameElement = e.currentTarget.querySelector('.event-name') as HTMLElement;
              if (eventNameElement) {
                eventNameElement.style.color = 'white'; // Revert text color
              }
            }}
          >
            <div className="w-full h-full rounded-sm"></div>
            {!isDot && (
              <div className="absolute left-2 text-xs text-white truncate event-name">
                {event.name}
              </div>
            )}
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
};

// ... existing code ...

  return (
    <div className="w-5/6">
      <div 
        className="relative" 
        style={{ 
          height: `${rows.length * 48 + 50}px`, 
          overflowX: 'auto', 
          overflowY: 'hidden' 
        }} 
        ref={timelineRef}
      >
        <div style={{ 
          width: `${zoom}%`, 
          minWidth: '100%', 
          height: '100%', 
          position: 'relative' 
        }}>
          {timelineData.map((event, index) => {
            const overlappingEvents = getOverlappingEvents(event, timelineData);
            return renderEvent(event, index, overlappingEvents);
          })}
          
          {/* Current time indicator */}
          <div
            className="absolute top-0 w-px h-full bg-red-500"
            style={{ left: `${(currentTime / totalDuration) * 100}%` }}
          ></div>

          {/* Timeline ticks */}
          <div className="absolute bottom-0 left-0 right-0 h-10 border-t border-gray-300">
            {renderTimeTicks()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TimelineContent;