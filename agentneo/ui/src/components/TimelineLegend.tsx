import React from 'react';
import { TimelineData } from '../types/timeline';

interface TimelineLegendProps {
  timelineData: TimelineData[];
}

const TimelineLegend: React.FC<TimelineLegendProps> = ({ timelineData }) => {
  const uniqueTypes = Array.from(new Set(timelineData.map(d => d.type)));

  return (
    <div className="flex mb-4 space-x-4">
      {uniqueTypes.map(type => (
        <div key={type} className="flex items-center">
          <div 
            className="w-4 h-4 mr-1 rounded-sm" 
            style={{ backgroundColor: timelineData.find(d => d.type === type)?.color }}
          ></div>
          <span>{type}</span>
        </div>
      ))}
    </div>
  );
};

export default TimelineLegend;