import React from 'react';

interface TimelineRowsProps {
  rows: string[];
}

const TimelineRows: React.FC<TimelineRowsProps> = ({ rows }) => {
  return (
    <div className="w-1/6">
      {rows.map(row => (
        <div key={row} className="h-12 flex items-center text-sm text-gray-600">{row}</div>
      ))}
    </div>
  );
};

export default TimelineRows;