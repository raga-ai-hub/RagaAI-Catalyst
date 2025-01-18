import React from 'react';
import { Panel } from 'reactflow';
import { nodeStylesByType } from './ExecutionGraphNodes';

export const Legend = () => {
  return (
    <Panel position="top-left" className="bg-white p-2 rounded shadow-md m-2">
      <div className="text-sm font-semibold mb-2">Node Types</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        {Object.entries(nodeStylesByType).map(([type, style]) => (
          <div key={type} className="flex items-center">
            <div
              className={`w-3 h-3 rounded mr-2 ${style.className}`}
              style={{ backgroundColor: style.backgroundColor }}
            />
            <span className="capitalize text-xs">{type}</span>
          </div>
        ))}
      </div>
    </Panel>
  );
};