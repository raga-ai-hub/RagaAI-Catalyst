import React from 'react';
import { nodeStylesByType } from './ExecutionGraphNodes';

export const Legend = () => {
  const legendItems = [
    { label: 'Agent', color: nodeStylesByType.agent.backgroundColor, borderColor: nodeStylesByType.agent.borderColor },
    { label: 'LLM', color: nodeStylesByType.llm.backgroundColor, borderColor: nodeStylesByType.llm.borderColor },
    { label: 'Tool', color: nodeStylesByType.tool.backgroundColor, borderColor: nodeStylesByType.tool.borderColor },
    { label: 'User Interaction', color: nodeStylesByType.user_interaction.backgroundColor, borderColor: nodeStylesByType.user_interaction.borderColor },
    { label: 'Error', color: nodeStylesByType.error.backgroundColor, borderColor: nodeStylesByType.error.borderColor },
  ];

  return (
    <div className="absolute top-4 right-4 bg-white p-2 border rounded shadow-md z-10">
      {legendItems.map((item) => (
        <div key={item.label} className="flex items-center mb-1">
          <div
            className="w-4 h-4 mr-2 rounded"
            style={{ backgroundColor: item.color, borderColor: item.borderColor, borderWidth: '2px' }}
          ></div>
          <span className="text-sm">{item.label}</span>
        </div>
      ))}
    </div>
  );
};