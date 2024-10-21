import React from 'react';
import { Card, CardContent } from "@/components/ui/card";

const tools = [
  { name: 'CHAIN', color: '#8884d8' },
  { name: 'TOOL', color: '#82ca9d' },
  { name: 'LLM', color: '#ffc658' },
];

const ToolLegend: React.FC = () => {
  return (
    <Card className="mb-6">
      <CardContent className="flex justify-center items-center space-x-4 py-4">
        {tools.map((tool) => (
          <div key={tool.name} className="flex items-center">
            <div className="w-4 h-4 mr-2" style={{ backgroundColor: tool.color }}></div>
            <span>{tool.name}</span>
          </div>
        ))}
      </CardContent>
    </Card>
  );
};

export default ToolLegend;