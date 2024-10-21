import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const data = [
  { name: 'Web Search', 'Success Rate': 4, 'Response Time': 3, 'Cost Efficiency': 2, 'Usage Frequency': 1 },
  { name: 'Data Analysis', 'Success Rate': 3, 'Response Time': 2, 'Cost Efficiency': 4, 'Usage Frequency': 3 },
  { name: 'Code Interpreter', 'Success Rate': 2, 'Response Time': 4, 'Cost Efficiency': 3, 'Usage Frequency': 2 },
  { name: 'Image Generation', 'Success Rate': 1, 'Response Time': 1, 'Cost Efficiency': 1, 'Usage Frequency': 4 },
];

const metrics = ['Success Rate', 'Response Time', 'Cost Efficiency', 'Usage Frequency'];
const colors = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444'];

const ToolPerformanceAnalysis: React.FC = () => {
  return (
    <Card className="bg-white">
      <CardHeader>
        <CardTitle>Tool Performance Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            {metrics.map((metric, index) => (
              <Line 
                key={metric} 
                type="monotone" 
                dataKey={metric} 
                stroke={colors[index]} 
                activeDot={{ r: 8 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
        <div className="mt-4 flex flex-wrap justify-between text-xs">
          {metrics.map((metric, index) => (
            <span key={metric} className="flex items-center mb-2 mr-2">
              <div className={`w-3 h-3 mr-1`} style={{backgroundColor: colors[index]}}></div>
              {metric}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default ToolPerformanceAnalysis;