import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface TraceInsightsChartProps {
  data: {
    name: string;
    count: number;
    avgDuration: number;
  }[];
}

const TraceInsightsChart: React.FC<TraceInsightsChartProps> = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <XAxis dataKey="name" />
        <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
        <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
        <Tooltip />
        <Legend />
        <Bar yAxisId="left" dataKey="count" fill="#8884d8" name="Count" />
        <Bar yAxisId="right" dataKey="avgDuration" fill="#82ca9d" name="Avg Duration (ms)" />
      </BarChart>
    </ResponsiveContainer>
  );
};

export default TraceInsightsChart;