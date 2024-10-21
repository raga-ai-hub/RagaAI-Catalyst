import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface TraceAnalyticsProps {
  traceId: string;
}

const TraceAnalytics: React.FC<TraceAnalyticsProps> = ({ traceId }) => {
  // Dummy data for the chart
  const data = [
    { name: 'Start', llmCalls: 0, toolCalls: 0, agentCalls: 0 },
    { name: '25%', llmCalls: 2, toolCalls: 1, agentCalls: 1 },
    { name: '50%', llmCalls: 4, toolCalls: 2, agentCalls: 1 },
    { name: '75%', llmCalls: 5, toolCalls: 3, agentCalls: 2 },
    { name: 'End', llmCalls: 7, toolCalls: 4, agentCalls: 2 },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Trace Analytics</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="llmCalls" stroke="#8884d8" />
            <Line type="monotone" dataKey="toolCalls" stroke="#82ca9d" />
            <Line type="monotone" dataKey="agentCalls" stroke="#ffc658" />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default TraceAnalytics;