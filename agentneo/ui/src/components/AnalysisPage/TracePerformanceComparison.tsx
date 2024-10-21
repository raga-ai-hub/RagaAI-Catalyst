import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, ResponsiveContainer } from 'recharts';

const data = [
  { subject: 'Response Time', Trace1: 120, Trace2: 110, Trace3: 130 },
  { subject: 'Token Efficiency', Trace1: 98, Trace2: 96, Trace3: 95 },
  { subject: 'Cost Efficiency', Trace1: 86, Trace2: 90, Trace3: 84 },
  { subject: 'Success Rate', Trace1: 99, Trace2: 98, Trace3: 97 },
  { subject: 'Memory Usage', Trace1: 85, Trace2: 88, Trace3: 90 },
];

const COLORS = ['#3b82f6', '#10b981', '#f59e0b'];

const TracePerformanceComparison: React.FC = () => {
  const [selectedTraces, setSelectedTraces] = useState<string[]>(['Trace1']);

  const handleTraceToggle = (trace: string) => {
    setSelectedTraces(prev =>
      prev.includes(trace) ? prev.filter(t => t !== trace) : [...prev, trace]
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Trace Performance Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex space-x-4 mb-4">
          <span>Select Traces to Compare:</span>
          {['Trace1', 'Trace2', 'Trace3'].map((trace, index) => (
            <label key={trace} className="flex items-center space-x-2">
              <Checkbox
                checked={selectedTraces.includes(trace)}
                onCheckedChange={() => handleTraceToggle(trace)}
              />
              <span style={{ color: COLORS[index] }}>{trace}</span>
            </label>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={400}>
          <RadarChart data={data}>
            <PolarGrid />
            <PolarAngleAxis dataKey="subject" />
            <PolarRadiusAxis angle={30} domain={[0, 150]} />
            {selectedTraces.includes('Trace1') && (
              <Radar name="Trace 1" dataKey="Trace1" stroke={COLORS[0]} fill={COLORS[0]} fillOpacity={0.6} />
            )}
            {selectedTraces.includes('Trace2') && (
              <Radar name="Trace 2" dataKey="Trace2" stroke={COLORS[1]} fill={COLORS[1]} fillOpacity={0.6} />
            )}
            {selectedTraces.includes('Trace3') && (
              <Radar name="Trace 3" dataKey="Trace3" stroke={COLORS[2]} fill={COLORS[2]} fillOpacity={0.6} />
            )}
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default TracePerformanceComparison;