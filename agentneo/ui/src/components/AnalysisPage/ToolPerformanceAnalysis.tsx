import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Loader2 } from 'lucide-react';
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useProject } from '@/contexts/ProjectContext';
import { fetchTraces, fetchAnalysisTrace } from '@/utils/api';


interface ToolMetrics {
  name: string;
  avgResponseTime: number;
  count: number;
}

interface Summary {
  totalToolCalls: number;
  totalNetworkCalls: number;
}

interface Trace {
  id: string;
  tool_calls: Array<{
    name: string;
    duration: number;
    network_calls?: Array<any>; // Updated type to reflect actual data structure
  }>;
}

const ToolPerformanceAnalysis: React.FC = () => {
  const { selectedProject, selectedTraceId } = useProject();
  const [toolData, setToolData] = useState<ToolMetrics[]>([]);
  const [summary, setSummary] = useState<Summary>({ totalToolCalls: 0, totalNetworkCalls: 0 });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showPercentage, setShowPercentage] = useState(false);

  const chartColors = {
    calls: {
      gradient: ['#6366F1', '#818CF8'], // Vibrant indigo
      stroke: '#4F46E5'
    },
    time: {
      gradient: ['#F472B6', '#F9A8D4'], // Bright pink
      stroke: '#EC4899'
    }
  };

  const calculatePercentages = (data: ToolMetrics[]) => {
    const totalCalls = data.reduce((sum, item) => sum + item.count, 0);
    const totalResponseTime = data.reduce((sum, item) => sum + item.avgResponseTime, 0);

    return data.map(item => ({
      ...item,
      count: totalCalls > 0 ? (item.count / totalCalls) * 100 : 0,
      avgResponseTime: totalResponseTime > 0 ? (item.avgResponseTime / totalResponseTime) * 100 : 0
    }));
  };

  const formatNumber = (value: number): string => {
    if (showPercentage) {
      return `${value.toFixed(1)}%`;
    }
    if (value === 0) return '0';
    if (value < 0.00001) return value.toExponential(2);
    return value.toFixed(3);
  };

  const fetchData = async () => {
    if (!selectedProject) return;

    try {
      setIsLoading(true);
      setError(null);
      
      let toolMetrics: Record<string, ToolMetrics> = {};
      let totalToolCalls = 0;
      let totalNetworkCalls = 0;

      if (selectedTraceId) {
        const traceData = await fetchAnalysisTrace(selectedTraceId);

        for (const toolCall of traceData.tool_calls) {
          if (!toolMetrics[toolCall.name]) {
            toolMetrics[toolCall.name] = {
              name: toolCall.name,
              avgResponseTime: 0,
              count: 0,
            };
          }

          toolMetrics[toolCall.name].avgResponseTime += toolCall.duration;
          toolMetrics[toolCall.name].count += 1;
          totalToolCalls += 1;
          // Fix: Count the length of network_calls array
          totalNetworkCalls += toolCall.network_calls?.length || 0;
        }
      } else {
        const traces = await fetchTraces(selectedProject);
        
        for (const trace of traces) {
          const traceData = await fetchAnalysisTrace(trace.id);

          for (const toolCall of traceData.tool_calls) {
            if (!toolMetrics[toolCall.name]) {
              toolMetrics[toolCall.name] = {
                name: toolCall.name,
                avgResponseTime: 0,
                count: 0,
              };
            }

            toolMetrics[toolCall.name].avgResponseTime += toolCall.duration;
            toolMetrics[toolCall.name].count += 1;
            totalToolCalls += 1;
            // Fix: Count the length of network_calls array
            totalNetworkCalls += toolCall.network_calls?.length || 0;
          }
        }
      }

      const processedToolData = Object.values(toolMetrics).map(metric => ({
        ...metric,
        avgResponseTime: Number((metric.avgResponseTime / metric.count).toFixed(3)),
      }));

      setToolData(processedToolData);
      setSummary({ totalToolCalls, totalNetworkCalls });
    } catch (err) {
      console.error('Error fetching tool performance data:', err);
      setError('Failed to fetch tool performance data. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (selectedProject) {
      fetchData();
    }
  }, [selectedProject, selectedTraceId]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const value = payload[0].value;
      const metricName = payload[0].dataKey === 'count' ? 'Number of Calls' : 'Response Time';
      const unit = payload[0].dataKey === 'avgResponseTime' ? (showPercentage ? '%' : ' seconds') : (showPercentage ? '%' : '');
      
      return (
        <div className="bg-white p-4 border rounded shadow">
          <p className="font-bold">{label}</p>
          <p>{`${metricName}: ${formatNumber(value)}${unit}`}</p>
        </div>
      );
    }
    return null;
  };

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
          <span className="ml-2">Loading tool performance data...</span>
        </div>
      );
    }

    if (error) {
      return (
        <div className="text-center text-red-500 p-4">
          <p>{error}</p>
          <button
            className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            onClick={fetchData}
          >
            Retry
          </button>
        </div>
      );
    }

    if (!selectedProject) {
      return <div className="text-center p-4">Please select a project to view analytics</div>;
    }

    if (toolData.length === 0) {
      return <div className="text-center p-4">No tool performance data available</div>;
    }

    const displayData = showPercentage ? calculatePercentages(toolData) : toolData;

    return (
      <div className="space-y-8">
        <div className="bg-gray-100 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-2">Summary</h3>
          <p>Total Tool Calls: {summary.totalToolCalls.toLocaleString()}</p>
          <p>Total Network Calls: {summary.totalNetworkCalls.toLocaleString()}</p>
        </div>

        <div className="flex items-center space-x-2">
          <Switch
            id="tool-percentage-mode"
            checked={showPercentage}
            onCheckedChange={setShowPercentage}
          />
          <Label htmlFor="tool-percentage-mode">Show as percentage</Label>
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-2">Number of Tool Calls</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart
              data={displayData}
              margin={{ top: 20, right: 30, left: 80, bottom: 120 }}
            >
              <defs>
                <linearGradient id="callsGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={chartColors.calls.gradient[0]} />
                  <stop offset="100%" stopColor={chartColors.calls.gradient[1]} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="name"
                angle={-25}
                textAnchor="end"
                interval={0}
                height={120}
                label={{ 
                  value: "Tool Name", 
                  position: "bottom",
                  offset: 70
                }}
              />
              <YAxis
                tickFormatter={(value) => formatNumber(value)}
                label={{ 
                  value: showPercentage ? "% of Total Calls" : "Number of Calls", 
                  angle: -90,
                  position: "insideLeft",
                  offset: -60,
                  style: { textAnchor: 'middle' }
                }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar 
                dataKey="count" 
                fill="url(#callsGradient)"
                stroke={chartColors.calls.stroke}
                strokeWidth={1}
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-2">Avg Response Time</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart
              data={displayData}
              margin={{ top: 20, right: 30, left: 80, bottom: 120 }}
            >
              <defs>
                <linearGradient id="timeGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={chartColors.time.gradient[0]} />
                  <stop offset="100%" stopColor={chartColors.time.gradient[1]} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="name"
                angle={-25}
                textAnchor="end"
                interval={0}
                height={120}
                label={{ 
                  value: "Tool Name", 
                  position: "bottom",
                  offset: 70
                }}
              />
              <YAxis
                tickFormatter={(value) => `${formatNumber(value)}${showPercentage ? '%' : 's'}`}
                label={{ 
                  value: showPercentage 
                    ? "% of Total Time" 
                    : "Avg Response Time (s)", 
                  angle: -90,
                  position: "insideLeft",
                  offset: -60,
                  style: { textAnchor: 'middle' }
                }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar 
                dataKey="avgResponseTime" 
                fill="url(#timeGradient)"
                stroke={chartColors.time.stroke}
                strokeWidth={1}
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Tool Performance Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        {renderContent()}
      </CardContent>
    </Card>
  );
};

export default ToolPerformanceAnalysis;