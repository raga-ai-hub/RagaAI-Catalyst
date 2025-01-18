import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Loader2 } from 'lucide-react';
import { useProject } from '@/contexts/ProjectContext';
import { fetchTraces, fetchAnalysisTrace } from '@/utils/api';

interface ErrorData {
  error_type: string;
  count: number;
}

const ERROR_COLORS = {
  gradient: ['#ef4444', '#f87171'],
  stroke: '#dc2626'
};

const ErrorAnalysis: React.FC = () => {
  const { selectedProject, selectedTraceId } = useProject();
  const [errorData, setErrorData] = useState<ErrorData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    if (!selectedProject) return;

    try {
      setIsLoading(true);
      setError(null);
      
      let errorCounts: Record<string, number> = {};

      if (selectedTraceId) {
        const traceData = await fetchAnalysisTrace(selectedTraceId);

        traceData.errors.forEach((error: any) => {
          if (!errorCounts[error.error_type]) {
            errorCounts[error.error_type] = 0;
          }
          errorCounts[error.error_type]++;
        });
      } else {
        const traces = await fetchTraces(selectedProject);

        for (const trace of traces) {
          const traceData = await fetchAnalysisTrace(trace.id);

          traceData.errors.forEach((error: any) => {
            if (!errorCounts[error.error_type]) {
              errorCounts[error.error_type] = 0;
            }
            errorCounts[error.error_type]++;
          });
        }
      }

      const processedErrorData = Object.entries(errorCounts)
        .map(([error_type, count]) => ({
          error_type,
          count,
        }))
        .sort((a, b) => b.count - a.count);

      setErrorData(processedErrorData);
    } catch (err) {
      console.error('Error fetching error data:', err);
      setError('Failed to fetch error data. Please try again later.');
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
      return (
        <div className="bg-white p-4 border rounded shadow">
          <p className="font-bold">{label}</p>
          <p style={{ color: ERROR_COLORS.stroke }}>
            {`Count: ${payload[0].value}`}
          </p>
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
          <span className="ml-2">Loading error data...</span>
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

    if (errorData.length === 0) {
      return <div className="text-center p-4">No errors found</div>;
    }

    return (
      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={errorData}
          margin={{ top: 20, right: 30, left: 80, bottom: 120 }}
        >
          <defs>
            <linearGradient id="errorGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={ERROR_COLORS.gradient[0]} />
              <stop offset="100%" stopColor={ERROR_COLORS.gradient[1]} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="error_type"
            angle={-30}
            textAnchor="end"
            height={100}
            interval={0}
            label={{ 
              value: "Error Type", 
              position: "bottom",
              offset: 80
            }}
          />
          <YAxis
            label={{ 
              value: "Number of Occurrences",
              angle: -90,
              position: "insideLeft",
              offset: -60,
              style: { textAnchor: 'middle' }
            }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar 
            dataKey="count" 
            fill="url(#errorGradient)"
            stroke={ERROR_COLORS.stroke}
            strokeWidth={1}
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Error Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        {renderContent()}
      </CardContent>
    </Card>
  );
};

export default ErrorAnalysis;