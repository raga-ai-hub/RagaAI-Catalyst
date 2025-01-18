import React, { useEffect, useState } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { ArrowUpIcon, ArrowDownIcon } from 'lucide-react';
import { useProject } from '../../contexts/ProjectContext';
import { fetchTraces, fetchAnalysisTrace } from '@/utils/api';

const MetricCard: React.FC<{ title: string; value: string; previousValue: number; currentValue: number; titleColor: string }> = ({ title, value, previousValue, currentValue, titleColor }) => {
  const calculateChange = () => {
    if (previousValue === 0) return { value: '0%', isPositive: true };
    const change = ((currentValue - previousValue) / previousValue) * 100;
    return {
      value: `${Math.abs(change).toFixed(1)}%`,
      isPositive: change >= 0
    };
  };

  const { value: changeValue, isPositive } = calculateChange();

  return (
    <Card className="bg-white">
      <CardContent className="p-4">
        <h3 className={`text-sm font-medium ${titleColor}`}>{title}</h3>
        <div className="mt-2 flex items-baseline justify-between">
          <p className="text-2xl font-semibold">{value}</p>
          <p className={`flex items-center text-sm ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
            {isPositive ? <ArrowUpIcon className="w-4 h-4 mr-1" /> : <ArrowDownIcon className="w-4 h-4 mr-1" />}
            {changeValue}
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

interface CostObject {
  input: number;
  output: number;
  reasoning: number;
}

interface MetricsData {
  current: {
    successRate: number;
    avgLLMResponseTime: number;
    avgToolCallResponseTime: number;
    avgCostPerLLMCall: number;
  };
  previous: {
    successRate: number;
    avgLLMResponseTime: number;
    avgToolCallResponseTime: number;
    avgCostPerLLMCall: number;
  };
}

const PerformanceMetrics: React.FC = () => {
  const { selectedProject, selectedTraceId } = useProject();
  const [metrics, setMetrics] = useState<MetricsData>({
    current: {
      successRate: 0,
      avgLLMResponseTime: 0,
      avgToolCallResponseTime: 0,
      avgCostPerLLMCall: 0,
    },
    previous: {
      successRate: 0,
      avgLLMResponseTime: 0,
      avgToolCallResponseTime: 0,
      avgCostPerLLMCall: 0,
    }
  });

  useEffect(() => {
    const fetchMetrics = async () => {
      if (!selectedProject) return;

      let totalLLMCalls = 0;
      let totalToolCalls = 0;
      let totalErrors = 0;
      let totalLLMDuration = 0;
      let totalToolDuration = 0;
      let totalCost = 0;

      try {
        if (selectedTraceId) {
          const traceData = await fetchAnalysisTrace(selectedTraceId);

          totalLLMCalls = traceData.llm_calls.length;
          totalToolCalls = traceData.tool_calls.length;
          totalErrors = traceData.errors.length;
          totalLLMDuration = traceData.llm_calls.reduce((sum, call) => sum + call.duration, 0);
          totalToolDuration = traceData.tool_calls.reduce((sum, call) => sum + call.duration, 0);
          totalCost = traceData.llm_calls.reduce((sum, call) => {
            const costObj = JSON.parse(call.cost) as CostObject;
            return sum + Object.values(costObj).reduce((a, b) => a + b, 0);
          }, 0);
        } else {
          const traces = await fetchTraces(selectedProject);

          for (const trace of traces) {
            const traceData = await fetchAnalysisTrace(trace.id);

            totalLLMCalls += traceData.llm_calls.length;
            totalToolCalls += traceData.tool_calls.length;
            totalErrors += traceData.errors.length;
            totalLLMDuration += traceData.llm_calls.reduce((sum, call) => sum + call.duration, 0);
            totalToolDuration += traceData.tool_calls.reduce((sum, call) => sum + call.duration, 0);
            totalCost += traceData.llm_calls.reduce((sum, call) => {
              const costObj = JSON.parse(call.cost) as CostObject;
              return sum + Object.values(costObj).reduce((a, b) => a + b, 0);
            }, 0);
          }
        }

        const totalCalls = totalLLMCalls + totalToolCalls;
        const successRate = totalCalls > 0 ? ((totalCalls - totalErrors) / totalCalls) * 100 : 0;
        const avgLLMResponseTime = totalLLMCalls > 0 ? totalLLMDuration / totalLLMCalls : 0;
        const avgToolCallResponseTime = totalToolCalls > 0 ? totalToolDuration / totalToolCalls : 0;
        const avgCostPerLLMCall = totalLLMCalls > 0 ? totalCost / totalLLMCalls : 0;

        setMetrics(prev => ({
          previous: prev.current,
          current: {
            successRate,
            avgLLMResponseTime,
            avgToolCallResponseTime,
            avgCostPerLLMCall,
          }
        }));
      } catch (error) {
        console.error('Error fetching trace data:', error);
      }
    };

    fetchMetrics();
  }, [selectedProject, selectedTraceId]);

  return (
    <div className="bg-gray-200 p-4 rounded-lg">
      <h2 className="text-xl font-semibold mb-4">Performance Metrics</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard 
          title="Success Rate" 
          value={`${metrics.current.successRate.toFixed(2)}%`} 
          previousValue={metrics.previous.successRate}
          currentValue={metrics.current.successRate}
          titleColor="text-purple-600"
        />
        <MetricCard 
          title="Avg LLM Response Time" 
          value={`${metrics.current.avgLLMResponseTime.toFixed(2)}s`} 
          previousValue={metrics.previous.avgLLMResponseTime}
          currentValue={metrics.current.avgLLMResponseTime}
          titleColor="text-purple-600"
        />
        <MetricCard 
          title="Avg Tool Response Time" 
          value={`${metrics.current.avgToolCallResponseTime.toFixed(2)}s`} 
          previousValue={metrics.previous.avgToolCallResponseTime}
          currentValue={metrics.current.avgToolCallResponseTime}
          titleColor="text-purple-600"
        />
        <MetricCard 
          title="Avg Cost per LLM Call" 
          value={`$${metrics.current.avgCostPerLLMCall.toFixed(4)}`} 
          previousValue={metrics.previous.avgCostPerLLMCall}
          currentValue={metrics.current.avgCostPerLLMCall}
          titleColor="text-purple-600"
        />
      </div>
    </div>
  );
};

export default PerformanceMetrics;