import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Loader2 } from 'lucide-react';
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useProject } from '@/contexts/ProjectContext';
import { fetchTraces, fetchAnalysisTrace } from '@/utils/api';

interface ColorConfig {
  gradient: string[];
  stroke: string;
  text: string;
}

interface ColorConfigs {
  [key: string]: ColorConfig;
}

const TOKEN_COLORS: ColorConfigs = {
  Input: {
    gradient: ['#FF6B6B', '#FF8787'], // Vibrant coral red
    stroke: '#FF4949',
    text: '#FF4949'
  },
  Output: {
    gradient: ['#4ECDC4', '#45B7AF'], // Bright turquoise
    stroke: '#2FB4A9',
    text: '#2FB4A9'
  },
  Reasoning: {
    gradient: ['#FFD93D', '#FFE566'], // Bright yellow
    stroke: '#FFD000',
    text: '#CC9900'
  }
};

const COST_COLORS: ColorConfigs = {
  InputCost: {
    gradient: ['#845EC2', '#9B72D3'], // Rich purple
    stroke: '#7048B6',
    text: '#7048B6'
  },
  OutputCost: {
    gradient: ['#FF8066', '#FF9B85'], // Bright coral
    stroke: '#FF6B52',
    text: '#FF6B52'
  },
  ReasoningCost: {
    gradient: ['#00C2A8', '#1ADBC2'], // Vibrant teal
    stroke: '#00A894',
    text: '#00A894'
  }
};

interface TokenUsage {
  input: number;
  completion: number;
  reasoning: number;
}

interface CostUsage {
  input: number;
  output: number;
  reasoning: number;
}

interface LLMCall {
  model: string;
  token_usage: string;
  cost: string;
}

interface Summary {
  totalTokens: number;
  totalCost: number;
  totalCalls: number;
}

const LLMUsageAnalysis: React.FC = () => {
  const { selectedProject, selectedTraceId } = useProject();
  const [tokenData, setTokenData] = useState<any[]>([]);
  const [costData, setCostData] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showPercentage, setShowPercentage] = useState(false);
  const [summary, setSummary] = useState<Summary>({
    totalTokens: 0,
    totalCost: 0,
    totalCalls: 0
  });

  const parseJsonString = (jsonString: string): any => {
    try {
      return JSON.parse(jsonString);
    } catch (error) {
      console.error('Error parsing JSON string:', error);
      return {};
    }
  };

  const formatNumber = (value: number): string => {
    if (value === 0) return '0';
    if (value < 0.00001) return value.toExponential(2);
    return value.toFixed(5);
  };

  const calculatePercentages = (data: any[]) => {
    return data.map(item => {
      const total = Object.keys(item).reduce((sum, key) => {
        if (key !== 'model' && key !== 'Total' && key !== 'TotalCost') {
          return sum + (item[key] || 0);
        }
        return sum;
      }, 0);

      const percentages = { ...item };
      Object.keys(item).forEach(key => {
        if (key !== 'model' && key !== 'Total' && key !== 'TotalCost') {
          percentages[key] = total > 0 ? ((item[key] || 0) / total) * 100 : 0;
        }
      });
      return percentages;
    });
  };

  const processLLMCall = (
    call: LLMCall, 
    tokenUsage: Record<string, TokenUsage>, 
    costUsage: Record<string, CostUsage>
  ) => {
    const modelName = call.model;
    if (!tokenUsage[modelName]) {
      tokenUsage[modelName] = { input: 0, completion: 0, reasoning: 0 };
      costUsage[modelName] = { input: 0, output: 0, reasoning: 0 };
    }
    
    const tokens = parseJsonString(call.token_usage) as TokenUsage;
    const costs = parseJsonString(call.cost) as CostUsage;
    
    tokenUsage[modelName].input += tokens.input || 0;
    tokenUsage[modelName].completion += tokens.completion || 0;
    tokenUsage[modelName].reasoning += tokens.reasoning || 0;
    
    costUsage[modelName].input += costs.input || 0;
    costUsage[modelName].output += costs.output || 0;
    costUsage[modelName].reasoning += costs.reasoning || 0;

    return {
      tokens: tokens.input + tokens.completion + tokens.reasoning,
      cost: costs.input + costs.output + costs.reasoning
    };
  };

  const fetchData = async () => {
    if (!selectedProject) return;

    try {
      setIsLoading(true);
      setError(null);
      
      let llmTokenUsage: Record<string, TokenUsage> = {};
      let llmCostUsage: Record<string, CostUsage> = {};
      let totalTokens = 0;
      let totalCost = 0;
      let totalCalls = 0;

      if (selectedTraceId) {
        const traceData = await fetchAnalysisTrace(selectedTraceId);
        
        traceData.llm_calls.forEach((call: LLMCall) => {
          const { tokens, cost } = processLLMCall(call, llmTokenUsage, llmCostUsage);
          totalTokens += tokens;
          totalCost += cost;
          totalCalls++;
        });
      } else {
        const traces = await fetchTraces(selectedProject);
        
        for (const trace of traces) {
          const traceData = await fetchAnalysisTrace(trace.id);
          
          traceData.llm_calls.forEach((call: LLMCall) => {
            const { tokens, cost } = processLLMCall(call, llmTokenUsage, llmCostUsage);
            totalTokens += tokens;
            totalCost += cost;
            totalCalls++;
          });
        }
      }

      const formattedTokenData = Object.entries(llmTokenUsage).map(([model, usage]) => ({
        model,
        Input: usage.input,
        Output: usage.completion,
        Reasoning: usage.reasoning,
        Total: usage.input + usage.completion + usage.reasoning
      }));

      const formattedCostData = Object.entries(llmCostUsage).map(([model, usage]) => ({
        model,
        InputCost: usage.input,
        OutputCost: usage.output,
        ReasoningCost: usage.reasoning,
        TotalCost: usage.input + usage.output + usage.reasoning
      }));

      setTokenData(formattedTokenData);
      setCostData(formattedCostData);
      setSummary({ totalTokens, totalCost, totalCalls });
    } catch (err) {
      console.error('Error fetching LLM usage data:', err);
      setError('Failed to fetch LLM usage data. Please try again later.');
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
      const total = payload.reduce((sum: number, pld: any) => sum + (pld.value || 0), 0);
      const isCost = payload[0].name.includes('Cost');
      const colors = isCost ? COST_COLORS : TOKEN_COLORS;
      
      return (
        <div className="bg-white p-4 border rounded shadow">
          <p className="font-bold">{label}</p>
          {payload.map((pld: any) => {
            const colorKey = pld.dataKey;
            return (
              <p key={pld.name} style={{ color: colors[colorKey]?.text }}>
                {`${pld.name}: ${showPercentage 
                  ? `${(pld.value || 0).toFixed(2)}%` 
                  : isCost 
                    ? `$${formatNumber(pld.value)}` 
                    : Math.round(pld.value).toLocaleString()}`}
              </p>
            );
          })}
          <p className="font-bold mt-2">
            {`Total: ${showPercentage 
              ? '100%' 
              : isCost 
                ? `$${formatNumber(total)}` 
                : Math.round(total).toLocaleString()}`}
          </p>
        </div>
      );
    }
    return null;
  };

  const renderChart = (data: any[], title: string, dataKeys: string[], colors: ColorConfigs) => (
    <div>
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={showPercentage ? calculatePercentages(data) : data}
          margin={{ top: 20, right: 30, left: 80, bottom: 120 }}
        >
          {dataKeys.map((key) => (
            <defs key={`gradient-${key}`}>
              <linearGradient id={`${key}Gradient`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={colors[key].gradient[0]} />
                <stop offset="100%" stopColor={colors[key].gradient[1]} />
              </linearGradient>
            </defs>
          ))}
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="model"
            angle={-20}
            textAnchor="end"
            height={100}
            interval={0}
            label={{ 
              value: "Model Name", 
              position: "bottom",
              offset: 80
            }}
          />
          <YAxis
            tickFormatter={(value) => {
              if (showPercentage) return `${value.toFixed(0)}%`;
              if (title.includes("Cost")) return `$${formatNumber(value)}`;
              return Math.round(value).toLocaleString();
            }}
            domain={showPercentage ? [0, 100] : [0, 'auto']}
            label={{ 
              value: title.includes("Cost") 
                ? (showPercentage ? "% of Cost" : "Cost (USD)") 
                : (showPercentage ? "% of Tokens" : "Number of Tokens"),
              angle: -90,
              position: "insideLeft",
              offset: -60,
              style: { textAnchor: 'middle' }
            }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          {dataKeys.map((key) => (
            <Bar 
              key={key} 
              dataKey={key} 
              stackId="a" 
              fill={`url(#${key}Gradient)`}
              stroke={colors[key].stroke}
              strokeWidth={1}
              radius={[4, 4, 0, 0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
          <span className="ml-2">Loading LLM usage data...</span>
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

    if (tokenData.length === 0 || costData.length === 0) {
      return <div className="text-center p-4">No LLM usage data available</div>;
    }

    return (
      <div className="space-y-8">
        <div className="bg-gray-100 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-2">Summary</h3>
          <p>Total Tokens Used: {summary.totalTokens.toLocaleString()}</p>
          <p>Total Cost: ${formatNumber(summary.totalCost)}</p>
          <p>Total LLM Calls: {summary.totalCalls.toLocaleString()}</p>
        </div>

        <div className="flex items-center space-x-2">
          <Switch
            id="llm-percentage-mode"
            checked={showPercentage}
            onCheckedChange={setShowPercentage}
          />
          <Label htmlFor="llm-percentage-mode">Show as percentage</Label>
        </div>

        {renderChart(tokenData, "Token Usage", ["Input", "Output", "Reasoning"], TOKEN_COLORS)}
        {renderChart(costData, "Cost Analysis", ["InputCost", "OutputCost", "ReasoningCost"], COST_COLORS)}
      </div>
    );
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>LLM Usage Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        {renderContent()}
      </CardContent>
    </Card>
  );
};

export default LLMUsageAnalysis;