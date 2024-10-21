import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts';

const data = [
  {
    name: 'Cost Optimization',
    children: [
      { name: 'LLM Costs', size: 3500, value: '$3,500', percentage: '70%', trend: 'up' },
      { name: 'Tool Usage', size: 1000, value: '$1,000', percentage: '20%', trend: 'down' },
      { name: 'Infrastructure', size: 400, value: '$400', percentage: '8%', trend: 'right' },
      { name: 'Error-related', size: 100, value: '$100', percentage: '2%', trend: 'down' },
    ],
  },
];

const COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444'];

const CostOptimizationInsights: React.FC = () => {
  return (
    <Card className="bg-white">
      <CardHeader>
        <CardTitle>Cost Optimization Insights</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <Treemap
            data={data}
            dataKey="size"
            stroke="#fff"
            content={<CustomizedContent />}
          >
            <Tooltip content={<CustomTooltip />} />
          </Treemap>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

const CustomizedContent: React.FC<any> = ({ root, depth, x, y, width, height, index, colors, name, value, percentage, trend }) => {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        style={{
          fill: COLORS[index % COLORS.length],
          stroke: '#fff',
          strokeWidth: 2,
          strokeOpacity: 1,
        }}
      />
      {
        depth === 1 && (
          <>
            <text x={x + width / 2} y={y + height / 2 - 15} textAnchor="middle" fill="#fff" fontSize={16} fontWeight="bold">
              {name}
            </text>
            <text x={x + width / 2} y={y + height / 2 + 5} textAnchor="middle" fill="#fff" fontSize={14}>
              {value}
            </text>
            <text x={x + width / 2} y={y + height / 2 + 25} textAnchor="middle" fill="#fff" fontSize={12}>
              {percentage}
            </text>
            <text x={x + width / 2} y={y + height / 2 + 45} textAnchor="middle" fill="#fff" fontSize={16}>
              {trend === 'up' ? '▲' : trend === 'down' ? '▼' : '►'}
            </text>
          </>
        )
      }
    </g>
  );
};

const CustomTooltip: React.FC<any> = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-white p-2 border rounded shadow">
        <p>{`${data.name}: ${data.value} (${data.percentage})`}</p>
      </div>
    );
  }
  return null;
};

export default CostOptimizationInsights;
