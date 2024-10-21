import React from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { ArrowUpIcon, ArrowDownIcon } from 'lucide-react';

const MetricCard: React.FC<{ title: string; value: string; change: string; isPositive: boolean }> = ({ title, value, change, isPositive }) => (
  <Card className="bg-white">
    <CardContent className="p-4">
      <h3 className="text-sm font-medium text-gray-500">{title}</h3>
      <div className="mt-2 flex items-baseline justify-between">
        <p className="text-2xl font-semibold">{value}</p>
        <p className={`flex items-center text-sm ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
          {isPositive ? <ArrowUpIcon className="w-4 h-4 mr-1" /> : <ArrowDownIcon className="w-4 h-4 mr-1" />}
          {change}
        </p>
      </div>
    </CardContent>
  </Card>
);

const PerformanceMetrics: React.FC = () => {
  return (
    <div className="bg-gray-200 p-4 rounded-lg">
      <h2 className="text-xl font-semibold mb-4">Performance Metrics</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard title="Success Rate" value="98.5%" change="2.1%" isPositive={true} />
        <MetricCard title="Avg Response Time" value="1.2s" change="0.3s" isPositive={false} />
        <MetricCard title="Token Efficiency" value="85%" change="5%" isPositive={true} />
        <MetricCard title="Cost per Task" value="$0.021" change="$0.003" isPositive={false} />
      </div>
    </div>
  );
};

export default PerformanceMetrics;