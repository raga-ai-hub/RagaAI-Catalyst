import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Clock, CheckCircle2, DollarSign, Cpu } from 'lucide-react';

interface CriteriaItem {
  icon: React.ReactNode;
  title: string;
  description: string;
  trend: string;
  color: string;
}

const criteria: CriteriaItem[] = [
  {
    icon: <CheckCircle2 className="w-6 h-6" />,
    title: "Success Rate",
    description: "Percentage of tasks completed successfully",
    trend: "Higher is better",
    color: "text-emerald-600"
  },
  {
    icon: <Clock className="w-6 h-6" />,
    title: "Response Time",
    description: "Average time to complete tasks",
    trend: "Lower is better",
    color: "text-blue-600"
  },
  {
    icon: <DollarSign className="w-6 h-6" />,
    title: "Cost Efficiency",
    description: "Total cost per successful operation",
    trend: "Lower is better",
    color: "text-purple-600"
  },
  {
    icon: <Cpu className="w-6 h-6" />,
    title: "Memory Usage",
    description: "Efficiency of system memory utilization",
    trend: "Lower is better",
    color: "text-amber-600"
  }
];

const TopPerformanceCriteria: React.FC = () => {
  return (
    <Card className="shadow-lg">
      <CardHeader className="border-b">
        <CardTitle className="text-2xl font-bold text-gray-800">
          Top Performance Criteria
        </CardTitle>
      </CardHeader>
      <CardContent className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {criteria.map((item, index) => (
            <div
              key={index}
              className="flex items-start space-x-4 p-4 rounded-lg bg-white border border-gray-100 hover:border-gray-200 transition-all duration-200 hover:shadow-md"
            >
              <div className={`${item.color} p-2 rounded-lg bg-opacity-10`}>
                {item.icon}
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-lg text-gray-800 mb-1">
                  {item.title}
                </h3>
                <p className="text-gray-600 text-sm mb-2">
                  {item.description}
                </p>
                <div className="flex items-center">
                  <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                    item.trend.includes('Higher') 
                      ? 'bg-emerald-100 text-emerald-700'
                      : 'bg-blue-100 text-blue-700'
                  }`}>
                    {item.trend}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default TopPerformanceCriteria;