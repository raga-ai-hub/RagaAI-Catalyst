import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const TopPerformanceCriteria: React.FC = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Top Performance Criteria</CardTitle>
      </CardHeader>
      <CardContent>
        <table className="w-full">
          <tbody>
            <tr>
              <td className="py-2"><strong>Response Time:</strong></td>
              <td>Average time to complete tasks (lower is better)</td>
            </tr>
            <tr>
              <td className="py-2"><strong>Success Rate:</strong></td>
              <td>Percentage of tasks completed successfully (higher is better)</td>
            </tr>
            <tr>
              <td className="py-2"><strong>Token Efficiency:</strong></td>
              <td>Output quality vs. tokens used (higher is better)</td>
            </tr>
            <tr>
              <td className="py-2"><strong>Memory Usage:</strong></td>
              <td>Efficiency of system memory utilization (lower is better)</td>
            </tr>
            <tr>
              <td className="py-2"><strong>Cost Efficiency:</strong></td>
              <td>Cost per successful task completion (lower is better)</td>
            </tr>
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
};

export default TopPerformanceCriteria;