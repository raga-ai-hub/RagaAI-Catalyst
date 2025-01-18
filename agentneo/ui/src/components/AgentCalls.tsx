import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface AgentCallStats {
  name: string;
  count: number;
  avgDuration: number;
  avgToolCalls: number;
  avgLLMCalls: number;
}

interface AgentCallsProps {
  agentCalls: AgentCallStats[] | null;
}

const AgentCalls: React.FC<AgentCallsProps> = ({ agentCalls }) => {
  if (!agentCalls) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Agent Calls</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-gray-500 dark:text-gray-400">
            Loading agent calls...
          </div>
        </CardContent>
      </Card>
    );
  }

  if (agentCalls.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Agent Calls</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-gray-500 dark:text-gray-400">
            No agent calls found
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Agent Calls</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>NAME</TableHead>
              <TableHead>COUNT</TableHead>
              <TableHead>AVG DURATION (S)</TableHead>
              <TableHead>AVG TOOL CALLS</TableHead>
              <TableHead>AVG LLM CALLS</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {agentCalls.map((row, index) => (
              <TableRow key={index}>
                <TableCell>{row.name}</TableCell>
                <TableCell>{row.count}</TableCell>
                <TableCell>{row.avgDuration}</TableCell>
                <TableCell>{row.avgToolCalls}</TableCell>
                <TableCell>{row.avgLLMCalls}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
};

export default AgentCalls;