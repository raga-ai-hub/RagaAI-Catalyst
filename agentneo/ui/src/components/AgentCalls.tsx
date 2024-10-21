import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

const AgentCalls = ({ agentCalls }) => {
  if (!agentCalls) {
    return <div>Loading agent calls...</div>;
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