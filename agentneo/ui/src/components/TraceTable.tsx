import React from 'react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface TraceTableProps {
  traces: any[];
  onExpandTrace: (traceId: string) => void;
}

const TraceTable: React.FC<TraceTableProps> = ({ traces, onExpandTrace }) => {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>ID</TableHead>
          <TableHead>Start Time</TableHead>
          <TableHead>Duration</TableHead>
          <TableHead>LLM Calls</TableHead>
          <TableHead>Tool Calls</TableHead>
          <TableHead>Agent Calls</TableHead>
          <TableHead>Errors</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {traces.map((trace) => (
          <TableRow
            key={trace.id}
            className="cursor-pointer hover:bg-gray-50"
            onClick={() => onExpandTrace(trace.id)}
          >
            <TableCell>{trace.id}</TableCell>
            <TableCell>{new Date(trace.start_time).toLocaleString()}</TableCell>
            <TableCell>{trace.duration ? `${trace.duration.toFixed(2)}s` : 'N/A'}</TableCell>
            <TableCell>{trace.llm_call_count}</TableCell>
            <TableCell>{trace.tool_call_count}</TableCell>
            <TableCell>{trace.agent_call_count}</TableCell>
            <TableCell>{trace.error_count}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
};

export default TraceTable;