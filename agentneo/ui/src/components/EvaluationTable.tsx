import React, { useState } from 'react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { ArrowUpDown } from "lucide-react";

interface EvaluationTableProps {
  sortedData: any[];
  requestSort: (key: string) => void;
  metricNames: string[];
  onTraceSelect: (traceId: string) => void;
  selectedTraceId: string | null;
}

const EvaluationTable: React.FC<EvaluationTableProps> = ({
  sortedData,
  requestSort,
  metricNames,
  onTraceSelect,
  selectedTraceId
}) => {
  const [expandedCells, setExpandedCells] = useState<Set<string>>(new Set());

  const toggleCellExpansion = (cellId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent row click when expanding cell
    setExpandedCells(prev => {
      const newSet = new Set(prev);
      if (newSet.has(cellId)) {
        newSet.delete(cellId);
      } else {
        newSet.add(cellId);
      }
      return newSet;
    });
  };

  const renderCell = (cellData: any, field: string, traceId: string, metric: string) => {
    if (cellData === undefined || cellData === null || cellData === '') return <div>-</div>;

    const cellId = `${traceId}-${metric}-${field}`;
    const isExpanded = expandedCells.has(cellId);
    let content;

    switch (field) {
      case 'score':
        content = typeof cellData === 'number' ? cellData.toFixed(2) : cellData;
        return (
          <div key={cellId} className="mb-2">
            <span className="font-bold">Score: </span>
            <span>{content}</span>
          </div>
        );
      case 'reason':
      case 'result_detail':
        const truncatedContent = cellData.length > 50 ? cellData.substring(0, 50) + '...' : cellData;
        return (
          <div key={cellId} className="mb-2">
            <span className="font-bold">{field.charAt(0).toUpperCase() + field.slice(1)}: </span>
            <span
              className="cursor-pointer text-blue-600 hover:underline"
              onClick={(e) => toggleCellExpansion(cellId, e)}
            >
              {isExpanded ? cellData : truncatedContent}
            </span>
          </div>
        );
      default:
        content = String(cellData);
        return <div key={cellId} className="mb-2">{content}</div>;
    }
  };

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>
            <Button variant="ghost" onClick={() => requestSort('trace_id')}>
              Trace ID <ArrowUpDown className="ml-2 h-4 w-4" />
            </Button>
          </TableHead>
          {metricNames.map(metric => (
            <TableHead key={metric}>
              <Button variant="ghost" onClick={() => requestSort(metric)}>
                {metric.charAt(0).toUpperCase() + metric.slice(1).replace(/_/g, ' ')} <ArrowUpDown className="ml-2 h-4 w-4" />
              </Button>
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {sortedData.map((row) => (
          <TableRow
            key={row.trace_id}
            onClick={() => onTraceSelect(row.trace_id)}
            className={`cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors duration-150 ease-in-out ${selectedTraceId === row.trace_id ? 'bg-purple-50 dark:bg-purple-900' : ''
              }`}
          >
            <TableCell>
              <span className="text-blue-600 hover:underline cursor-pointer">
                {row.trace_id}
              </span>
            </TableCell>
            {metricNames.map(metric => (
              <TableCell key={`${row.trace_id}-${metric}`} className="align-top">
                <div className="space-y-2">
                  {renderCell(row[metric]?.score, 'score', row.trace_id, metric)}
                  {renderCell(row[metric]?.reason, 'reason', row.trace_id, metric)}
                  {renderCell(row[metric]?.result_detail, 'result_detail', row.trace_id, metric)}
                </div>
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
};

export default EvaluationTable;