import React, { useState } from 'react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { ArrowUpDown } from "lucide-react";

interface EvaluationTableProps {
  sortedData: any[];
  requestSort: (key: string) => void;
  resultFields: string[];
}

const EvaluationTable: React.FC<EvaluationTableProps> = ({ sortedData, requestSort, resultFields }) => {
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [expandedDetails, setExpandedDetails] = useState<Set<string>>(new Set());

  const toggleRowExpansion = (metricId: string) => {
    setExpandedRows(prev => {
      const newSet = new Set(prev);
      if (newSet.has(metricId)) {
        newSet.delete(metricId);
      } else {
        newSet.add(metricId);
      }
      return newSet;
    });
  };

  const toggleDetailExpansion = (cellId: string) => {
    setExpandedDetails(prev => {
      const newSet = new Set(prev);
      if (newSet.has(cellId)) {
        newSet.delete(cellId);
      } else {
        newSet.add(cellId);
      }
      return newSet;
    });
  };

  const renderCell = (cellData: any, field: string, metricId: string, resultIndex: number) => {
    if (cellData === undefined || cellData === null || cellData === '') return <TableCell>-</TableCell>;

    const cellId = `${metricId}-${resultIndex}-${field}`;
    let content;

    switch (field) {
      case 'score':
        content = typeof cellData === 'number' ? cellData.toFixed(2) : cellData;
        break;
      case 'reason':
      case 'result_detail':
        const isExpanded = expandedDetails.has(cellId);
        const truncatedContent = cellData.length > 50 ? cellData.substring(0, 50) + '...' : cellData;
        content = (
          <div 
            className="cursor-pointer text-blue-600 hover:underline"
            onClick={() => toggleDetailExpansion(cellId)}
          >
            {isExpanded ? cellData : truncatedContent}
          </div>
        );
        break;
      case 'config':
        try {
          const parsedData = JSON.parse(cellData);
          content = (
            <pre className="whitespace-pre-wrap overflow-hidden text-xs">
              {JSON.stringify(parsedData, null, 2)}
            </pre>
          );
        } catch {
          content = cellData;
        }
        break;
      case 'start_time':
      case 'end_time':
        content = new Date(cellData).toLocaleString();
        break;
      case 'duration':
        content = typeof cellData === 'number' ? `${cellData.toFixed(2)}s` : cellData;
        break;
      default:
        content = String(cellData);
    }

    return <TableCell key={cellId}>{content}</TableCell>;
  };

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Metric</TableHead>
          {resultFields.map(header => (
            <TableHead key={header}>
              <Button variant="ghost" onClick={() => requestSort(header)}>
                {header.charAt(0).toUpperCase() + header.slice(1).replace(/_/g, ' ')} <ArrowUpDown className="ml-2 h-4 w-4" />
              </Button>
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {sortedData.map((row) => (
          <React.Fragment key={row.metric}>
            <TableRow 
              className="hover:bg-gray-100 cursor-pointer"
              onClick={() => toggleRowExpansion(row.metric)}
            >
              <TableCell>{row.metric}</TableCell>
              {resultFields.map(field => renderCell(row.results[0]?.[field], field, row.metric, 0))}
            </TableRow>
            {expandedRows.has(row.metric) && row.results.slice(1).map((result, index) => (
              <TableRow key={`${row.metric}-${index + 1}`} className="bg-gray-50">
                <TableCell></TableCell>
                {resultFields.map(field => renderCell(result[field], field, row.metric, index + 1))}
              </TableRow>
            ))}
          </React.Fragment>
        ))}
      </TableBody>
    </Table>
  );
};

export default EvaluationTable;