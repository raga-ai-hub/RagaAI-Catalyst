import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface TraceInformationProps {
  trace: {
    id: string;
    startTime: string;
    duration: number;
    llmCallCount: number;
    toolCallCount: number;
    agentCallCount: number;
    errorCount: number;
  };
}

const TraceInformation: React.FC<TraceInformationProps> = ({ trace }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Trace Information</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-1 gap-x-4 gap-y-8 sm:grid-cols-2">
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">Trace ID</dt>
            <dd className="mt-1 text-sm text-gray-900">{trace.id}</dd>
          </div>
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">Start Time</dt>
            <dd className="mt-1 text-sm text-gray-900">{new Date(trace.startTime).toLocaleString()}</dd>
          </div>
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">Duration</dt>
            <dd className="mt-1 text-sm text-gray-900">{trace.duration.toFixed(2)}s</dd>
          </div>
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">LLM Calls</dt>
            <dd className="mt-1 text-sm text-gray-900">{trace.llmCallCount}</dd>
          </div>
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">Tool Calls</dt>
            <dd className="mt-1 text-sm text-gray-900">{trace.toolCallCount}</dd>
          </div>
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">Agent Calls</dt>
            <dd className="mt-1 text-sm text-gray-900">{trace.agentCallCount}</dd>
          </div>
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">Errors</dt>
            <dd className="mt-1 text-sm text-gray-900">{trace.errorCount}</dd>
          </div>
        </dl>
      </CardContent>
    </Card>
  );
};

export default TraceInformation;