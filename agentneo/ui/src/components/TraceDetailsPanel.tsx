import React from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface TraceDetailsPanelProps {
  traceData: any; // Replace 'any' with a more specific type when available
}

const TraceDetailsPanel: React.FC<TraceDetailsPanelProps> = ({ traceData }) => {
  if (!traceData) return null;

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>Trace Details</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[calc(100vh-200px)]">
          <div className="space-y-4">
            <Section title="Trace Information">
              <InfoItem label="ID" value={traceData.id} />
              <InfoItem label="Start Time" value={new Date(traceData.start_time).toLocaleString()} />
              <InfoItem label="End Time" value={traceData.end_time ? new Date(traceData.end_time).toLocaleString() : 'N/A'} />
              <InfoItem label="Duration" value={`${traceData.duration?.toFixed(2) || 'N/A'} seconds`} />
            </Section>

            <Section title="User Interactions">
              {traceData.user_interactions?.map((interaction: any, index: number) => (
                <div key={index} className="mb-2">
                  <InfoItem label="Type" value={interaction.interaction_type} />
                  <InfoItem label="Content" value={interaction.content} />
                  <InfoItem label="Timestamp" value={new Date(interaction.timestamp).toLocaleString()} />
                </div>
              ))}
            </Section>

            <Section title="LLM Calls">
              {traceData.llm_calls?.map((call: any, index: number) => (
                <div key={index} className="mb-2">
                  <InfoItem label="Name" value={call.name} />
                  <InfoItem label="Model" value={call.model} />
                  <InfoItem label="Input Prompt" value={call.input_prompt} />
                  <InfoItem label="Output" value={call.output} />
                  <InfoItem label="Duration" value={`${call.duration?.toFixed(2) || 'N/A'} seconds`} />
                  <InfoItem label="Token Usage" value={JSON.stringify(call.token_usage)} />
                  <InfoItem label="Cost" value={JSON.stringify(call.cost)} />
                </div>
              ))}
            </Section>

            <Section title="Tool Calls">
              {traceData.tool_calls?.map((call: any, index: number) => (
                <div key={index} className="mb-2">
                  <InfoItem label="Name" value={call.name} />
                  <InfoItem label="Input Parameters" value={call.input_parameters} />
                  <InfoItem label="Output" value={call.output} />
                  <InfoItem label="Duration" value={`${call.duration?.toFixed(2) || 'N/A'} seconds`} />
                  <InfoItem label="Network Calls" value={JSON.stringify(call.network_calls)} />
                </div>
              ))}
            </Section>

            <Section title="Agent Calls">
              {traceData.agent_calls?.map((call: any, index: number) => (
                <div key={index} className="mb-2">
                  <InfoItem label="Name" value={call.name} />
                  <InfoItem label="Start Time" value={new Date(call.start_time).toLocaleString()} />
                  <InfoItem label="End Time" value={call.end_time ? new Date(call.end_time).toLocaleString() : 'N/A'} />
                </div>
              ))}
            </Section>

            <Section title="Errors">
              {traceData.errors?.map((error: any, index: number) => (
                <div key={index} className="mb-2">
                  <InfoItem label="Error Type" value={error.error_type} />
                  <InfoItem label="Error Message" value={error.error_message} />
                  <InfoItem label="Timestamp" value={new Date(error.timestamp).toLocaleString()} />
                </div>
              ))}
            </Section>
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div>
    <h3 className="text-lg font-semibold mb-2">{title}</h3>
    {children}
  </div>
);

const InfoItem: React.FC<{ label: string; value: string | number }> = ({ label, value }) => (
  <p className="text-sm">
    <span className="font-medium">{label}:</span> {value}
  </p>
);

export default TraceDetailsPanel;