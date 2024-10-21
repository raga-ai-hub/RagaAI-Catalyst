import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

interface CompleteTraceDataProps {
  traceData: any; // Replace 'any' with a more specific type if available
}

const CompleteTraceData: React.FC<CompleteTraceDataProps> = ({ traceData }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Complete Trace Data</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[400px]">
          <pre className="text-sm whitespace-pre-wrap break-words">
            {JSON.stringify(traceData, null, 2)}
          </pre>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export default CompleteTraceData;