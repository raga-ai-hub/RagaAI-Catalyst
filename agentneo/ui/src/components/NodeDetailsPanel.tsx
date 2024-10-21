import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface NodeDetailsPanelProps {
  selectedNode: any | null;
}

const NodeDetailsPanel: React.FC<NodeDetailsPanelProps> = ({ selectedNode }) => {
  if (!selectedNode) {
    return null;
  }

  return (
    <Card className="mt-4 bg-white">
      <CardHeader>
        <CardTitle className="text-indigo-600">Node Details</CardTitle>
      </CardHeader>
      <CardContent>
        <p><strong>Name:</strong> {selectedNode.data.label}</p>
        <p><strong>Type:</strong> {selectedNode.data.type}</p>
        <p><strong>Duration:</strong> {selectedNode.data.duration}s</p>
        <p><strong>Memory Used:</strong> {selectedNode.data.memory_used} MB</p>
        {selectedNode.data.type === 'llm' && (
          <p><strong>Tokens:</strong> {selectedNode.data.token_usage?.input + selectedNode.data.token_usage?.completion || 'N/A'}</p>
        )}
      </CardContent>
    </Card>
  );
};

export default NodeDetailsPanel;