import React, { useMemo, useCallback, useState, useEffect } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Panel,
  Node,
  Edge,
} from 'reactflow';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import { fetchTraceDetails } from '../utils/api';
import 'reactflow/dist/style.css';
import '../styles/ExecutionGraph.css';
import { CustomNode, nodeTypes, nodeStylesByType } from './ExecutionGraphNodes';
import { Legend } from './ExecutionGraphLegend';
import { getLayoutedElements } from './ExecutionGraphLayout';
import { useProject } from '../contexts/ProjectContext';

interface GraphData {
  nodes: Node[];
  edges: Edge[];
}

const calculateTotalCounts = (traceData: any) => {
  let llmCount = traceData.llm_calls?.length || 0;
  let toolCount = traceData.tool_calls?.length || 0;
  let interactionCount = traceData.user_interactions?.length || 0;
  let errorCount = traceData.errors?.length || 0;

  // Add counts from agent calls
  traceData.agent_calls?.forEach((agent: any) => {
    llmCount += agent.llm_calls?.length || 0;
    toolCount += agent.tool_calls?.length || 0;
    interactionCount += agent.user_interactions?.length || 0;
    errorCount += agent.errors?.length || 0;
  });

  return {
    llmCount,
    toolCount,
    interactionCount,
    errorCount
  };
};


const formatDuration = (startTime: string, endTime: string) => {
  const start = new Date(startTime).getTime();
  const end = new Date(endTime).getTime();
  const durationMs = end - start;
  
  if (durationMs < 1000) {
    return `${durationMs}ms`;
  } else if (durationMs < 60000) {
    return `${(durationMs / 1000).toFixed(2)}s`;
  } else {
    return `${(durationMs / 60000).toFixed(2)}min`;
  }
};

const getTotalCounts = (traceData: any) => {
  let llmCount = traceData.llm_calls?.length || 0;
  let toolCount = traceData.tool_calls?.length || 0;
  let interactionCount = traceData.user_interactions?.length || 0;
  let errorCount = traceData.errors?.length || 0;

  // Add counts from agent calls
  traceData.agent_calls?.forEach((agent: any) => {
    llmCount += agent.llm_calls?.length || 0;
    toolCount += agent.tool_calls?.length || 0;
    interactionCount += agent.user_interactions?.length || 0;
    errorCount += agent.errors?.length || 0;
  });

  return { llmCount, toolCount, interactionCount, errorCount };
};


const HORIZONTAL_SPACING = 180;
const NODE_WIDTH = 160;
const NODE_HEIGHT = 80;
const TIME_PADDING = 50;

const ExecutionGraph = () => {
  const { selectedProject, selectedTraceId } = useProject();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [reactFlowInstance, setReactFlowInstance] = useState(null);

  const createTimeBasedGraphData = (traceData: any): GraphData => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    let nodeId = 0;
  
    const getAllNodes = (traceData: any) => {
      const allNodes = [];
      const counts = calculateTotalCounts(traceData);
      
      // Update trace node
      allNodes.push({
        type: 'trace',
        id: `trace-${traceData.id}`,
        startTime: new Date(traceData.start_time).getTime(),
        endTime: new Date(traceData.end_time).getTime(),
        data: {
          type: 'trace',
          name: traceData.name || `Trace ${traceData.id}`,
          startTime: traceData.start_time,
          endTime: traceData.end_time,
          duration: formatDuration(traceData.start_time, traceData.end_time),
          metadata: {
            id: traceData.id,
            agent_calls: traceData.agent_calls,
            llm_calls: traceData.llm_calls,
            tool_calls: traceData.tool_calls,
            user_interactions: traceData.user_interactions,
            errors: traceData.errors,
            totalCounts: counts // Add total counts to metadata
          }
        },
        parentId: null
      });
  
      // Process errors at trace level
      traceData.errors?.forEach((error: any) => {
        if (!error.agent_id && !error.tool_call_id && !error.llm_call_id) {
          allNodes.push({
            type: 'error',
            id: `error-${error.id}`,
            startTime: new Date(error.timestamp).getTime(),
            endTime: new Date(error.timestamp).getTime(),
            data: {
              type: 'error',
              name: error.error_type,
              timestamp: error.timestamp,
              metadata: error
            },
            parentId: `trace-${traceData.id}`
          });
        }
      });
  
      // Add agents and their children
      traceData.agent_calls?.forEach((agent: any) => {
        allNodes.push({
          type: 'agent',
          id: `agent-${agent.id}`,
          startTime: new Date(agent.start_time).getTime(),
          endTime: new Date(agent.end_time).getTime(),
          data: {
            type: 'agent',
            name: agent.name,
            startTime: agent.start_time,
            endTime: agent.end_time,
            duration: formatDuration(agent.start_time, agent.end_time),
            metadata: agent
          },
          parentId: `trace-${traceData.id}`
        });
  
        // Add LLM calls
        agent.llm_calls?.forEach((llm: any) => {
          allNodes.push({
            type: 'llm',
            id: `llm-${llm.id}`,
            startTime: new Date(llm.start_time).getTime(),
            endTime: new Date(llm.end_time).getTime(),
            data: {
              type: 'llm',
              name: llm.name,
              startTime: llm.start_time,
              endTime: llm.end_time,
              duration: formatDuration(llm.start_time, llm.end_time),
              metadata: llm
            },
            parentId: `agent-${agent.id}`
          });
        });
  
        // Add tool calls
        agent.tool_calls?.forEach((tool: any) => {
          allNodes.push({
            type: 'tool',
            id: `tool-${tool.id}`,
            startTime: new Date(tool.start_time).getTime(),
            endTime: new Date(tool.end_time).getTime(),
            data: {
              type: 'tool',
              name: tool.name,
              startTime: tool.start_time,
              endTime: tool.end_time,
              duration: formatDuration(tool.start_time, tool.end_time),
              metadata: tool
            },
            parentId: `agent-${agent.id}`
          });
        });
  
        // Add user interactions
        agent.user_interactions?.forEach((interaction: any) => {
          allNodes.push({
            type: 'interaction',
            id: `interaction-${interaction.id}`,
            startTime: new Date(interaction.timestamp).getTime(),
            endTime: new Date(interaction.timestamp).getTime(),
            data: {
              type: 'interaction',
              name: interaction.interaction_type,
              timestamp: interaction.timestamp,
              metadata: interaction
            },
            parentId: `agent-${agent.id}`
          });
        });
  
        // Add agent-level errors
        agent.errors?.forEach((error: any) => {
          allNodes.push({
            type: 'error',
            id: `error-${error.id}`,
            startTime: new Date(error.timestamp).getTime(),
            endTime: new Date(error.timestamp).getTime(),
            data: {
              type: 'error',
              name: error.error_type,
              timestamp: error.timestamp,
              metadata: error
            },
            parentId: `agent-${agent.id}`
          });
        });
      });
  
      return allNodes;
    };

    const allNodes = getAllNodes(traceData);
    const minTime = Math.min(...allNodes.map(n => n.startTime));
    const maxTime = Math.max(...allNodes.map(n => n.endTime));
    const timeRange = maxTime - minTime;

    const getYPosition = (startTime: number) => {
      return ((startTime - minTime) / timeRange) * (1000 - TIME_PADDING * 2) + TIME_PADDING;
    };

    allNodes.forEach(node => {
      const xPos = node.parentId ? HORIZONTAL_SPACING * (node.type === 'llm' || node.type === 'tool' ? 2 : 1) : 0;
      const yPos = getYPosition(node.startTime);

      nodes.push({
        id: node.id,
        type: 'custom',
        data: {
          ...node.data,
          label: `${node.data.type}: ${node.data.name}`,
        },
        position: { x: xPos, y: yPos },
        style: {
          width: NODE_WIDTH,
          height: NODE_HEIGHT,
        },
      });

      if (node.parentId) {
        edges.push({
          id: `edge-${nodeId++}`,
          source: node.parentId,
          target: node.id,
          type: 'smoothstep',
          animated: true,
        });
      }
    });

    return { nodes, edges };
  };

  useEffect(() => {
    const loadGraphData = async () => {
      if (selectedTraceId) {
        try {
          const traceData = await fetchTraceDetails(selectedTraceId);
          const { nodes: graphNodes, edges: graphEdges } = createTimeBasedGraphData(traceData);
          setNodes(graphNodes);
          setEdges(graphEdges);
        } catch (error) {
          console.error('Error loading graph data:', error);
        }
      }
    };

    loadGraphData();
  }, [selectedTraceId, setNodes, setEdges]);

  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(
    () => getLayoutedElements(nodes, edges, 'TB'),
    [nodes, edges]
  );

  const filteredNodes = useMemo(() => {
    if (!searchTerm) return layoutedNodes;
    return layoutedNodes.filter(node =>
      node.data.label.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [layoutedNodes, searchTerm]);

  const onInit = useCallback((instance) => {
    setReactFlowInstance(instance);
    instance.fitView({ padding: 0.2 });
  }, []);

  const handleSearch = () => {
    if (searchTerm && reactFlowInstance) {
      const matchingNode = filteredNodes.find(node =>
        node.data.label.toLowerCase().includes(searchTerm.toLowerCase())
      );
      if (matchingNode) {
        reactFlowInstance.fitView({ padding: 0.2, nodes: [matchingNode] });
      }
    }
  };

  const handleZoomIn = () => reactFlowInstance?.zoomIn();
  const handleZoomOut = () => reactFlowInstance?.zoomOut();
  const handleFitView = () => reactFlowInstance?.fitView({ padding: 0.2 });

  return (
    <Card className="mt-8">
      <CardHeader>
        <CardTitle className="text-2xl font-bold">Execution Graph</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-4 flex items-center space-x-2">
          <Input
            type="text"
            placeholder="Search nodes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="max-w-sm"
          />
          <Button variant="outline" size="icon" onClick={handleSearch}>
            <Search className="h-4 w-4" />
          </Button>
        </div>
        <div style={{ height: '70vh' }} className="border rounded-lg overflow-hidden shadow-lg">
          <ReactFlow
            nodes={filteredNodes}
            edges={layoutedEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            onInit={onInit}
            fitView
            attributionPosition="bottom-left"
            minZoom={0.1}
            maxZoom={1.5}
            defaultEdgeOptions={{
              type: 'smoothstep',
              style: { strokeWidth: 2 },
            }}
          >
            <Background color="#e0e7ff" gap={16} />
            <Controls showInteractive={false} />
            <MiniMap
              nodeColor={(node) => {
                const type = (node.data?.type || 'default') as keyof typeof nodeStylesByType;
                return nodeStylesByType[type]?.backgroundColor || nodeStylesByType.agent.backgroundColor;
              }}
              maskColor="#f0f0f080"
            />
            <Panel position="top-right" className="bg-white p-2 rounded shadow-md">
              <div className="flex space-x-2">
                <Button variant="outline" size="icon" onClick={handleZoomIn}>
                  <ZoomIn className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="icon" onClick={handleZoomOut}>
                  <ZoomOut className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="icon" onClick={handleFitView}>
                  <Maximize2 className="h-4 w-4" />
                </Button>
              </div>
            </Panel>
            <Legend />
          </ReactFlow>
        </div>
      </CardContent>
    </Card>
  );
};

export default ExecutionGraph;