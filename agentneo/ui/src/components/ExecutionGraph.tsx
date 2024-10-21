import React, { useMemo, useCallback, useState, useEffect } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Panel,
} from 'reactflow';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import 'reactflow/dist/style.css';
import '../styles/ExecutionGraph.css';
import { CustomNode, nodeTypes, nodeStylesByType } from './ExecutionGraphNodes';
import { Legend } from './ExecutionGraphLegend';
import { getLayoutedElements } from './ExecutionGraphLayout';
import { useProject } from '../contexts/ProjectContext';
import { fetchExecutionGraphData, GraphNode, GraphEdge } from '../utils/databaseUtils';
import { debounce } from 'lodash';

const ExecutionGraph = () => {
  const { selectedProject, selectedTraceId } = useProject();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [reactFlowInstance, setReactFlowInstance] = useState(null);

  useEffect(() => {
    const loadGraphData = async () => {
      if (selectedProject && selectedTraceId) {
        const { nodes: fetchedNodes, edges: fetchedEdges } = await fetchExecutionGraphData(selectedProject, selectedTraceId);
        const initialNodes = fetchedNodes.map((node) => ({
          id: node.id,
          type: 'custom',
          data: {
            ...node.data,
            type: node.type,
            label: `${node.type}: ${node.data.name}`,
          },
          position: { x: 0, y: 0 },
        }));
        setNodes(initialNodes);
        setEdges(fetchedEdges);
      }
    };

    loadGraphData();
  }, [selectedProject, selectedTraceId, setNodes, setEdges]);

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