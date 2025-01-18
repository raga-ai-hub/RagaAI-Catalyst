import React, { useState, useMemo, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Sidebar from '../components/Sidebar';
import EvaluationTable from '../components/EvaluationTable';
import DateTimePicker from '../components/DateTimePicker';
import { ClipboardCheck } from 'lucide-react';
import { useSidebar } from '../contexts/SidebarContext';
import { useProject } from '../contexts/ProjectContext';
// import { fetchEvaluationData } from '../utils/databaseUtils';
import TraceDetailsPanel from '../components/TraceDetailsPanel';
// import { fetchTraceDetails } from '../utils/api';
import type { DetailedTraceComponents } from '../types/trace';

import { fetchEvaluationData, fetchTraceDetails } from '../utils/api';

const metricNames = [
  'goal_decomposition_efficiency',
  'goal_fulfillment_rate',
  'tool_call_correctness_rate',
  'tool_call_success_rate'
];

const Evaluation: React.FC = () => {
  const { isCollapsed } = useSidebar();
  const { selectedProject, setSelectedProject, projects } = useProject();
  const [startDate, setStartDate] = useState<Date>();
  const [endDate, setEndDate] = useState<Date>();
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'ascending' | 'descending' } | null>(null);
  const [evaluationData, setEvaluationData] = useState<any[]>([]);
  const [filteredData, setFilteredData] = useState<any[]>([]);
  const [allTraces, setAllTraces] = useState<{ id: string, name: string }[]>([]);
  const [selectedTraceId, setSelectedTraceId] = useState<string>('all');
  const [selectedTraceData, setSelectedTraceData] = useState<DetailedTraceComponents | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      if (selectedProject) {
        try {
          const data = await fetchEvaluationData(selectedProject, selectedTraceId === 'all' ? null : selectedTraceId);
          
          // Group metrics by trace_id
          const groupedData = data.reduce((acc, item) => {
            if (!acc[item.trace_id]) {
              acc[item.trace_id] = {
                trace_id: item.trace_id,
                goal_decomposition_efficiency: null,
                goal_fulfillment_rate: null,
                tool_call_correctness_rate: null,
                tool_call_success_rate: null
              };
            }
            
            // Match metric_name to the corresponding property
            const metricKey = item.metric_name.toLowerCase();
            acc[item.trace_id][metricKey] = {
              score: item.score,
              reason: item.reason,
              result_detail: item.result_detail
            };
            
            return acc;
          }, {});

          const transformedData = Object.values(groupedData);
          
          setEvaluationData(transformedData);
          setFilteredData(transformedData);

          // Extract unique trace IDs
          const uniqueTraces = Array.from(new Set(data.map(item => item.trace_id)));
          const tracesArray = [
            { id: 'all', name: 'All Traces' },
            ...uniqueTraces.map(id => ({ id: id.toString(), name: `Trace ${id}` }))
          ];

          setAllTraces(tracesArray);
        } catch (error) {
          console.error('Error fetching evaluation data:', error);
        }
      }
    };
    fetchData();
  }, [selectedProject, selectedTraceId]);

  useEffect(() => {
    if (!startDate && !endDate) {
      setFilteredData(evaluationData);
      return;
    }

    const filtered = evaluationData.filter(item => {
      const itemDate = new Date(item.start_time);
      const startOfDay = startDate ? new Date(startDate.setHours(0, 0, 0, 0)) : null;
      const endOfDay = endDate ? new Date(endDate.setHours(23, 59, 59, 999)) : null;

      const isAfterStart = startOfDay ? itemDate >= startOfDay : true;
      const isBeforeEnd = endOfDay ? itemDate <= endOfDay : true;

      return isAfterStart && isBeforeEnd;
    });

    setFilteredData(filtered);
  }, [startDate, endDate, evaluationData]);

  const prepareDataForTable = (data: any[]) => {
    const preparedData = {};
    data.forEach(item => {
      if (!preparedData[item.trace_id]) {
        preparedData[item.trace_id] = { trace_id: item.trace_id };
      }
      metricNames.forEach(metric => {
        const metricKey = metric.toLowerCase().replace(/ /g, '_');
        if (item[metricKey]) {
          preparedData[item.trace_id][metric] = {
            score: item[metricKey].score,
            reason: item[metricKey].reason,
            result_detail: JSON.stringify(item[metricKey].result_detail),
          };
        }
      });
    });
    return Object.values(preparedData);
  };

  const sortedData = useMemo(() => {
    const preparedData = prepareDataForTable(filteredData);
    if (sortConfig !== null) {
      preparedData.sort((a, b) => {
        const aValue = sortConfig.key === 'trace_id' ? a[sortConfig.key] : a[sortConfig.key]?.score || '';
        const bValue = sortConfig.key === 'trace_id' ? b[sortConfig.key] : b[sortConfig.key]?.score || '';
        if (aValue < bValue) {
          return sortConfig.direction === 'ascending' ? -1 : 1;
        }
        if (aValue > bValue) {
          return sortConfig.direction === 'ascending' ? 1 : -1;
        }
        return 0;
      });
    }
    return preparedData;
  }, [sortConfig, filteredData]);

  const requestSort = (key: string) => {
    let direction: 'ascending' | 'descending' = 'ascending';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') {
      direction = 'descending';
    }
    setSortConfig({ key, direction });
  };

  const metricsData = useMemo(() => {
    return metricNames.map(metric => {
      const metricKey = metric.toLowerCase().replace(/ /g, '_');
      let scores;

      if (selectedTraceId === 'all') {
        // Calculate metrics across all traces
        scores = filteredData
          .filter(item => item[metricKey])
          .map(item => item[metricKey].score || 0);
      } else {
        // Calculate metrics for selected trace only
        scores = filteredData
          .filter(item => item.trace_id.toString() === selectedTraceId && item[metricKey])
          .map(item => item[metricKey].score || 0);
      }

      // If no scores available, return zeros
      if (scores.length === 0) {
        return {
          name: metric,
          min: 0,
          max: 0,
          avg: 0
        };
      }

      return {
        name: metric,
        min: Math.min(...scores),
        max: Math.max(...scores),
        avg: scores.reduce((a, b) => a + b, 0) / scores.length
      };
    });
  }, [filteredData, selectedTraceId]);

  const handleTraceSelect = async (traceId: string) => {
    setSelectedTraceId(traceId);
    if (traceId !== 'all') {
      setIsPanelOpen(true);
      try {
        const traceData = await fetchTraceDetails(traceId);
        setSelectedTraceData(traceData);
      } catch (error) {
        console.error('Error fetching trace details:', error);
        setSelectedTraceData(null);
      }
    } else {
      setIsPanelOpen(false);
      setSelectedTraceData(null);
    }
  };

  const handleCloseSidebar = () => {
    setIsPanelOpen(false);
    setSelectedTraceId('all');
    setSelectedTraceData(null);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-gray-100 dark:bg-gray-900">
      <Sidebar />
      <div className={`flex-1 flex flex-col overflow-hidden transition-all duration-300 ${isPanelOpen ? 'mr-96' : ''}`}>
        {/* Fixed Header */}
        <div className="flex-shrink-0 p-8 bg-gray-100 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
          <div className="flex items-center mb-6">
            <ClipboardCheck className="mr-2 h-8 w-8 text-indigo-600 dark:text-indigo-400" />
            <h1 className="text-3xl font-bold text-gray-800 dark:text-white">Evaluation</h1>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Select
              value={selectedProject?.toString() || ''}
              onValueChange={(value) => setSelectedProject(Number(value))}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select Project" />
              </SelectTrigger>
              <SelectContent>
                {projects.map((project) => (
                  <SelectItem key={project.id} value={project.id.toString()}>
                    {project.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select
              value={selectedTraceId}
              onValueChange={(value) => handleTraceSelect(value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="All Traces" />
              </SelectTrigger>
              <SelectContent>
                {allTraces.map((trace) => (
                  <SelectItem key={trace.id} value={trace.id}>
                    {trace.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <DateTimePicker
              date={startDate}
              setDate={setStartDate}
            />
            <DateTimePicker
              date={endDate}
              setDate={setEndDate}
            />
          </div>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto min-h-0">
          <div className="p-8 space-y-6">
            {selectedProject ? (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle>Metrics Distribution</CardTitle>
                  </CardHeader>
                  <CardContent className="h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={metricsData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="name"
                          interval={0}
                          tick={({ x, y, payload, index }) => {
                            const totalLabels = metricsData.length;
                            let textAnchor: "start" | "middle" | "end";
                            let xOffset: number;

                            if (index === 0) {
                              textAnchor = "start";
                              xOffset = 0;
                            } else if (index === totalLabels - 1) {
                              textAnchor = "end";
                              xOffset = 0;
                            } else {
                              textAnchor = "middle";
                              xOffset = 0;
                            }

                            return (
                              <g transform={`translate(${x},${y})`}>
                                <text
                                  x={xOffset}
                                  y={0}
                                  dy={16}
                                  textAnchor={textAnchor}
                                  fill="#666"
                                  fontSize={12}
                                >
                                  {payload.value}
                                </text>
                              </g>
                            );
                          }}
                          height={60}
                        />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="min" stroke="#8884d8" />
                        <Line type="monotone" dataKey="max" stroke="#82ca9d" />
                        <Line type="monotone" dataKey="avg" stroke="#ffc658" />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Evaluation Details</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <Input
                        type="text"
                        placeholder="Search..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                      />
                      <div className="overflow-auto">
                        <EvaluationTable
                          sortedData={sortedData}
                          requestSort={requestSort}
                          metricNames={metricNames}
                          onTraceSelect={handleTraceSelect}
                          selectedTraceId={selectedTraceId}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </>
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-gray-500 dark:text-gray-400">
                  Please select a project to view analytics
                </div>
              </div>
            )}
          </div>
        </div>

        <TraceDetailsPanel
          isOpen={isPanelOpen}
          onClose={handleCloseSidebar}
          traceData={selectedTraceData}
        />
      </div>
    </div>
  );
};

export default Evaluation;