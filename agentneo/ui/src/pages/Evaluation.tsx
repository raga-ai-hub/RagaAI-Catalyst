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
import { fetchEvaluationData } from '../utils/databaseUtils';

const metricNames = [
  'goal_decomposition_efficiency',
  'goal_fulfillment_rate',
  'tool_correctness_metric',
  'tool_call_success_rate_metric'
];

const resultFields = ['trace_id', 'score', 'reason', 'result_detail', 'config', 'start_time', 'end_time', 'duration'];

const Evaluation: React.FC = () => {
  const { isCollapsed } = useSidebar();
  const { selectedProject, setSelectedProject, projects } = useProject();
  const [selectedTraceId, setSelectedTraceId] = useState<string>('all');
  const [startDate, setStartDate] = useState<Date>();
  const [endDate, setEndDate] = useState<Date>();
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'ascending' | 'descending' } | null>(null);
  const [evaluationData, setEvaluationData] = useState<any[]>([]);
  const [allTraces, setAllTraces] = useState<{id: string, name: string}[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      if (selectedProject) {
        const data = await fetchEvaluationData(selectedProject, selectedTraceId === 'all' ? null : selectedTraceId);
        setEvaluationData(data);

        // If we're fetching all traces, update the allTraces state
        if (selectedTraceId === 'all') {
          const uniqueTraces = new Set(data.map(item => item.trace_id));
          setAllTraces([
            { id: 'all', name: 'All Traces' },
            ...Array.from(uniqueTraces).map(id => ({ id, name: `Trace ${id}` }))
          ]);
        }
      }
    };
    fetchData();
  }, [selectedProject, selectedTraceId]);

  const prepareDataForTable = (data: any[]) => {
    const metricMap = new Map();

    data.forEach(item => {
      metricNames.forEach(metric => {
        const metricKey = metric.toLowerCase().replace(/ /g, '_');
        if (item[metricKey]) {
          if (!metricMap.has(metric)) {
            metricMap.set(metric, { metric, results: [] });
          }
          const result = {
            trace_id: item.trace_id,
            score: item[metricKey].score,
            reason: item[metricKey].reason,
            result_detail: JSON.stringify(item[metricKey].result_detail),
            config: JSON.stringify(item[metricKey].config),
            start_time: item.start_time,
            end_time: item.end_time,
            duration: item.duration
          };
          metricMap.get(metric).results.push(result);
        }
      });
    });

    return Array.from(metricMap.values());
  };

  const sortedData = useMemo(() => {
    const preparedData = prepareDataForTable(evaluationData);
    if (sortConfig !== null) {
      preparedData.forEach(metricData => {
        metricData.results.sort((a, b) => {
          const aValue = a[sortConfig.key] || '';
          const bValue = b[sortConfig.key] || '';
          if (aValue < bValue) {
            return sortConfig.direction === 'ascending' ? -1 : 1;
          }
          if (aValue > bValue) {
            return sortConfig.direction === 'ascending' ? 1 : -1;
          }
          return 0;
        });
      });
    }
    return preparedData;
  }, [sortConfig, evaluationData]);

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
      const scores = evaluationData
        .filter(item => item[metricKey])
        .map(item => item[metricKey].score || 0);
      return {
        name: metric,
        min: Math.min(...scores),
        max: Math.max(...scores),
        avg: scores.reduce((a, b) => a + b, 0) / scores.length || 0
      };
    });
  }, [evaluationData]);

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900">
      <Sidebar />
      <div className="flex-1 overflow-y-auto">
        <div className="p-8">
          <div className="flex items-center mb-6">
            <ClipboardCheck className="mr-2 h-8 w-8 text-indigo-600 dark:text-indigo-400" />
            <h1 className="text-3xl font-bold text-gray-800 dark:text-white">Evaluation</h1>
          </div>
          <div className="grid grid-cols-4 gap-4 mb-6">
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
            <Select value={selectedTraceId} onValueChange={setSelectedTraceId}>
              <SelectTrigger>
                <SelectValue placeholder="Select Trace" />
              </SelectTrigger>
              <SelectContent>
                {allTraces.map((trace) => (
                  <SelectItem key={trace.id} value={trace.id}>
                    {trace.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <DateTimePicker date={startDate} setDate={setStartDate} />
            <DateTimePicker date={endDate} setDate={setEndDate} />
          </div>
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Metrics Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
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
              <Input
                type="text"
                placeholder="Search..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="mb-4"
              />
              <EvaluationTable 
                sortedData={sortedData} 
                requestSort={requestSort}
                resultFields={resultFields}
              />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Evaluation;