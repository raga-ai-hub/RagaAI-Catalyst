import React, { useState, useEffect } from 'react';
import { useSidebar } from '../contexts/SidebarContext';
import Sidebar from '../components/Sidebar';
import TraceDetails from '../components/TraceDetails';
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Search, History, ChevronLeft, ChevronRight } from 'lucide-react';
import { useProject } from '../contexts/ProjectContext';
import { fetchTraceHistory, TraceHistoryItem, fetchDetailedTraceComponents, DetailedTraceComponents } from '../utils/databaseUtils';

const TraceHistory: React.FC = () => {
  const { isCollapsed } = useSidebar();
  const { selectedProject, setSelectedProject, projects } = useProject();
  const [traces, setTraces] = useState<TraceHistoryItem[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedTraceData, setExpandedTraceData] = useState<DetailedTraceComponents | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [expandedTrace, setExpandedTrace] = useState<string | null>(null);
  const tracesPerPage = 10;

  useEffect(() => {
    const loadTraces = async () => {
      if (selectedProject) {
        setIsLoading(true);
        try {
          const fetchedTraces = await fetchTraceHistory(selectedProject);
          setTraces(fetchedTraces);
        } catch (error) {
          console.error('Error loading traces:', error);
        } finally {
          setIsLoading(false);
        }
      } else {
        setTraces([]);
      }
    };

    loadTraces();
  }, [selectedProject]);

  const filteredTraces = traces.filter(trace =>
    trace.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const totalPages = Math.ceil(filteredTraces.length / tracesPerPage);
  const paginatedTraces = filteredTraces.slice(
    (currentPage - 1) * tracesPerPage,
    currentPage * tracesPerPage
  );

  const handleSearch = () => {
    setCurrentPage(1);
  };

  const handleExpandTrace = async (traceId: string) => {
    setExpandedTrace(traceId);
    try {
      const detailedData = await fetchDetailedTraceComponents(traceId);
      setExpandedTraceData(detailedData);
    } catch (error) {
      console.error('Error fetching detailed trace data:', error);
      setExpandedTraceData(null);
    }
  };

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900">
      <Sidebar />
      <div className="flex-1 overflow-y-auto p-8">
        <div className="flex justify-between items-center mb-8">
          <div className="flex items-center">
            <History className="mr-2 h-8 w-8 text-indigo-600 dark:text-indigo-400" />
            <h1 className="text-3xl font-bold text-gray-800 dark:text-white">Trace History</h1>
          </div>
          <Select value={selectedProject?.toString()} onValueChange={(value) => setSelectedProject(Number(value))}>
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="Select project" />
            </SelectTrigger>
            <SelectContent>
              {projects.map((project) => (
                <SelectItem key={project.id} value={project.id.toString()}>
                  {project.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Card className="mb-8">
          <CardContent className="p-4">
            <div className="flex space-x-4">
              <Input
                type="text"
                placeholder="Search by Trace ID..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="flex-grow"
              />
              <Button onClick={handleSearch} className="flex items-center">
                <Search className="w-4 h-4 mr-2" /> Search
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">ID</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Start Time</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Duration</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">LLM Calls</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Tool Calls</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Agent Calls</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Errors</th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {isLoading ? (
                <tr>
                  <td colSpan={7} className="px-6 py-4 text-center text-gray-500 dark:text-gray-300">
                    Loading traces...
                  </td>
                </tr>
              ) : paginatedTraces.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-6 py-4 text-center text-gray-500 dark:text-gray-300">
                    No traces found. {!selectedProject && "Please select a project."}
                  </td>
                </tr>
              ) : (
                paginatedTraces.map((trace) => (
                  <tr
                    key={trace.id}
                    className="hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors duration-150 ease-in-out"
                    onClick={() => handleExpandTrace(trace.id)}
                  >
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-indigo-600 dark:text-indigo-400">{trace.id}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{new Date(trace.start_time).toLocaleString()}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{trace.duration.toFixed(2)}s</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{trace.llm_call_count}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{trace.tool_call_count}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{trace.agent_call_count}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{trace.error_count}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <div className="mt-6 flex justify-between items-center">
          <Button
            onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
            disabled={currentPage === 1}
            variant="outline"
            className="flex items-center"
          >
            <ChevronLeft className="w-4 h-4 mr-2" /> Previous
          </Button>
          <span className="text-sm text-gray-600 dark:text-gray-300">
            Page {currentPage} of {totalPages}
          </span>
          <Button
            onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
            disabled={currentPage === totalPages}
            variant="outline"
            className="flex items-center"
          >
            Next <ChevronRight className="w-4 h-4 ml-2" />
          </Button>
        </div>

        <Dialog open={!!expandedTrace} onOpenChange={() => {
          setExpandedTrace(null);
          setExpandedTraceData(null);
        }}>
          <DialogContent className="max-w-6xl w-11/12 max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Trace Details: {expandedTrace}</DialogTitle>
            </DialogHeader>
            {expandedTraceData ? (
              <TraceDetails traceData={expandedTraceData} />
            ) : (
              <div>Loading trace details...</div>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
};

export default TraceHistory;