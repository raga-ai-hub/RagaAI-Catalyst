import React, { useState, useEffect } from 'react';
import { useSidebar } from '../contexts/SidebarContext';
import { useProject } from '../contexts/ProjectContext';
import Sidebar from '../components/Sidebar';
import TraceDetailsPanel from '../components/TraceDetailsPanel';
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, History, ChevronLeft, ChevronRight } from 'lucide-react';
import type { TraceHistoryItem, DetailedTraceComponents } from '../types/trace';
import { fetchTraces, fetchTraceDetails } from '../utils/api';

const TraceHistory: React.FC = () => {
  const { isCollapsed } = useSidebar();
  const { selectedProject, setSelectedProject, projects } = useProject();
  const [traces, setTraces] = useState<TraceHistoryItem[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);
  const [selectedTraceData, setSelectedTraceData] = useState<DetailedTraceComponents | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const tracesPerPage = 10;

  useEffect(() => {
    const loadTraces = async () => {
      if (selectedProject) {
        setIsLoading(true);
        try {
          const fetchedTraces = await fetchTraces(selectedProject);
          if (Array.isArray(fetchedTraces)) {
            setTraces(fetchedTraces);
          } else {
            console.error('Fetched traces is not an array:', fetchedTraces);
            setTraces([]);
          }
        } catch (error) {
          console.error('Error loading traces:', error);
          setTraces([]);
        } finally {
          setIsLoading(false);
        }
      } else {
        setTraces([]);
      }
    };

    loadTraces();
  }, [selectedProject]);

  const handleTraceSelect = async (traceId: string) => {
    setSelectedTraceId(traceId);
    setIsPanelOpen(true);

    try {
      // const traceIdNumber = parseInt(traceId, 10);
      // if (isNaN(traceIdNumber)) {
      //   throw new Error('Invalid trace ID');
      // }
      const traceData = await fetchTraceDetails(traceId);
      setSelectedTraceData(traceData);
    } catch (error) {
      console.error('Error fetching trace details:', error);
      setSelectedTraceData(null);
    }
  };

  const handleCloseSidebar = () => {
    setIsPanelOpen(false);
    setSelectedTraceId(null);
    setSelectedTraceData(null);
  };

  const filteredTraces = traces.filter(trace =>
    String(trace.id).toLowerCase().includes(searchTerm.toLowerCase())
  );

  const totalPages = Math.ceil(filteredTraces.length / tracesPerPage);
  const paginatedTraces = filteredTraces.slice(
    (currentPage - 1) * tracesPerPage,
    currentPage * tracesPerPage
  );

  return (
    <div className="flex h-screen overflow-hidden bg-gray-100 dark:bg-gray-900">
      <Sidebar />
      <div className={`flex-1 flex flex-col overflow-hidden transition-all duration-300 ${isPanelOpen ? 'mr-96' : ''}`}>
        {/* Fixed Header Section */}
        <div className="flex-shrink-0 p-8 bg-gray-100 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
          <div className="flex justify-between items-center mb-8">
            <div className="flex items-center">
              <History className="mr-2 h-8 w-8 text-indigo-600 dark:text-indigo-400" />
              <h1 className="text-3xl font-bold text-gray-800 dark:text-white">Trace History</h1>
            </div>
            <Select
              value={selectedProject?.toString()}
              onValueChange={(value) => setSelectedProject(Number(value))}
            >
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
                <Button onClick={() => setCurrentPage(1)} className="flex items-center">
                  <Search className="w-4 h-4 mr-2" /> Search
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Scrollable Content Area */}
        <div className="flex-1 overflow-hidden flex flex-col min-h-0 p-8">
          {/* Table Container */}
          <div className="flex-1 overflow-hidden flex flex-col bg-white dark:bg-gray-800 rounded-lg shadow">
            <div className="flex-1 overflow-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-700 sticky top-0 z-1">
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
                        onClick={() => handleTraceSelect(trace.id)}
                        className={`hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors duration-150 ease-in-out ${selectedTraceId === trace.id ? 'bg-purple-50 dark:bg-purple-900' : ''
                          }`}
                      >
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-indigo-600 dark:text-indigo-400">
                          {trace.id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                          {new Date(trace.start_time).toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                          {trace.duration ? `${trace.duration.toFixed(2)}s` : "-"}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                          {trace.total_llm_calls ?? "-"}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                          {trace.total_tool_calls ?? "-"}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                          {trace.total_agent_calls ?? "-"}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                          {trace.total_errors ?? "-"}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Fixed Footer Pagination */}
          <div className="flex-shrink-0 mt-6 flex justify-between items-center">
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

export default TraceHistory;