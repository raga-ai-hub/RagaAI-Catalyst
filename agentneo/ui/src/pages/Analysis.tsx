import React from 'react';
import PerformanceMetrics from '../components/AnalysisPage/PerformanceMetrics';
import LLMUsageAnalysis from '../components/AnalysisPage/LLMUsageAnalysis';
import ToolPerformanceAnalysis from '../components/AnalysisPage/ToolPerformanceAnalysis';
import ErrorAnalysis from '../components/AnalysisPage/ErrorAnalysis';
import TimeAnalysis from '../components/AnalysisPage/TimeAnalysis';
import TracePerformanceComparison from '../components/AnalysisPage/TracePerformanceComparison';
import TopPerformanceCriteria from '../components/AnalysisPage/TopPerformanceCriteria';
import Sidebar from '../components/Sidebar';
import { useSidebar } from '../contexts/SidebarContext';
import { useProject } from '../contexts/ProjectContext';
import { PieChart } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const Analysis: React.FC = () => {
  const { isCollapsed } = useSidebar();
  const {
    selectedProject,
    setSelectedProject,
    selectedTraceId,
    setSelectedTraceId,
    projects,
    traces
  } = useProject();

  return (
    <div className="flex h-screen overflow-hidden bg-gray-100 dark:bg-gray-900">
      <Sidebar />

      {/* Main content container with flex column layout */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Fixed header section */}
        <div className="flex-shrink-0 p-8 bg-gray-100 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div className="flex items-center">
              <PieChart className="mr-2 h-8 w-8 text-indigo-600 dark:text-indigo-400" />
              <h1 className="text-3xl font-bold text-gray-800 dark:text-white">Analytics</h1>
            </div>
            <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto">
              <Select
                value={selectedProject?.toString() || ''}
                onValueChange={(value) => setSelectedProject(Number(value))}
              >
                <SelectTrigger className="w-full sm:w-[200px]">
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
              <Select
                value={selectedTraceId || ''}
                onValueChange={setSelectedTraceId}
                disabled={!selectedProject}
              >
                <SelectTrigger className="w-full sm:w-[200px]">
                  <SelectValue placeholder="Select trace" />
                </SelectTrigger>
                <SelectContent>
                  {traces.map((trace) => (
                    <SelectItem key={trace.id} value={trace.id}>
                      {trace.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {/* Scrollable content area */}
        <div className="flex-1 overflow-y-auto min-h-0">
          <div className="p-8">
            <div className="space-y-6 max-w-full">
              {selectedProject ? (
                <>
                  {/* Performance Metrics Section */}
                  <div className="w-full">
                    <PerformanceMetrics />
                  </div>

                  {/* LLM Usage and Tool Performance Grid */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="h-full">
                      <LLMUsageAnalysis />
                    </div>
                    <div className="h-full">
                      <ToolPerformanceAnalysis />
                    </div>
                  </div>

                  {/* Error and Time Analysis Grid */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="h-full">
                      <ErrorAnalysis />
                    </div>
                    <div className="h-full">
                      <TimeAnalysis />
                    </div>
                  </div>

                  {/* Full Width Sections */}
                  <div className="w-full">
                    <TracePerformanceComparison />
                  </div>
                  <div className="w-full">
                    <TopPerformanceCriteria />
                  </div>
                </>
              ) : (
                <div className="flex items-center justify-center min-h-[200px] bg-white dark:bg-gray-800 rounded-lg shadow">
                  <div className="text-center text-gray-500 dark:text-gray-400">
                    Please select a project to view analytics
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analysis;