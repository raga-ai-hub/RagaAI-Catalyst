import React from 'react';
import PerformanceMetrics from '../components/AnalysisPage/PerformanceMetrics';
import LLMUsageAnalysis from '../components/AnalysisPage/LLMUsageAnalysis';
import ToolPerformanceAnalysis from '../components/AnalysisPage/ToolPerformanceAnalysis';
import ErrorAnalysis from '../components/AnalysisPage/ErrorAnalysis';
import CostOptimizationInsights from '../components/AnalysisPage/CostOptimizationInsights';
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
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900">
      <Sidebar />
      <div className="flex-1 overflow-y-auto">
        <div className="p-8">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <PieChart className="mr-2 h-8 w-8 text-indigo-600 dark:text-indigo-400" />
              <h1 className="text-3xl font-bold text-gray-800 dark:text-white">Analytics</h1>
            </div>
            <div className="flex space-x-4">
              <Select
                value={selectedProject?.toString() || ''}
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
              <Select
                value={selectedTraceId || ''}
                onValueChange={setSelectedTraceId}
              >
                <SelectTrigger className="w-[200px]">
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
          <div className="space-y-6">
            <PerformanceMetrics />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <LLMUsageAnalysis projectId={selectedProject} />
              <ToolPerformanceAnalysis />
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ErrorAnalysis />
              <CostOptimizationInsights />
            </div>
            <TracePerformanceComparison />
            <TopPerformanceCriteria />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analysis;