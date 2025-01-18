import React, { useEffect, useCallback, useState } from 'react';
import ProjectInformation from '../components/ProjectInformation';
import SystemInformation from '../components/SystemInformation';
import ExecutionGraph from '../components/ExecutionGraph';
import AgentCalls from '../components/AgentCalls';
import { InstalledPackages } from '../components/InstalledPackages';
import Sidebar from '../components/Sidebar';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { LayoutDashboard } from 'lucide-react';
import { useProject } from '../contexts/ProjectContext';
import { fetchProjectDetails, fetchTraceDetails } from '../utils/api';
import ExecutionTimeline from '../components/ExecutionTimeline';

// Define interfaces to match component props
interface ProjectData {
  id: number;
  project_name: string;
  start_time: string;
  end_time: string | null;
  duration: number | null;
  total_cost: number | string | null;
  total_tokens: number | null;
}

interface SystemData {
  pythonVersion: string;
  os: string;
  cpu: string;
  gpu: string;
  totalMemory: string;
  diskSpace: string;
}

interface AgentCall {
  id: string;
  name: string;
  start_time: string;
  end_time: string;
  llm_calls: any[];
  tool_calls: any[];
  user_interactions: any[];
  errors: any[];
}

interface AgentCallStats {
  name: string;
  count: number;
  avgDuration: number;
  avgToolCalls: number;
  avgLLMCalls: number;
}

interface PackageInfo {
  name: string;
  version: string;
}

const Overview = () => {
  const {
    selectedProject,
    setSelectedProject,
    selectedTraceId,
    setSelectedTraceId,
    setSelectedTrace,
    projects,
    traces,
    setError
  } = useProject();

  const [projectData, setProjectData] = useState<ProjectData | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemData | null>(null);
  const [installedPackages, setInstalledPackages] = useState<PackageInfo[]>([]);
  const [agentCalls, setAgentCalls] = useState<AgentCallStats[] | null>(null);

  const processAgentCallStats = (traceDetails: any): AgentCallStats[] => {
    if (!traceDetails?.agent_calls) return [];

    // Group agent calls by name
    const callsByName = traceDetails.agent_calls.reduce((acc: any, call: any) => {
      if (!acc[call.name]) {
        acc[call.name] = [];
      }
      acc[call.name].push({
        duration: new Date(call.end_time).getTime() - new Date(call.start_time).getTime(),
        toolCalls: call.tool_calls?.length || 0,
        llmCalls: call.llm_calls?.length || 0,
      });
      return acc;
    }, {});

    // Calculate averages for each agent
    return Object.entries(callsByName).map(([name, calls]: [string, any[]]) => ({
      name,
      count: calls.length,
      avgDuration: Number((calls.reduce((sum, c) => sum + c.duration, 0) / calls.length / 1000).toFixed(2)), // Convert to seconds
      avgToolCalls: Number((calls.reduce((sum, c) => sum + c.toolCalls, 0) / calls.length).toFixed(2)),
      avgLLMCalls: Number((calls.reduce((sum, c) => sum + c.llmCalls, 0) / calls.length).toFixed(2)),
    }));
  };

  const loadProjectData = useCallback(async (projectId: number) => {
    if (!projectId) return;

    try {
      const projectDetails = await fetchProjectDetails(projectId);
      
      // Set project data
      setProjectData({
        id: projectDetails.id,
        project_name: projectDetails.project_name,
        start_time: projectDetails.start_time,
        end_time: projectDetails.end_time,
        duration: projectDetails.duration,
        total_cost: projectDetails.total_cost,
        total_tokens: projectDetails.total_tokens,
      });

      // Set system info if available
      if (projectDetails.system_info) {
        setSystemInfo({
          pythonVersion: projectDetails.system_info.python_version,
          os: `${projectDetails.system_info.os_name} ${projectDetails.system_info.os_version}`,
          cpu: projectDetails.system_info.cpu_info,
          gpu: projectDetails.system_info.gpu_info,
          totalMemory: projectDetails.system_info.memory_total,
          diskSpace: projectDetails.system_info.disk_info,
        });

        // Set installed packages if available
        if (projectDetails.system_info?.installed_packages) {
          try {
            const packagesObj = JSON.parse(projectDetails.system_info.installed_packages);
            const formattedPackages = Object.entries(packagesObj).map(([name, version]) => ({
              name,
              version: version.toString()
            }));
            setInstalledPackages(formattedPackages);
          } catch (error) {
            console.error('Error parsing installed packages:', error);
            setInstalledPackages([]);
          }
        }
      }

    } catch (error) {
      console.error('Error loading project data:', error);
      setError('Failed to load project data');
    }
  }, [setError]);

  const loadTraceData = useCallback(async (traceId: string) => {
    if (!traceId) return;

    try {
      const traceDetails = await fetchTraceDetails(traceId);
      setSelectedTrace(traceDetails);
      // Process and set agent call statistics
      setAgentCalls(processAgentCallStats(traceDetails));
    } catch (error) {
      console.error('Error loading trace data:', error);
      setError('Failed to load trace data');
    }
  }, [setSelectedTrace, setError]);

  useEffect(() => {
    if (selectedProject) {
      loadProjectData(selectedProject);
    }
  }, [selectedProject, loadProjectData]);

  useEffect(() => {
    if (selectedTraceId) {
      loadTraceData(selectedTraceId);
    }
  }, [selectedTraceId, loadTraceData]);

  return (
    <div className="flex h-screen overflow-hidden bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <Sidebar />

      {/* Main content area with proper overflow handling */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Fixed header section */}
        <div className="flex-shrink-0 p-8 bg-opacity-90 backdrop-blur-sm bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-8">
            <div className="flex items-center">
              <LayoutDashboard className="mr-2 h-8 w-8 text-indigo-600 dark:text-indigo-400" />
              <div>
                <h1 className="text-3xl font-bold text-gray-800 dark:text-white">Project Dashboard</h1>
                <p className="text-gray-600 dark:text-gray-300">Overview of project metrics and performance</p>
              </div>
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
                onValueChange={(value) => setSelectedTraceId(value)}
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
            {/* First row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
              <div className="h-full">
                <ProjectInformation projectData={projectData} />
              </div>
              <div className="h-full">
                <SystemInformation systemData={systemInfo} />
              </div>
            </div>

            {/* Middle row
            <div className="mb-8">
              <ExecutionGraph />
            </div> */}
            <div className="grid grid-cols-1 gap-8 mb-8">
              <ExecutionGraph />
              <ExecutionTimeline />
            </div>

            {/* Last row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="h-full">
                <AgentCalls agentCalls={agentCalls} />
              </div>
              <div className="h-full">
                <InstalledPackages packages={installedPackages} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Overview;