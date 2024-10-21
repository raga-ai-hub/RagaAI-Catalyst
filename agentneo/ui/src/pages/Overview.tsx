import React, { useEffect, useCallback, useState } from 'react';
import ProjectInformation from '../components/ProjectInformation';
import SystemInformation from '../components/SystemInformation';
import ExecutionGraph from '../components/ExecutionGraph';
import AgentCalls from '../components/AgentCalls';
import InstalledPackages from '../components/InstalledPackages';
import Sidebar from '../components/Sidebar';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { LayoutDashboard } from 'lucide-react';
import { useProject } from '../contexts/ProjectContext';

import { fetchTraceData, fetchSystemInfo, fetchInstalledPackages, fetchAgentCalls, fetchProjectInfo } from '../utils/databaseUtils';

const Index = () => {
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

  const [projectData, setProjectData] = useState(null);
  const [systemInfo, setSystemInfo] = useState(null);
  const [installedPackages, setInstalledPackages] = useState(null);
  const [agentCalls, setAgentCalls] = useState(null);

  const loadProjectData = useCallback(async (projectId: number) => {
    if (!projectId) return;

    console.log(`Loading data for project ${projectId}`);

    try {
      const [projectInfo, sysInfo, packages, calls] = await Promise.all([
        fetchProjectInfo(projectId),
        fetchSystemInfo(projectId),
        fetchInstalledPackages(projectId),
        fetchAgentCalls(projectId)
      ]);

      setProjectData(projectInfo);
      setSystemInfo(sysInfo);
      setInstalledPackages(packages);
      setAgentCalls(calls);
    } catch (error) {
      console.error('Error loading project data:', error);
      setError('Failed to load project data');
    }
  }, [setError]);

  const loadTraceData = useCallback(async (projectId: number, traceId: string) => {
    if (!projectId || !traceId) return;

    try {
      const data = await fetchTraceData(projectId, traceId);
      setSelectedTrace(data);
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
    if (selectedProject && selectedTraceId) {
      loadTraceData(selectedProject, selectedTraceId);
    }
  }, [selectedProject, selectedTraceId, loadTraceData]);

  return (
    <div className="flex h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <Sidebar />
      <div className="flex-1 overflow-y-auto">
        <div className="p-8">
          <div className="flex justify-between items-center mb-8">
            <div className="flex items-center">
              <LayoutDashboard className="mr-2 h-8 w-8 text-indigo-600 dark:text-indigo-400" />
              <div>
                <h1 className="text-3xl font-bold text-gray-800 dark:text-white">Project Dashboard</h1>
                <p className="text-gray-600 dark:text-gray-300">Overview of project metrics and performance</p>
              </div>
            </div>
            <div className="flex space-x-4">
              <Select
                value={selectedProject?.toString() || ''}
                onValueChange={(value) => {
                  console.log('Selected project:', value);
                  setSelectedProject(Number(value));
                }}
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
                onValueChange={(value) => {
                  console.log('Selected trace:', value);
                  setSelectedTraceId(value);
                }}
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

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <ProjectInformation projectData={projectData} />
            <SystemInformation systemData={systemInfo} />
          </div>

          <div className="grid grid-cols-1 gap-8 mb-8">
            <ExecutionGraph />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-8">
            <AgentCalls agentCalls={agentCalls} />
            <InstalledPackages packages={installedPackages} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;