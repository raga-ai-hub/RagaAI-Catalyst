import React, { useState, useEffect } from 'react'
import { useSidebar } from '../contexts/SidebarContext'
import { useProject } from '../contexts/ProjectContext'
import { useTheme } from '../theme/ThemeProvider'
import Sidebar from '../components/Sidebar'
import TraceDetailsPanel from '../components/TraceDetailsPanel'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Search, History, ChevronLeft, ChevronRight } from 'lucide-react'
import type { TraceHistoryItem, DetailedTraceComponents } from '../types/trace'
import { fetchTraces, fetchTraceDetails } from '../utils/api'

const TraceHistory: React.FC = () => {
  const { isCollapsed } = useSidebar()
  const { selectedProject, setSelectedProject, projects } = useProject()
  const { theme } = useTheme()
  const [traces, setTraces] = useState<TraceHistoryItem[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [currentPage, setCurrentPage] = useState(1)
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null)
  const [selectedTraceData, setSelectedTraceData] =
    useState<DetailedTraceComponents | null>(null)
  const [isPanelOpen, setIsPanelOpen] = useState(false)
  const tracesPerPage = 10

  useEffect(() => {
    const loadTraces = async () => {
      if (selectedProject) {
        setIsLoading(true)
        try {
          const fetchedTraces = await fetchTraces(selectedProject)
          if (Array.isArray(fetchedTraces)) {
            setTraces(fetchedTraces)
          } else {
            console.error('Fetched traces is not an array:', fetchedTraces)
            setTraces([])
          }
        } catch (error) {
          console.error('Error loading traces:', error)
          setTraces([])
        } finally {
          setIsLoading(false)
        }
      } else {
        setTraces([])
      }
    }

    loadTraces()
  }, [selectedProject])

  const handleTraceSelect = async (traceId: string) => {
    setSelectedTraceId(traceId)
    setIsPanelOpen(true)

    try {
      const traceData = await fetchTraceDetails(traceId)
      setSelectedTraceData(traceData)
    } catch (error) {
      console.error('Error fetching trace details:', error)
      setSelectedTraceData(null)
    }
  }

  const handleCloseSidebar = () => {
    setIsPanelOpen(false)
    setSelectedTraceId(null)
    setSelectedTraceData(null)
  }

  const filteredTraces = traces.filter(trace =>
    String(trace.id).toLowerCase().includes(searchTerm.toLowerCase())
  )

  const totalPages = Math.ceil(filteredTraces.length / tracesPerPage)
  const paginatedTraces = filteredTraces.slice(
    (currentPage - 1) * tracesPerPage,
    currentPage * tracesPerPage
  )

  return (
    <div
      className={`flex h-screen overflow-hidden 
      ${
        theme === 'dark'
          ? 'bg-gray-800 text-white border-gray-700'
          : 'bg-gray-100 text-gray-900 border-gray-200'
      }`}
    >
      <Sidebar />
      <div
        className={`flex-1 flex flex-col overflow-hidden transition-all duration-300 ${
          isPanelOpen ? 'mr-96' : ''
        }`}
      >
        {/* Fixed Header Section */}
        <div
          className={`flex-shrink-0 p-8 border-b 
          ${
            theme === 'dark'
              ? 'bg-gray-800 border-gray-700 text-white'
              : 'bg-gray-100 border-gray-200 text-gray-800'
          }`}
        >
          <div className='flex justify-between items-center mb-8'>
            <div className='flex items-center'>
              <History
                className={`mr-2 h-8 w-8 
                ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}
              />
              <h1
                className={`text-3xl font-bold 
                ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}
              >
                Trace History
              </h1>
            </div>
            <Select
              value={selectedProject?.toString()}
              onValueChange={value => setSelectedProject(Number(value))}
            >
              <SelectTrigger
                className={`w-[200px] 
                ${
                  theme === 'dark'
                    ? 'bg-gray-800 text-white border-gray-700'
                    : 'bg-white text-gray-900 border-gray-300'
                }`}
              >
                <SelectValue placeholder='Select project' />
              </SelectTrigger>
              <SelectContent>
                {projects.map(project => (
                  <SelectItem
                    key={project.id}
                    value={project.id.toString()}
                    className={`
                      ${
                        theme === 'dark'
                          ? 'hover:bg-gray-700 focus:bg-gray-700'
                          : 'hover:bg-gray-100 focus:bg-gray-100'
                      }`}
                  >
                    {project.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Card
            className={`mb-8 
            ${
              theme === 'dark'
                ? 'bg-gray-900 border-gray-700'
                : 'bg-white border-gray-200'
            }`}
          >
            <CardContent className='p-4'>
              <div className='flex space-x-4'>
                <Input
                  type='text'
                  placeholder='Search by Trace ID...'
                  value={searchTerm}
                  onChange={e => setSearchTerm(e.target.value)}
                  className={`flex-grow 
                    ${
                      theme === 'dark'
                        ? 'bg-gray-900 text-white border-gray-600 placeholder-gray-400'
                        : 'bg-white text-gray-900 border-gray-300'
                    }`}
                />
                <Button
                  onClick={() => setCurrentPage(1)}
                  className={`flex items-center 
                    ${
                      theme === 'dark'
                        ? 'bg-indigo-600 hover:bg-indigo-700 text-white'
                        : 'bg-indigo-500 hover:bg-indigo-600 text-white'
                    }`}
                >
                  <Search className='w-4 h-4 mr-2' /> Search
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Scrollable Content Area */}
        <div className='flex-1 overflow-hidden flex flex-col min-h-0 p-8'>
          {/* Table Container */}
          <div
            className={`flex-1 overflow-hidden flex flex-col rounded-lg shadow 
            ${
              theme === 'dark'
                ? 'bg-gray-900 border border-gray-700'
                : 'bg-white border border-gray-200'
            }`}
          >
            <div className='flex-1 overflow-auto'>
              <table
                className={`min-w-full divide-y 
                ${
                  theme === 'dark'
                    ? 'divide-gray-900 text-white'
                    : 'divide-gray-200 text-gray-900'
                }`}
              >
                <thead
                  className={`sticky top-0 z-1 
                  ${
                    theme === 'dark'
                      ? 'bg-gray-700 text-gray-300'
                      : 'bg-gray-50 text-gray-500'
                  }`}
                >
                  <tr>
                    {[
                      'ID',
                      'Start Time',
                      'Duration',
                      'LLM Calls',
                      'Tool Calls',
                      'Agent Calls',
                      'Errors'
                    ].map(header => (
                      <th
                        key={header}
                        className='px-6 py-3 text-left text-xs font-medium uppercase tracking-wider'
                      >
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody
                  className={`
                  ${
                    theme === 'dark'
                      ? 'bg-gray-900 divide-gray-700'
                      : 'bg-white divide-gray-200'
                  }`}
                >
                  {isLoading ? (
                    <tr>
                      <td
                        colSpan={7}
                        className={`px-6 py-4 text-center 
                        ${
                          theme === 'dark' ? 'text-gray-300' : 'text-gray-500'
                        }`}
                      >
                        Loading traces...
                      </td>
                    </tr>
                  ) : paginatedTraces.length === 0 ? (
                    <tr>
                      <td
                        colSpan={7}
                        className={`px-6 py-4 text-center 
                        ${
                          theme === 'dark' ? 'text-gray-300' : 'text-gray-500'
                        }`}
                      >
                        No traces found.{' '}
                        {!selectedProject && 'Please select a project.'}
                      </td>
                    </tr>
                  ) : (
                    paginatedTraces.map(trace => (
                      <tr
                        key={trace.id}
                        onClick={() => handleTraceSelect(trace.id)}
                        className={`hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors duration-150 ease-in-out 
                          ${
                            selectedTraceId === trace.id
                              ? theme === 'dark'
                                ? 'bg-purple-900'
                                : 'bg-purple-50'
                              : ''
                          }`}
                      >
                        {[
                          trace.id,
                          new Date(trace.start_time).toLocaleString(),
                          trace.duration
                            ? `${trace.duration.toFixed(2)}s`
                            : '-',
                          trace.total_llm_calls ?? '-',
                          trace.total_tool_calls ?? '-',
                          trace.total_agent_calls ?? '-',
                          trace.total_errors ?? '-'
                        ].map((value, index) => (
                          <td
                            key={index}
                            className={`px-6 py-4 whitespace-nowrap text-sm 
                              ${
                                index === 0
                                  ? theme === 'dark'
                                    ? 'text-indigo-400'
                                    : 'text-indigo-600'
                                  : theme === 'dark'
                                  ? 'text-gray-300'
                                  : 'text-gray-500'
                              }`}
                          >
                            {value}
                          </td>
                        ))}
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Fixed Footer Pagination */}
          <div className='flex-shrink-0 mt-6 flex justify-between items-center'>
            <Button
              onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
              variant='outline'
              className={`flex items-center 
                ${
                  theme === 'dark'
                    ? 'border-gray-700 text-white hover:bg-gray-700'
                    : 'border-gray-300 text-gray-700 hover:bg-gray-100'
                }`}
            >
              <ChevronLeft className='w-4 h-4 mr-2' /> Previous
            </Button>
            <span
              className={`text-sm 
              ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}
            >
              Page {currentPage} of {totalPages}
            </span>
            <Button
              onClick={() =>
                setCurrentPage(prev => Math.min(prev + 1, totalPages))
              }
              disabled={currentPage === totalPages}
              variant='outline'
              className={`flex items-center 
                ${
                  theme === 'dark'
                    ? 'border-gray-700 text-white hover:bg-gray-700'
                    : 'border-gray-300 text-gray-700 hover:bg-gray-100'
                }`}
            >
              Next <ChevronRight className='w-4 h-4 ml-2' />
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
  )
}

export default TraceHistory
