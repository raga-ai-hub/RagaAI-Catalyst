import React, { useState, useMemo, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'
import Sidebar from '../components/Sidebar'
import EvaluationTable from '../components/EvaluationTable'
import DateTimePicker from '../components/DateTimePicker'
import { ClipboardCheck } from 'lucide-react'
import { useSidebar } from '../contexts/SidebarContext'
import { useProject } from '../contexts/ProjectContext'
import { useTheme } from '../theme/ThemeProvider'
import TraceDetailsPanel from '../components/TraceDetailsPanel'
import { fetchEvaluationData, fetchTraceDetails } from '../utils/api'
import type { DetailedTraceComponents } from '../types/trace'

const metricNames = [
  'goal_decomposition_efficiency',
  'goal_fulfillment_rate',
  'tool_call_correctness_rate',
  'tool_call_success_rate'
]

const Evaluation: React.FC = () => {
  const { theme } = useTheme()
  const { isCollapsed } = useSidebar()
  const { selectedProject, setSelectedProject, projects } = useProject()
  const [startDate, setStartDate] = useState<Date>()
  const [endDate, setEndDate] = useState<Date>()
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [sortConfig, setSortConfig] = useState<{
    key: string
    direction: 'ascending' | 'descending'
  } | null>(null)
  const [evaluationData, setEvaluationData] = useState<any[]>([])
  const [filteredData, setFilteredData] = useState<any[]>([])
  const [allTraces, setAllTraces] = useState<{ id: string; name: string }[]>([])
  const [selectedTraceId, setSelectedTraceId] = useState<string>('all')
  const [selectedTraceData, setSelectedTraceData] =
    useState<DetailedTraceComponents | null>(null)
  const [isPanelOpen, setIsPanelOpen] = useState(false)

  // Theme-based styles
  const themeStyles = {
    background: theme === 'dark' ? '#111827' : '#ffffff', // bg-gray-900
    text: theme === 'dark' ? '#ffffff' : '#000000', // text-white
    border: theme === 'dark' ? '#374151' : '#e5e5e5', // border-gray-700
    muted: theme === 'dark' ? '#D1D5DB' : '#6b7280', // text-gray-300
    primary: theme === 'dark' ? '#4F46E5' : '#3b82f6', // bg-indigo-600
    chart: {
      line1: theme === 'dark' ? '#60a5fa' : '#3b82f6',
      line2: theme === 'dark' ? '#34d399' : '#10b981',
      line3: theme === 'dark' ? '#f472b6' : '#ec4899',
      grid: theme === 'dark' ? '#374151' : '#e5e5e5' // border-gray-700
    },
    secondary: {
      background: theme === 'dark' ? '#1F2937' : '#f9fafb', // bg-gray-800
      hover: theme === 'dark' ? '#374151' : '#f3f4f6', // hover:bg-gray-700
      selected: theme === 'dark' ? '#701A75' : '#FAE8FF' // bg-purple-900 : bg-purple-50
    }
  }

  useEffect(() => {
    const fetchData = async () => {
      if (selectedProject) {
        try {
          const data = await fetchEvaluationData(
            selectedProject,
            selectedTraceId === 'all' ? null : selectedTraceId
          )

          // Group metrics by trace_id
          const groupedData = data.reduce((acc, item) => {
            if (!acc[item.trace_id]) {
              acc[item.trace_id] = {
                trace_id: item.trace_id,
                goal_decomposition_efficiency: null,
                goal_fulfillment_rate: null,
                tool_call_correctness_rate: null,
                tool_call_success_rate: null
              }
            }

            // Match metric_name to the corresponding property
            const metricKey = item.metric_name.toLowerCase()
            acc[item.trace_id][metricKey] = {
              score: item.score,
              reason: item.reason,
              result_detail: item.result_detail
            }

            return acc
          }, {})

          const transformedData = Object.values(groupedData)

          setEvaluationData(transformedData)
          setFilteredData(transformedData)

          // Extract unique trace IDs
          const uniqueTraces = Array.from(
            new Set(data.map(item => item.trace_id))
          )
          const tracesArray = [
            { id: 'all', name: 'All Traces' },
            ...uniqueTraces.map(id => ({
              id: id.toString(),
              name: `Trace ${id}`
            }))
          ]

          setAllTraces(tracesArray)
        } catch (error) {
          console.error('Error fetching evaluation data:', error)
        }
      }
    }
    fetchData()
  }, [selectedProject, selectedTraceId])

  useEffect(() => {
    if (!startDate && !endDate) {
      setFilteredData(evaluationData)
      return
    }

    const filtered = evaluationData.filter(item => {
      const itemDate = new Date(item.start_time)
      const startOfDay = startDate
        ? new Date(startDate.setHours(0, 0, 0, 0))
        : null
      const endOfDay = endDate
        ? new Date(endDate.setHours(23, 59, 59, 999))
        : null

      const isAfterStart = startOfDay ? itemDate >= startOfDay : true
      const isBeforeEnd = endOfDay ? itemDate <= endOfDay : true

      return isAfterStart && isBeforeEnd
    })

    setFilteredData(filtered)
  }, [startDate, endDate, evaluationData])

  const prepareDataForTable = (data: any[]) => {
    const preparedData = {}
    data.forEach(item => {
      if (!preparedData[item.trace_id]) {
        preparedData[item.trace_id] = { trace_id: item.trace_id }
      }
      metricNames.forEach(metric => {
        const metricKey = metric.toLowerCase().replace(/ /g, '_')
        if (item[metricKey]) {
          preparedData[item.trace_id][metric] = {
            score: item[metricKey].score,
            reason: item[metricKey].reason,
            result_detail: JSON.stringify(item[metricKey].result_detail)
          }
        }
      })
    })
    return Object.values(preparedData)
  }

  const sortedData = useMemo(() => {
    const preparedData = prepareDataForTable(filteredData)
    if (sortConfig !== null) {
      preparedData.sort((a, b) => {
        const aValue =
          sortConfig.key === 'trace_id'
            ? a[sortConfig.key]
            : a[sortConfig.key]?.score || ''
        const bValue =
          sortConfig.key === 'trace_id'
            ? b[sortConfig.key]
            : b[sortConfig.key]?.score || ''
        if (aValue < bValue) {
          return sortConfig.direction === 'ascending' ? -1 : 1
        }
        if (aValue > bValue) {
          return sortConfig.direction === 'ascending' ? 1 : -1
        }
        return 0
      })
    }
    return preparedData
  }, [sortConfig, filteredData])

  const requestSort = (key: string) => {
    let direction: 'ascending' | 'descending' = 'ascending'
    if (
      sortConfig &&
      sortConfig.key === key &&
      sortConfig.direction === 'ascending'
    ) {
      direction = 'descending'
    }
    setSortConfig({ key, direction })
  }

  const metricsData = useMemo(() => {
    return metricNames.map(metric => {
      const metricKey = metric.toLowerCase().replace(/ /g, '_')
      let scores

      if (selectedTraceId === 'all') {
        // Calculate metrics across all traces
        scores = filteredData
          .filter(item => item[metricKey])
          .map(item => item[metricKey].score || 0)
      } else {
        // Calculate metrics for selected trace only
        scores = filteredData
          .filter(
            item =>
              item.trace_id.toString() === selectedTraceId && item[metricKey]
          )
          .map(item => item[metricKey].score || 0)
      }

      // If no scores available, return zeros
      if (scores.length === 0) {
        return {
          name: metric,
          min: 0,
          max: 0,
          avg: 0
        }
      }

      return {
        name: metric,
        min: Math.min(...scores),
        max: Math.max(...scores),
        avg: scores.reduce((a, b) => a + b, 0) / scores.length
      }
    })
  }, [filteredData, selectedTraceId])

  const handleTraceSelect = async (traceId: string) => {
    setSelectedTraceId(traceId)
    if (traceId !== 'all') {
      setIsPanelOpen(true)
      try {
        const traceData = await fetchTraceDetails(traceId)
        setSelectedTraceData(traceData)
      } catch (error) {
        console.error('Error fetching trace details:', error)
        setSelectedTraceData(null)
      }
    } else {
      setIsPanelOpen(false)
      setSelectedTraceData(null)
    }
  }

  const handleCloseSidebar = () => {
    setIsPanelOpen(false)
    setSelectedTraceId('all')
    setSelectedTraceData(null)
  }

  return (
    <div
      className='flex h-screen overflow-hidden'
      style={{ backgroundColor: themeStyles.secondary.background }}
    >
      <Sidebar />
      <div
        className={`flex-1 flex flex-col overflow-hidden transition-all duration-300 ${
          isPanelOpen ? 'mr-96' : ''
        }`}
      >
        {/* Fixed Header */}
        <div
          className='flex-shrink-0 p-8'
          style={{
            backgroundColor: themeStyles.secondary.background,
            borderBottom: `1px solid ${themeStyles.border}`
          }}
        >
          <div className='flex items-center mb-6'>
            <ClipboardCheck
              className='mr-2 h-8 w-8'
              style={{ color: themeStyles.primary }}
            />
            <h1
              className='text-3xl font-bold'
              style={{ color: themeStyles.text }}
            >
              Evaluation
            </h1>
          </div>

          <div className='grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4'>
            <Select
              value={selectedProject?.toString() || ''}
              onValueChange={value => setSelectedProject(Number(value))}
            >
              <SelectTrigger>
                <SelectValue placeholder='Select Project' />
              </SelectTrigger>
              <SelectContent>
                {projects.map(project => (
                  <SelectItem key={project.id} value={project.id.toString()}>
                    {project.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select
              value={selectedTraceId}
              onValueChange={value => handleTraceSelect(value)}
            >
              <SelectTrigger>
                <SelectValue placeholder='All Traces' />
              </SelectTrigger>
              <SelectContent>
                {allTraces.map(trace => (
                  <SelectItem key={trace.id} value={trace.id}>
                    {trace.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <DateTimePicker date={startDate} setDate={setStartDate} />
            <DateTimePicker date={endDate} setDate={setEndDate} />
          </div>
        </div>

        {/* Scrollable Content */}
        <div className='flex-1 overflow-y-auto min-h-0'>
          <div className='p-8 space-y-6'>
            {selectedProject ? (
              <>
                <Card
                  style={{
                    backgroundColor: themeStyles.background,
                    border: `1px solid ${themeStyles.border}`
                  }}
                >
                  <CardHeader>
                    <CardTitle style={{ color: themeStyles.text }}>
                      Metrics Distribution
                    </CardTitle>
                  </CardHeader>
                  <CardContent className='h-[400px]'>
                    <ResponsiveContainer width='100%' height='100%'>
                      <LineChart
                        data={metricsData}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid
                          strokeDasharray='3 3'
                          stroke={themeStyles.chart.grid}
                        />
                        <XAxis
                          dataKey='name'
                          interval={0}
                          tick={({ x, y, payload, index }) => {
                            const totalLabels = metricsData.length
                            let textAnchor: 'start' | 'middle' | 'end'
                            let xOffset: number

                            if (index === 0) {
                              textAnchor = 'start'
                              xOffset = 0
                            } else if (index === totalLabels - 1) {
                              textAnchor = 'end'
                              xOffset = 0
                            } else {
                              textAnchor = 'middle'
                              xOffset = 0
                            }

                            return (
                              <g transform={`translate(${x},${y})`}>
                                <text
                                  x={xOffset}
                                  y={0}
                                  dy={16}
                                  textAnchor={textAnchor}
                                  fill={themeStyles.muted}
                                  style={{ fontSize: '12px' }}
                                >
                                  {payload.value}
                                </text>
                              </g>
                            )
                          }}
                          height={60}
                          stroke={themeStyles.muted}
                        />
                        <YAxis stroke={themeStyles.muted} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: themeStyles.background,
                            border: `1px solid ${themeStyles.border}`,
                            color: themeStyles.text
                          }}
                        />
                        <Legend />
                        <Line
                          type='monotone'
                          dataKey='min'
                          stroke={themeStyles.chart.line1}
                        />
                        <Line
                          type='monotone'
                          dataKey='max'
                          stroke={themeStyles.chart.line2}
                        />
                        <Line
                          type='monotone'
                          dataKey='avg'
                          stroke={themeStyles.chart.line3}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card
                  style={{
                    backgroundColor: themeStyles.background,
                    border: `1px solid ${themeStyles.border}`
                  }}
                >
                  <CardHeader>
                    <CardTitle style={{ color: themeStyles.text }}>
                      Evaluation Details
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className='space-y-4'>
                      <Input
                        type='text'
                        placeholder='Search...'
                        value={searchTerm}
                        onChange={e => setSearchTerm(e.target.value)}
                        style={{
                          backgroundColor: themeStyles.background,
                          color: themeStyles.text,
                          borderColor: themeStyles.border
                        }}
                      />
                      <div className='overflow-auto'>
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
              <div className='flex items-center justify-center h-full'>
                <div
                  className='text-center'
                  style={{ color: themeStyles.muted }}
                >
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
  )
}

export default Evaluation
