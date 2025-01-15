import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend
} from 'recharts'
import { Loader2 } from 'lucide-react'
import { useProject } from '@/contexts/ProjectContext'
import { useTheme } from '@/theme/ThemeProvider'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { fetchTraces, fetchAnalysisTrace } from '@/utils/api'

interface TraceData {
  id: string
  name: string
  avgResponseTime: number
  totalTokens: number
  totalCost: number
  avgMemoryUsage: number
  toolCallCount: number
  llmCallCount: number
}

// Color schemes for light and dark themes
const THEME_COLORS = {
  light: [
    {
      gradient: ['#3b82f6', '#60a5fa'],
      stroke: '#2563eb',
      fill: 'url(#gradient1)'
    },
    {
      gradient: ['#10b981', '#34d399'],
      stroke: '#059669',
      fill: 'url(#gradient2)'
    },
    {
      gradient: ['#f59e0b', '#fbbf24'],
      stroke: '#d97706',
      fill: 'url(#gradient3)'
    },
    {
      gradient: ['#8b5cf6', '#a78bfa'],
      stroke: '#7c3aed',
      fill: 'url(#gradient4)'
    },
    {
      gradient: ['#ec4899', '#f472b6'],
      stroke: '#db2777',
      fill: 'url(#gradient5)'
    }
  ],
  dark: [
    {
      gradient: ['#60a5fa', '#3b82f6'],
      stroke: '#93c5fd',
      fill: 'url(#gradient1)'
    },
    {
      gradient: ['#34d399', '#10b981'],
      stroke: '#6ee7b7',
      fill: 'url(#gradient2)'
    },
    {
      gradient: ['#fbbf24', '#f59e0b'],
      stroke: '#fcd34d',
      fill: 'url(#gradient3)'
    },
    {
      gradient: ['#a78bfa', '#8b5cf6'],
      stroke: '#c4b5fd',
      fill: 'url(#gradient4)'
    },
    {
      gradient: ['#f472b6', '#ec4899'],
      stroke: '#f9a8d4',
      fill: 'url(#gradient5)'
    }
  ]
}

const METRICS = [
  {
    key: 'avgResponseTime',
    label: 'Average Response Time per Call',
    unit: 's',
    format: (value: number) => `${value.toFixed(2)}s`
  },
  {
    key: 'totalTokens',
    label: 'Total Tokens Used',
    unit: '',
    format: (value: number) => value.toFixed(0)
  },
  {
    key: 'totalCost',
    label: 'Total Cost',
    unit: '$',
    format: (value: number) => `$${value.toFixed(4)}`
  },
  {
    key: 'avgMemoryUsage',
    label: 'Average Memory Usage per Call',
    unit: 'MB',
    format: (value: number) => `${(value / (1024 * 1024)).toFixed(2)} MB`
  },
  {
    key: 'toolCallCount',
    label: 'Number of Tool Calls',
    unit: '',
    format: (value: number) => value.toFixed(0)
  },
  {
    key: 'llmCallCount',
    label: 'Number of LLM Calls',
    unit: '',
    format: (value: number) => value.toFixed(0)
  }
]

const TracePerformanceComparison: React.FC = () => {
  const { selectedProject } = useProject()
  const { theme } = useTheme()
  const [traces, setTraces] = useState<TraceData[]>([])
  const [selectedTraces, setSelectedTraces] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showDropdown, setShowDropdown] = useState(false)
  const [showPercentage, setShowPercentage] = useState(true)

  const COLORS = THEME_COLORS[theme === 'dark' ? 'dark' : 'light']

  const fetchTraceData = async () => {
    if (!selectedProject) return

    try {
      setIsLoading(true)
      setError(null)

      const tracesData = await fetchTraces(selectedProject)

      const processedTraces = await Promise.all(
        tracesData.map(async (trace: any) => {
          const traceData = await fetchAnalysisTrace(trace.id)

          let totalTokens = 0
          let totalCost = 0
          let totalDuration = 0
          let totalMemory = 0

          traceData.llm_calls.forEach((call: any) => {
            const tokenUsage = JSON.parse(call.token_usage)
            totalTokens +=
              tokenUsage.input + tokenUsage.completion + tokenUsage.reasoning

            const cost = JSON.parse(call.cost)
            totalCost += cost.input + cost.output + cost.reasoning

            totalDuration += call.duration
            totalMemory += call.memory_used || 0
          })

          return {
            id: trace.id,
            name: trace.name || `Trace ${trace.id}`,
            avgResponseTime: totalDuration / traceData.llm_calls.length,
            totalTokens,
            totalCost,
            avgMemoryUsage: totalMemory / traceData.llm_calls.length,
            toolCallCount: traceData.tool_calls?.length || 0,
            llmCallCount: traceData.llm_calls.length
          }
        })
      )

      setTraces(processedTraces)
      if (processedTraces.length > 0 && selectedTraces.length === 0) {
        setSelectedTraces([processedTraces[0].id])
      }
    } catch (err) {
      console.error('Error fetching trace data:', err)
      setError('Failed to fetch trace data. Please try again later.')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (selectedProject) {
      fetchTraceData()
    }
  }, [selectedProject])

  const handleTraceSelect = (traceId: string) => {
    setSelectedTraces(prev => {
      if (prev.includes(traceId)) {
        return prev.filter(id => id !== traceId)
      }
      return [...prev, traceId]
    })
  }

  const getComparisonData = () => {
    const selectedTraceData = traces.filter(trace =>
      selectedTraces.includes(trace.id)
    )
    return METRICS.map(({ key, label, unit }) => {
      const metricData: any = {
        metric: label,
        unit
      }

      const values: number[] = selectedTraceData.map(trace =>
        Number(trace[key as keyof TraceData])
      )
      const total = values.reduce((sum, val) => sum + val, 0)

      selectedTraceData.forEach(trace => {
        const value = Number(trace[key as keyof TraceData])
        metricData[trace.name] = showPercentage ? (value / total) * 100 : value
      })

      return metricData
    })
  }

  const formatValue = (value: number, metric: any) => {
    if (showPercentage) return `${value.toFixed(1)}%`
    return metric.format(value)
  }

  const renderTraceSelector = () => (
    <div className='w-full relative'>
      <label className='text-base font-semibold mb-2 block'>
        Select Traces to Compare
      </label>
      <div className='relative'>
        <div
          className={`w-full px-4 py-2 text-left border rounded-md flex justify-between items-center
            ${
              theme === 'dark'
                ? 'bg-gray-800 border-gray-700 hover:bg-gray-700'
                : 'bg-white border-gray-200 hover:bg-gray-50'
            }
            cursor-pointer`}
          onClick={() => setShowDropdown(!showDropdown)}
        >
          <span>
            {selectedTraces.length === 0
              ? 'Select traces...'
              : `${selectedTraces.length} trace${
                  selectedTraces.length === 1 ? '' : 's'
                } selected`}
          </span>
          <span className='ml-2'>▼</span>
        </div>
        {showDropdown && (
          <div
            className={`absolute z-10 w-full mt-1 border rounded-md shadow-lg max-h-[300px] overflow-auto
            ${
              theme === 'dark'
                ? 'bg-gray-800 border-gray-700'
                : 'bg-white border-gray-200'
            }`}
          >
            {traces.map(trace => (
              <label
                key={trace.id}
                className={`flex items-center px-4 py-2 cursor-pointer
                  ${
                    theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
                  }`}
                onClick={e => e.stopPropagation()}
              >
                <input
                  type='checkbox'
                  checked={selectedTraces.includes(trace.id)}
                  onChange={() => handleTraceSelect(trace.id)}
                  className='mr-2 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500'
                />
                <span className='select-none'>{trace.name}</span>
              </label>
            ))}
          </div>
        )}
      </div>
      <p
        className={`text-sm mt-1 ${
          theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
        }`}
      >
        Select multiple traces to compare their performance metrics
      </p>
    </div>
  )

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className='flex items-center justify-center h-64'>
          <Loader2 className='w-8 h-8 animate-spin text-blue-500' />
          <span className='ml-2'>Loading trace data...</span>
        </div>
      )
    }

    if (error) {
      return (
        <div className='text-center text-red-500 p-4'>
          <p>{error}</p>
          <button
            className='mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600'
            onClick={fetchTraceData}
          >
            Retry
          </button>
        </div>
      )
    }

    if (!selectedProject) {
      return (
        <div className='text-center p-4'>
          Please select a project to view comparison
        </div>
      )
    }

    return (
      <div className='space-y-8 w-full'>
        {renderTraceSelector()}

        <div className='flex items-center space-x-2'>
          <Switch
            id='trace-percentage-mode'
            checked={showPercentage}
            onCheckedChange={setShowPercentage}
          />
          <Label htmlFor='trace-percentage-mode'>Show as percentage</Label>
        </div>

        {selectedTraces.length > 0 && (
          <div
            className={`space-y-6 rounded-lg p-6 shadow-sm w-full
            ${
              theme === 'dark'
                ? 'bg-gray-800 border border-gray-700'
                : 'bg-white border border-gray-200'
            }`}
          >
            <div className='h-[500px] w-full'>
              <ResponsiveContainer width='100%' height='100%'>
                <BarChart
                  data={getComparisonData()}
                  layout='vertical'
                  margin={{ top: 20, right: 50, left: 200, bottom: 5 }}
                >
                  {COLORS.map((color, index) => (
                    <defs key={`gradient-${index}`}>
                      <linearGradient
                        id={`gradient${index + 1}`}
                        x1='0'
                        y1='0'
                        x2='0'
                        y2='1'
                      >
                        <stop offset='0%' stopColor={color.gradient[0]} />
                        <stop offset='100%' stopColor={color.gradient[1]} />
                      </linearGradient>
                    </defs>
                  ))}
                  <CartesianGrid
                    strokeDasharray='3 3'
                    stroke={theme === 'dark' ? '#374151' : '#e5e7eb'}
                  />
                  <XAxis
                    type='number'
                    domain={showPercentage ? [0, 100] : [0, 'auto']}
                    tickFormatter={value =>
                      showPercentage ? `${value}%` : value
                    }
                    stroke={theme === 'dark' ? '#9ca3af' : '#4b5563'}
                  />
                  <YAxis
                    dataKey='metric'
                    type='category'
                    width={180}
                    style={{ fontSize: '12px' }}
                    stroke={theme === 'dark' ? '#9ca3af' : '#4b5563'}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: theme === 'dark' ? '#1f2937' : '#ffffff',
                      border: `1px solid ${
                        theme === 'dark' ? '#374151' : '#e5e7eb'
                      }`,
                      color: theme === 'dark' ? '#ffffff' : '#000000'
                    }}
                    formatter={(value: number, name: string, props: any) => {
                      const metric = METRICS.find(
                        m => m.label === props.payload.metric
                      )
                      return metric ? formatValue(value, metric) : value
                    }}
                  />
                  <Legend
                    wrapperStyle={{
                      color: theme === 'dark' ? '#ffffff' : '#000000'
                    }}
                  />
                  {selectedTraces.map((traceId, index) => {
                    const trace = traces.find(t => t.id === traceId)
                    return (
                      <Bar
                        key={traceId}
                        dataKey={trace?.name || ''}
                        fill={COLORS[index % COLORS.length].fill}
                        stroke={COLORS[index % COLORS.length].stroke}
                      />
                    )
                  })}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    )
  }

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (showDropdown && !(event.target as Element).closest('.relative')) {
        setShowDropdown(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [showDropdown])

  return (
    <div className='w-full'>
      <Card className='w-full'>
        <CardHeader>
          <CardTitle>Trace Performance Comparison</CardTitle>
        </CardHeader>
        <CardContent>{renderContent()}</CardContent>
      </Card>
    </div>
  )
}

export default TracePerformanceComparison
