import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts'
import { Loader2 } from 'lucide-react'
import { useProject } from '@/contexts/ProjectContext'
import { useTheme } from '@/theme/ThemeProvider'
import { fetchTraces, fetchAnalysisTrace } from '@/utils/api'

// Update Theme type to match ThemeProvider
type Theme = 'light' | 'dark' | 'system'
type VisualTheme = 'light' | 'dark'

interface TimeData {
  name: string
  size: number
  duration: string
  percentage: string
}

interface ColorConfig {
  gradient: string[]
  stroke: string
  fill: string
}

interface ColorConfigs {
  [key: string]: {
    light: ColorConfig
    dark: ColorConfig
  }
}

const COLORS: ColorConfigs = {
  'LLM Calls': {
    light: {
      gradient: ['#9333EA', '#A855F7'],
      stroke: '#7E22CE',
      fill: '#9333EA'
    },
    dark: {
      gradient: ['#A855F7', '#C084FC'],
      stroke: '#9333EA',
      fill: '#A855F7'
    }
  },
  'Tool Calls': {
    light: {
      gradient: ['#EA580C', '#FB923C'],
      stroke: '#C2410C',
      fill: '#EA580C'
    },
    dark: {
      gradient: ['#FB923C', '#FD9D50'],
      stroke: '#EA580C',
      fill: '#FB923C'
    }
  }
}

// Helper function to resolve theme
const resolveTheme = (theme: Theme): VisualTheme => {
  if (theme === 'system') {
    // Check system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light'
  }
  return theme
}

const TimeAnalysis: React.FC = () => {
  const { selectedProject, selectedTraceId } = useProject()
  const { theme } = useTheme()
  const [timeData, setTimeData] = useState<TimeData[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Resolve the theme for visual purposes
  const visualTheme = resolveTheme(theme)

  const formatDuration = (seconds: number): string => {
    if (seconds < 0.01) {
      return `${(seconds * 1000).toFixed(2)}ms`
    } else {
      return `${seconds.toFixed(2)}s`
    }
  }

  const fetchTimeData = async () => {
    if (!selectedProject) return

    try {
      setIsLoading(true)
      setError(null)

      let totalLLMTime = 0
      let totalToolTime = 0

      if (selectedTraceId) {
        const traceData = await fetchAnalysisTrace(selectedTraceId)

        totalLLMTime = traceData.llm_calls.reduce((sum: number, call: any) => {
          const duration = typeof call.duration === 'number' ? call.duration : 0
          return sum + duration
        }, 0)

        totalToolTime =
          traceData.tool_calls?.reduce((sum: number, call: any) => {
            const duration =
              typeof call.duration === 'number' ? call.duration : 0
            return sum + duration
          }, 0) || 0
      } else {
        const traces = await fetchTraces(selectedProject)

        for (const trace of traces) {
          const traceData = await fetchAnalysisTrace(trace.id)

          totalLLMTime += traceData.llm_calls.reduce(
            (sum: number, call: any) => {
              const duration =
                typeof call.duration === 'number' ? call.duration : 0
              return sum + duration
            },
            0
          )

          totalToolTime +=
            traceData.tool_calls?.reduce((sum: number, call: any) => {
              const duration =
                typeof call.duration === 'number' ? call.duration : 0
              return sum + duration
            }, 0) || 0
        }
      }

      const totalTime = totalLLMTime + totalToolTime

      const data: TimeData[] = []

      if (totalLLMTime > 0) {
        data.push({
          name: 'LLM Calls',
          size: totalLLMTime,
          duration: `Duration: ${formatDuration(totalLLMTime)}`,
          percentage: `${((totalLLMTime / totalTime) * 100).toFixed(1)}%`
        })
      }

      if (totalToolTime > 0) {
        data.push({
          name: 'Tool Calls',
          size: totalToolTime,
          duration: `Duration: ${formatDuration(totalToolTime)}`,
          percentage: `${((totalToolTime / totalTime) * 100).toFixed(1)}%`
        })
      }

      setTimeData(data)
    } catch (err) {
      console.error('Error fetching time data:', err)
      setError('Failed to fetch time data. Please try again later.')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (selectedProject) {
      fetchTimeData()
    }
  }, [selectedProject, selectedTraceId])

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className='flex items-center justify-center h-64'>
          <Loader2 className='w-8 h-8 animate-spin text-primary' />
          <span className='ml-2'>Loading time analysis data...</span>
        </div>
      )
    }

    if (error) {
      return (
        <div className='text-center text-destructive p-4'>
          <p>{error}</p>
          <button
            className='mt-2 px-4 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90'
            onClick={fetchTimeData}
          >
            Retry
          </button>
        </div>
      )
    }

    if (!selectedProject) {
      return (
        <div className='text-center p-4 text-muted-foreground'>
          Please select a project to view analytics
        </div>
      )
    }

    if (timeData.length === 0) {
      return (
        <div className='text-center p-4 text-muted-foreground'>
          No time analysis data available
        </div>
      )
    }

    return (
      <div className='h-[400px]'>
        <ResponsiveContainer width='100%' height='100%'>
          <Treemap
            data={[{ name: 'Time Analysis', children: timeData }]}
            dataKey='size'
            stroke='#fff'
            content={<CustomizedContent colors={COLORS} theme={visualTheme} />}
          >
            <Tooltip content={<CustomTooltip theme={visualTheme} />} />
          </Treemap>
        </ResponsiveContainer>
      </div>
    )
  }

  return (
    <Card className='w-full max-w-4xl mx-auto border'>
      <CardHeader>
        <CardTitle>Time Analysis</CardTitle>
      </CardHeader>
      <CardContent>{renderContent()}</CardContent>
    </Card>
  )
}

interface CustomizedContentProps {
  root?: any
  depth?: number
  x?: number
  y?: number
  width?: number
  height?: number
  index?: number
  colors?: ColorConfigs
  theme: VisualTheme
  name?: string
}

const CustomizedContent: React.FC<CustomizedContentProps> = ({
  root,
  x = 0,
  y = 0,
  width = 0,
  height = 0,
  index = 0,
  colors = {},
  theme,
  name = ''
}) => {
  const data = root?.children?.[index]
  if (!data) return null

  const BORDER_RADIUS = 8

  const path = `
    M${x + BORDER_RADIUS},${y}
    h${width - 2 * BORDER_RADIUS}
    a${BORDER_RADIUS},${BORDER_RADIUS} 0 0 1 ${BORDER_RADIUS},${BORDER_RADIUS}
    v${height - 2 * BORDER_RADIUS}
    a${BORDER_RADIUS},${BORDER_RADIUS} 0 0 1 -${BORDER_RADIUS},${BORDER_RADIUS}
    h-${width - 2 * BORDER_RADIUS}
    a${BORDER_RADIUS},${BORDER_RADIUS} 0 0 1 -${BORDER_RADIUS},-${BORDER_RADIUS}
    v-${height - 2 * BORDER_RADIUS}
    a${BORDER_RADIUS},${BORDER_RADIUS} 0 0 1 ${BORDER_RADIUS},-${BORDER_RADIUS}
  `

  const colorConfig = colors[name]?.[theme]

  return (
    <g>
      <path
        d={path}
        style={{
          fill: colorConfig?.fill || '#gray',
          stroke: colorConfig?.stroke || '#fff',
          strokeWidth: 2,
          strokeOpacity: 1
        }}
      />
      <text
        x={x + width / 2}
        y={y + height / 2 - 12}
        textAnchor='middle'
        fill='#fff'
        fontSize={16}
        fontWeight='bold'
      >
        {name}
      </text>
      <text
        x={x + width / 2}
        y={y + height / 2 + 12}
        textAnchor='middle'
        fill='#fff'
        fontSize={14}
      >
        {data.duration}
      </text>
      <text
        x={x + width / 2}
        y={y + height / 2 + 36}
        textAnchor='middle'
        fill='#fff'
        fontSize={14}
      >
        {data.percentage}
      </text>
    </g>
  )
}

interface CustomTooltipProps {
  active?: boolean
  payload?: any[]
  theme: VisualTheme
}

const CustomTooltip: React.FC<CustomTooltipProps> = ({
  active,
  payload,
  theme
}) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div
        className={`${
          theme === 'dark'
            ? 'bg-card border-border'
            : 'bg-white border-slate-200'
        } p-4 border rounded-lg shadow`}
      >
        <p className='font-bold text-lg mb-1 text-foreground'>{data.name}</p>
        <p className='text-muted-foreground'>{data.duration}</p>
        <p className='text-muted-foreground'>Percentage: {data.percentage}</p>
      </div>
    )
  }
  return null
}

export default TimeAnalysis
