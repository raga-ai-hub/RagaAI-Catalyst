import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Clock, CheckCircle2, DollarSign, Cpu } from 'lucide-react'

interface CriteriaItem {
  icon: React.ReactNode
  title: string
  description: string
  trend: string
  color: string
  bgColor: string
}

const criteria: CriteriaItem[] = [
  {
    icon: <CheckCircle2 className='w-6 h-6' />,
    title: 'Success Rate',
    description: 'Percentage of tasks completed successfully',
    trend: 'Higher is better',
    color: 'text-emerald-600 dark:text-emerald-400',
    bgColor: 'bg-emerald-50 dark:bg-emerald-950'
  },
  {
    icon: <Clock className='w-6 h-6' />,
    title: 'Response Time',
    description: 'Average time to complete tasks',
    trend: 'Lower is better',
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-50 dark:bg-blue-950'
  },
  {
    icon: <DollarSign className='w-6 h-6' />,
    title: 'Cost Efficiency',
    description: 'Total cost per successful operation',
    trend: 'Lower is better',
    color: 'text-purple-600 dark:text-purple-400',
    bgColor: 'bg-purple-50 dark:bg-purple-950'
  },
  {
    icon: <Cpu className='w-6 h-6' />,
    title: 'Memory Usage',
    description: 'Efficiency of system memory utilization',
    trend: 'Lower is better',
    color: 'text-amber-600 dark:text-amber-400',
    bgColor: 'bg-amber-50 dark:bg-amber-950'
  }
]

const TopPerformanceCriteria: React.FC = () => {
  return (
    <Card className='shadow-lg'>
      <CardHeader className='border-b border-border'>
        <CardTitle className='text-2xl font-bold text-foreground'>
          Top Performance Criteria
        </CardTitle>
      </CardHeader>
      <CardContent className='p-6'>
        <div className='grid grid-cols-1 md:grid-cols-2 gap-6'>
          {criteria.map((item, index) => (
            <div
              key={index}
              className='flex items-start space-x-4 p-4 rounded-lg border border-border bg-card hover:bg-accent/50 transition-all duration-200'
            >
              <div className={`${item.color} ${item.bgColor} p-2 rounded-lg`}>
                {item.icon}
              </div>
              <div className='flex-1'>
                <h3 className='font-semibold text-lg mb-1 text-foreground'>
                  {item.title}
                </h3>
                <p className='text-muted-foreground text-sm mb-2'>
                  {item.description}
                </p>
                <div className='flex items-center'>
                  <span
                    className={`text-xs font-medium px-2 py-1 rounded-full ${
                      item.trend.includes('Higher')
                        ? 'bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300'
                        : 'bg-blue-100 dark:bg-blue-950 text-blue-700 dark:text-blue-300'
                    }`}
                  >
                    {item.trend}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export default TopPerformanceCriteria
