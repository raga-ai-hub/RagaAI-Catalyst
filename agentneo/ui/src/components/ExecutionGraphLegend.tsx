import React from 'react'
import { Panel } from 'reactflow'
import { nodeStylesByType } from './ExecutionGraphNodes'
import { useTheme } from '@/theme/ThemeProvider'

export const Legend = () => {
  const { theme } = useTheme()

  return (
    <Panel position='top-left' className={`p-2 rounded m-2 bg-background border border-border shadow-md`}>
      <div className='text-sm font-semibold mb-2 text-foreground'>Node Types</div>
      <div className='grid grid-cols-2 gap-x-4 gap-y-1'>
        {Object.entries(nodeStylesByType).map(([type, style]) => (
          <div key={type} className='flex items-center'>
            <div
              className={`w-3 h-3 rounded mr-2 ${style.className}`}
              style={{ backgroundColor: style.backgroundColor }}
            />
            <span className='capitalize text-xs text-foreground'>{type}</span>
          </div>
        ))}
      </div>
    </Panel>
  )
}
