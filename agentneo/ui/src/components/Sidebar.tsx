import React from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  ChevronLeft,
  ChevronRight,
  LayoutDashboard,
  PieChart,
  History,
  ClipboardCheck,
  BookOpen,
  Github
} from 'lucide-react'
import { useSidebar } from '../contexts/SidebarContext'
import { SidebarLink, SidebarExternalLink } from './SidebarComponents'
import { ModeToggle } from '@/components/ModeToggle'
import { useTheme } from '../theme/ThemeProvider'

const Sidebar: React.FC = () => {
  const { isCollapsed, toggleSidebar } = useSidebar()
  const { theme } = useTheme()

  // Define theme-based styles that match the dashboard
  const sidebarStyles = {
    background:
      theme === 'dark'
        ? 'linear-gradient(to bottom, #2D1B69, #471069, #591069)'
        : 'linear-gradient(to bottom right, rgb(79, 70, 229), rgb(147, 51, 234), rgb(236, 72, 153))',
    color: '#ffffff',
    selectedBackground:
      theme === 'dark' ? 'rgba(139, 92, 246, 0.2)' : 'rgba(255, 255, 255, 0.2)'
  }

  return (
    <aside
      className={`relative overflow-visible flex flex-col transition-all duration-300 ${
        isCollapsed ? 'w-sidebar-collapsed' : 'w-sidebar-expanded'
      } fixed h-full border-r border-white border-opacity-20`}
      style={{
        background: sidebarStyles.background
      }}
    >
      {/* Overlay for depth effect */}
      <div className="absolute inset-0 bg-black opacity-10"></div>
      <div className="absolute inset-0 backdrop-blur-sm bg-white bg-opacity-5"></div>
      <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white to-transparent opacity-5"></div>

      <div className='relative z-10 flex-grow p-6'>
        <div className='flex items-center justify-between mb-8'>
          <div className='flex items-center'>
            <img
              src='/assets/logo.png'
              alt='RagaAI Logo'
              className={`w-10 h-10 ${isCollapsed ? 'mx-auto' : 'mr-3'}`}
            />
            {!isCollapsed && (
              <div>
                <h1 className='text-2xl font-bold text-white'>AgentNeo</h1>
                <p className='text-sm text-white/80 -mt-1'>
                  by{' '}
                  <a
                    href='https://raga.ai'
                    target='_blank'
                    rel='noopener noreferrer'
                    className='hover:underline text-white'
                  >
                    RagaAI
                  </a>
                </p>
              </div>
            )}
            {!isCollapsed && <ModeToggle />}
          </div>
        </div>

        <ScrollArea className='h-[calc(100vh-16rem)]'>
          <nav>
            <ul className='space-y-4'>
              <SidebarLink
                to='/'
                icon={LayoutDashboard}
                text='Overview'
                isCollapsed={isCollapsed}
              />
              <SidebarLink
                to='/analysis'
                icon={PieChart}
                text='Analytics'
                isCollapsed={isCollapsed}
              />
              <SidebarLink
                to='/trace-history'
                icon={History}
                text='Trace History'
                isCollapsed={isCollapsed}
              />
              <SidebarLink
                to='/evaluation'
                icon={ClipboardCheck}
                text='Evaluation'
                isCollapsed={isCollapsed}
              />
            </ul>
          </nav>
        </ScrollArea>
      </div>

      <div className='text-white relative z-10 mt-auto p-6 border-t border-white/10'>
        <ul className='space-y-4'>
          <SidebarExternalLink
            href='https://docs.raga.ai/agentneo'
            icon={BookOpen}
            text='Documentation'
            isCollapsed={isCollapsed}
          />
          <SidebarExternalLink
            href='https://github.com/raga-ai-hub/agentneo'
            icon={Github}
            text='Code Repository'
            isCollapsed={isCollapsed}
          />
        </ul>
      </div>

      <div className="absolute bottom-0 left-0 w-32 h-32 bg-white rounded-full opacity-5 -mb-16 -ml-16"></div>

      {/* Toggle button with matching gradient */}
      <button
        onClick={toggleSidebar}
        className='absolute -right-6 top-1/2 transform -translate-y-1/2 w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200 z-20 focus:outline-none'
        style={{
          background:
            theme === 'dark'
              ? 'linear-gradient(135deg, #2D1B69, #471069)'
              : 'linear-gradient(to bottom right, rgb(79, 70, 229), rgb(147, 51, 234), rgb(236, 72, 153))'
        }}
      >
        {isCollapsed ? (
          <ChevronRight className='h-6 w-6 relative z-10 text-white' />
        ) : (
          <ChevronLeft className='h-6 w-6 relative z-10 text-white' />
        )}
      </button>
    </aside>
  )
}

export default Sidebar