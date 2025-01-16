import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { SidebarProvider } from './contexts/SidebarContext'
import { ProjectProvider } from './contexts/ProjectContext'
import Overview from './pages/Overview'
import Analysis from './pages/Analysis'
import TraceHistory from './pages/TraceHistory'
import Evaluation from './pages/Evaluation'
import { ThemeProvider } from '@/theme/ThemeProvider'

function App () {
  return (
    <ThemeProvider defaultTheme='system' storageKey='vite-ui-theme'>
      <ProjectProvider>
        <SidebarProvider>
          <Router>
            <Routes>
              <Route path='/' element={<Overview />} />
              <Route path='/analysis' element={<Analysis />} />
              <Route path='/trace-history' element={<TraceHistory />} />
              <Route path='/evaluation' element={<Evaluation />} />
            </Routes>
          </Router>
        </SidebarProvider>
      </ProjectProvider>
    </ThemeProvider>
  )
}

export default App
