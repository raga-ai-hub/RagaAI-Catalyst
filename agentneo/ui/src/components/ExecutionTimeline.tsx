import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import TimelineControls from './TimelineControls';
import TimelineDetails from './TimelineDetails';
import TimelineLegend from './TimelineLegend';
import TimelineRows from './TimelineRows';
import TimelineContent from './TimelineContent';
import { TimelineData } from '../types/timeline';
import { fetchTimelineData } from '../api/traceApi';
import { fetchTracesForProject } from '../utils/databaseUtils';
import { useQuery } from '@tanstack/react-query';
import { useProject } from '../contexts/ProjectContext';


const ExecutionTimeline: React.FC = () => {
  const [selectedEvent, setSelectedEvent] = useState<TimelineData | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [zoom, setZoom] = useState(100);
  const timelineRef = useRef<HTMLDivElement>(null);
  const { selectedProject } = useProject();
  const [selectedTraceId, setSelectedTraceId] = useState<string>('');

  const { data: traces = [] } = useQuery({
    queryKey: ['traces', selectedProject],
    queryFn: () => fetchTracesForProject(selectedProject || 0),
    enabled: !!selectedProject,
  });

  const { data: timelineData = [], isLoading } = useQuery({
    queryKey: ['timelineData', selectedProject, selectedTraceId],
    queryFn: () => fetchTimelineData(selectedProject?.toString() || ''),
    enabled: !!selectedProject && !!selectedTraceId,
  });

  useEffect(() => {
    if (traces.length > 0 && !selectedTraceId) {
      setSelectedTraceId(traces[0].id);
    }
  }, [traces, selectedTraceId]);

  const totalDuration = timelineData.length > 0 ?
    Math.max(...timelineData.map(event => new Date(event.end_time).getTime() - new Date(event.start_time).getTime())) / 1000 : 0;

  const rows = ['LLM', 'Action', 'Tool', 'Error', 'UserInteraction'];

  const handleEventClick = (event: TimelineData) => {
    setSelectedEvent(event);
    setCurrentTime((new Date(event.startTime).getTime() - new Date(timelineData[0].startTime).getTime()) / 1000);
  };

  useEffect(() => {
    if (timelineRef.current) {
      const timelineWidth = timelineRef.current.offsetWidth;
      const scrollPosition = (currentTime / totalDuration) * timelineWidth * (zoom / 100) - timelineWidth / 2;
      timelineRef.current.scrollLeft = scrollPosition;
    }
  }, [currentTime, zoom, totalDuration]);

  if (isLoading) {
    return <div>Loading...</div>;
  }

  return (
    <Card className="mt-8 bg-white shadow-md">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-2xl font-bold">Execution Timeline</CardTitle>
        <Select value={selectedTraceId} onValueChange={setSelectedTraceId}>
          <SelectTrigger className="w-[200px]">
            <SelectValue placeholder="Select Trace" />
          </SelectTrigger>
          <SelectContent>
            {traces.map((trace) => (
              <SelectItem key={trace.id} value={trace.id}>
                {trace.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent>
        <div className="flex">
          <div className="w-3/4 pr-4">
            <TimelineControls
              zoom={zoom}
              setZoom={setZoom}
              currentTime={currentTime}
              setCurrentTime={setCurrentTime}
              totalDuration={totalDuration}
            />
            <TimelineLegend timelineData={timelineData} />
            <div className="flex">
              <TimelineRows rows={rows} />
              <TimelineContent
                timelineRef={timelineRef}
                zoom={zoom}
                rows={rows}
                timelineData={timelineData}
                totalDuration={totalDuration}
                currentTime={currentTime}
                handleEventClick={handleEventClick}
              />
            </div>
          </div>
          <div className="w-1/4 pl-4 border-l border-gray-200">
            <TimelineDetails selectedEvent={selectedEvent} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ExecutionTimeline;