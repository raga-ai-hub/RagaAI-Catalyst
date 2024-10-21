import React from 'react';
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Rewind, FastForward } from 'lucide-react';

interface TimelineControlsProps {
  zoom: number;
  setZoom: (zoom: number) => void;
  currentTime: number;
  setCurrentTime: (time: number) => void;
  totalDuration: number;
}

const TimelineControls: React.FC<TimelineControlsProps> = ({
  zoom,
  setZoom,
  currentTime,
  setCurrentTime,
  totalDuration,
}) => {
  const handleZoomChange = (value: number[]) => {
    setZoom(value[0]);
  };

  const handlePrevious = () => {
    const newTime = Math.max(0, currentTime - 10);
    setCurrentTime(newTime);
  };

  const handleNext = () => {
    const newTime = Math.min(totalDuration, currentTime + 10);
    setCurrentTime(newTime);
  };

  return (
    <div className="flex items-center justify-between mb-4 bg-gray-50 p-3 rounded-md">
      <div className="flex items-center space-x-4">
        <span className="text-sm font-medium">Zoom: {zoom}%</span>
        <div className="w-40">
          <Slider
            value={[zoom]}
            onValueChange={handleZoomChange}
            min={100}
            max={200}
            step={10}
            className="w-full"
          />
        </div>
      </div>
      <div className="flex items-center space-x-2">
        <Button variant="outline" size="sm" onClick={handlePrevious}>
          <Rewind className="h-4 w-4 mr-1" />
          Previous
        </Button>
        <Button variant="outline" size="sm" onClick={handleNext}>
          Next
          <FastForward className="h-4 w-4 ml-1" />
        </Button>
      </div>
    </div>
  );
};

export default TimelineControls;