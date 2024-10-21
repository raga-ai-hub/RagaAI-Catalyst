import React from 'react';
import { TimelineData } from '../types/timeline';

interface TimelineDetailsProps {
  selectedEvent: TimelineData | null;
}

const TimelineDetails: React.FC<TimelineDetailsProps> = ({ selectedEvent }) => {
  if (!selectedEvent) {
    return <div className="text-gray-500">Select an event to see details</div>;
  }

  const startTime = new Date(selectedEvent.startTime);
  const endTime = new Date(selectedEvent.endTime);
  const duration = (endTime.getTime() - startTime.getTime()) / 1000;

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Event Details</h3>
      <div className="space-y-2">
        <p><strong>Name:</strong> {selectedEvent.name}</p>
        <p><strong>Type:</strong> {selectedEvent.type}</p>
        <p><strong>Start Time:</strong> {startTime.toLocaleString()}</p>
        <p><strong>End Time:</strong> {endTime.toLocaleString()}</p>
        <p><strong>Duration:</strong> {duration.toFixed(2)}s</p>
      </div>
      {selectedEvent.details && (
        <div className="space-y-2">
          <h4 className="text-md font-semibold">Additional Details</h4>
          {selectedEvent.details.agent && <p><strong>Agent:</strong> {selectedEvent.details.agent}</p>}
          {selectedEvent.details.function && <p><strong>Function:</strong> {selectedEvent.details.function}</p>}
          {selectedEvent.details.model && <p><strong>Model:</strong> {selectedEvent.details.model}</p>}
          {selectedEvent.details.token_usage && <p><strong>Token Usage:</strong> {selectedEvent.details.token_usage}</p>}
          {selectedEvent.details.cost && <p><strong>Cost:</strong> {selectedEvent.details.cost}</p>}
          {selectedEvent.details.input && (
            <div>
              <strong>Input:</strong>
              <pre className="mt-1 bg-gray-100 p-2 rounded text-sm overflow-x-auto">{selectedEvent.details.input}</pre>
            </div>
          )}
          {selectedEvent.details.output && (
            <div>
              <strong>Output:</strong>
              <pre className="mt-1 bg-gray-100 p-2 rounded text-sm overflow-x-auto">{selectedEvent.details.output}</pre>
            </div>
          )}
          {selectedEvent.details.error_message && (
            <div>
              <strong>Error Message:</strong>
              <pre className="mt-1 bg-gray-100 p-2 rounded text-sm overflow-x-auto text-red-500">{selectedEvent.details.error_message}</pre>
            </div>
          )}
          {selectedEvent.details.interaction_type && (
            <p><strong>Interaction Type:</strong> {selectedEvent.details.interaction_type}</p>
          )}
          {selectedEvent.details.content && (
            <div>
              <strong>Content:</strong>
              <pre className="mt-1 bg-gray-100 p-2 rounded text-sm overflow-x-auto">{selectedEvent.details.content}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TimelineDetails;