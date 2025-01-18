import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Terminal, Monitor, Cpu, HardDrive, Microchip, Database } from 'lucide-react';

interface SystemData {
  pythonVersion: string;
  os: string;
  cpu: string;
  gpu: string;
  totalMemory: string;
  diskSpace: string;
}

interface SystemInformationProps {
  systemData: SystemData | null;
}

const SystemInformation: React.FC<SystemInformationProps> = React.memo(({ systemData }) => {
  if (!systemData) {
    return <div>Loading system information...</div>;
  }

  const formatDiskSpace = (diskSpace) => {
    if (typeof diskSpace === 'string') {
      try {
        const parsed = JSON.parse(diskSpace);
        const available = (parsed.available).toFixed(2);
        const total = (parsed.total).toFixed(2);
        return `${total} GB (${available} GB free)`;
      } catch (e) {
        return diskSpace; // Return original string if parsing fails
      }
    }
    return diskSpace; // Return as-is if it's not a string
  };

  const formatMemory = (memory: string) => {
    if (!memory) return 'N/A';
    const memoryNumber = parseFloat(memory);
    return `${memoryNumber.toFixed(2)} GB`;
  };

  const formatGPU = (gpu: string) => {
    if (!gpu || gpu.toLowerCase() === 'none' || gpu.trim() === '') {
      return 'N/A';
    }
    return gpu;
  };

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="text-xl font-semibold">System Information</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <InfoItem icon={<Terminal className="w-5 h-5 text-green-600" />} label="Python Version" value={systemData.pythonVersion} />
          <InfoItem icon={<Monitor className="w-5 h-5 text-blue-600" />} label="OS" value={systemData.os} />
          <InfoItem icon={<Cpu className="w-5 h-5 text-purple-600" />} label="CPU" value={systemData.cpu} />
          <InfoItem icon={<Microchip className="w-5 h-5 text-red-600" />} label="GPU" value={formatGPU(systemData.gpu)} />
          <InfoItem icon={<HardDrive className="w-5 h-5 text-yellow-600" />} label="Total Memory" value={formatMemory(systemData.totalMemory)} />
          <InfoItem
            icon={<Database className="w-5 h-5 text-indigo-600" />}
            label="Disk Space"
            value={formatDiskSpace(systemData.diskSpace)}
          />
        </div>
      </CardContent>
    </Card>
  );
});


interface InfoItemProps {
  icon: React.ReactNode;
  label: string;
  value: string;
}

const InfoItem: React.FC<InfoItemProps> = ({ icon, label, value }) => (
  <div className="flex items-center space-x-3">
    {icon}
    <div className="flex-1">
      <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{label}</p>
      <p className="text-base font-medium">{value}</p>
    </div>
  </div>
);

export default SystemInformation;