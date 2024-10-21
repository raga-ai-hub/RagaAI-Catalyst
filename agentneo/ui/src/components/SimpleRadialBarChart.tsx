import React from 'react';
import { RadialBarChart, RadialBar, Legend, ResponsiveContainer } from 'recharts';

interface DataItem {
  name: string;
  value: number;
  fill: string;
}

interface SimpleRadialBarChartProps {
  data: DataItem[];
}

const SimpleRadialBarChart: React.FC<SimpleRadialBarChartProps> = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <RadialBarChart 
        cx="50%" 
        cy="50%" 
        innerRadius="10%" 
        outerRadius="80%" 
        barSize={10} 
        data={data}
      >
        <RadialBar
          label={{ position: 'insideStart', fill: '#fff' }}
          background
          dataKey="value"
        />
        <Legend iconSize={10} layout="vertical" verticalAlign="middle" align="right" />
      </RadialBarChart>
    </ResponsiveContainer>
  );
};

export default SimpleRadialBarChart;