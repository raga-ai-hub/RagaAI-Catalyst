import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

interface EvaluationDetailsProps {
  traceId: string;
}

const EvaluationDetails: React.FC<EvaluationDetailsProps> = ({ traceId }) => {
  // Dummy evaluation data (expanded with more details)
  const evaluationData = {
    toolSelection: { 
      score: 0.92, 
      reason: 'Appropriate tools selected for the task.',
      details: 'The agent correctly identified and utilized the database query tool to fetch the required information. This demonstrates a good understanding of the available tools and their applicability to the given task.',
      recommendations: 'Consider exploring more specialized tools for complex queries to further improve efficiency.'
    },
    toolUsage: { 
      score: 0.88, 
      reason: 'Efficient use of selected tools.',
      details: 'The agent constructed well-formed queries and utilized tool features effectively. There was a slight inefficiency in the query structure that could have been optimized for better performance.',
      recommendations: 'Review and optimize query structures for more complex operations. Consider caching frequently accessed data.'
    },
    goalDecomposition: { 
      score: 0.78, 
      reason: 'Good breakdown of complex task.',
      details: 'The agent divided the main goal into subtasks: understanding the request, formulating the query, executing the query, and presenting the results. However, there was room for improvement in further decomposing the query formulation step.',
      recommendations: 'Implement a more granular approach to task decomposition, especially for complex queries or multi-step processes.'
    },
    planAdaptability: { 
      score: 0.85, 
      reason: 'Flexible adaptation to changing requirements.',
      details: 'When faced with unexpected data formats, the agent quickly adjusted its approach to parse and present the information correctly. This demonstrates good adaptability, although the initial plan could have anticipated potential variations better.',
      recommendations: 'Enhance the initial planning phase to consider a wider range of potential scenarios and data variations.'
    },
    executionError: { 
      score: 0.95, 
      reason: 'Minimal errors during execution.',
      details: 'The agent successfully completed the task with only minor issues in data formatting. No critical errors were encountered, and the final output was accurate and useful.',
      recommendations: 'Implement more robust error handling and data validation to catch and address minor formatting issues automatically.'
    },
  };

  return (
    <div className="space-y-4">
      {Object.entries(evaluationData).map(([key, value]) => (
        <Card key={key}>
          <CardHeader>
            <CardTitle>{key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center mb-2">
              <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 mr-2">
                <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${value.score * 100}%` }}></div>
              </div>
              <span className="text-sm font-medium text-gray-500 dark:text-gray-400">{(value.score * 100).toFixed(0)}%</span>
            </div>
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="item-1">
                <AccordionTrigger>View Details</AccordionTrigger>
                <AccordionContent>
                  <p className="text-sm text-gray-600 dark:text-gray-300 mb-2"><strong>Reason:</strong> {value.reason}</p>
                  <p className="text-sm text-gray-600 dark:text-gray-300 mb-2"><strong>Details:</strong> {value.details}</p>
                  <p className="text-sm text-gray-600 dark:text-gray-300"><strong>Recommendations:</strong> {value.recommendations}</p>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

export default EvaluationDetails;