import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { DetailedTraceComponents } from '../utils/databaseUtils';

const TraceDetails: React.FC<{ traceData: DetailedTraceComponents }> = ({ traceData }) => {
    const { agents, tools, llmCalls, errors } = traceData;

    return (
        <div className="w-full h-full flex flex-col">
            <Tabs defaultValue="llm_calls" className="w-full flex-grow flex flex-col">
                <TabsList className="grid w-full grid-cols-4 mb-4">
                    <TabsTrigger value="llm_calls">LLM Calls ({llmCalls.length})</TabsTrigger>
                    <TabsTrigger value="tool_calls">Tool Calls ({tools.length})</TabsTrigger>
                    <TabsTrigger value="agent_calls">Agent Calls ({agents.length})</TabsTrigger>
                    <TabsTrigger value="errors">Errors ({errors.length})</TabsTrigger>
                </TabsList>
                <div className="flex-grow overflow-hidden">
                    <ScrollArea className="h-full w-full">
                        <div className="min-w-[600px] p-1">
                            <TabsContent value="llm_calls" className="h-full">
                                <div className="space-y-4">
                                    {llmCalls.map((call, index) => (
                                        <Card key={index} className="relative">
                                            <CardContent className="p-4">
                                                <div className="flex items-center mb-2">
                                                    <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold mr-3">
                                                        {index + 1}
                                                    </div>
                                                    <h3 className="text-lg font-semibold">{call.name}</h3>
                                                </div>
                                                <div className="grid grid-cols-2 gap-4 mb-4">
                                                    <div><span className="font-medium">Model:</span> {call.model}</div>
                                                    <div><span className="font-medium">Duration:</span> {call.duration?.toFixed(2)}s</div>
                                                    <div><span className="font-medium">Start Time:</span> {new Date(call.start_time).toLocaleString()}</div>
                                                    <div><span className="font-medium">End Time:</span> {new Date(call.end_time).toLocaleString()}</div>
                                                    <div><span className="font-medium">Token Usage:</span> {JSON.stringify(call.token_usage)}</div>
                                                    <div><span className="font-medium">Cost:</span> {JSON.stringify(call.cost)}</div>
                                                </div>
                                                <div className="mt-4">
                                                    <h4 className="font-medium mb-2">Input Prompt:</h4>
                                                    <ScrollArea className="h-40 w-full">
                                                        <pre className="bg-gray-100 p-2 rounded whitespace-pre-wrap">{call.input_prompt}</pre>
                                                    </ScrollArea>
                                                </div>
                                                <div className="mt-4">
                                                    <h4 className="font-medium mb-2">Output:</h4>
                                                    <ScrollArea className="h-40 w-full">
                                                        <pre className="bg-gray-100 p-2 rounded whitespace-pre-wrap">{call.output}</pre>
                                                    </ScrollArea>
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            </TabsContent>
                            <TabsContent value="tool_calls" className="h-full">
                                <div className="space-y-4">
                                    {tools.map((call, index) => (
                                        <Card key={index} className="relative">
                                            <CardContent className="p-4">
                                                <div className="flex items-center mb-2">
                                                    <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center text-white font-bold mr-3">
                                                        {index + 1}
                                                    </div>
                                                    <h3 className="text-lg font-semibold">{call.name}</h3>
                                                </div>
                                                <div className="grid grid-cols-2 gap-4 mb-4">
                                                    <div><span className="font-medium">Duration:</span> {call.duration?.toFixed(2)}s</div>
                                                    <div><span className="font-medium">Start Time:</span> {new Date(call.start_time).toLocaleString()}</div>
                                                    <div><span className="font-medium">End Time:</span> {new Date(call.end_time).toLocaleString()}</div>
                                                </div>
                                                <div className="mt-4">
                                                    <h4 className="font-medium mb-2">Input Parameters:</h4>
                                                    <ScrollArea className="h-40 w-full">
                                                        <pre className="bg-gray-100 p-2 rounded whitespace-pre-wrap">{JSON.stringify(call.input_parameters, null, 2)}</pre>
                                                    </ScrollArea>
                                                </div>
                                                <div className="mt-4">
                                                    <h4 className="font-medium mb-2">Output:</h4>
                                                    <ScrollArea className="h-40 w-full">
                                                        <pre className="bg-gray-100 p-2 rounded whitespace-pre-wrap">{JSON.stringify(call.output, null, 2)}</pre>
                                                    </ScrollArea>
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            </TabsContent>
                            <TabsContent value="agent_calls" className="h-full">
                                <div className="space-y-4">
                                    {agents.map((call, index) => (
                                        <Card key={index} className="relative">
                                            <CardContent className="p-4">
                                                <div className="flex items-center mb-2">
                                                    <div className="w-8 h-8 rounded-full bg-purple-500 flex items-center justify-center text-white font-bold mr-3">
                                                        {index + 1}
                                                    </div>
                                                    <h3 className="text-lg font-semibold">{call.name}</h3>
                                                </div>
                                                <div className="grid grid-cols-2 gap-4 mb-4">
                                                    <div><span className="font-medium">Start Time:</span> {new Date(call.start_time).toLocaleString()}</div>
                                                    <div><span className="font-medium">End Time:</span> {new Date(call.end_time).toLocaleString()}</div>
                                                </div>
                                                <div className="mt-4">
                                                    <h4 className="font-medium mb-2">LLM Call IDs:</h4>
                                                    <ScrollArea className="h-20 w-full">
                                                        <pre className="bg-gray-100 p-2 rounded whitespace-pre-wrap">{JSON.stringify(call.llm_call_ids, null, 2)}</pre>
                                                    </ScrollArea>
                                                </div>
                                                <div className="mt-4">
                                                    <h4 className="font-medium mb-2">Tool Call IDs:</h4>
                                                    <ScrollArea className="h-20 w-full">
                                                        <pre className="bg-gray-100 p-2 rounded whitespace-pre-wrap">{JSON.stringify(call.tool_call_ids, null, 2)}</pre>
                                                    </ScrollArea>
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            </TabsContent>
                            <TabsContent value="errors" className="h-full">
                                <div className="space-y-4">
                                    {errors.map((error, index) => (
                                        <Card key={index} className="relative">
                                            <CardContent className="p-4">
                                                <div className="flex items-center mb-2">
                                                    <div className="w-8 h-8 rounded-full bg-red-500 flex items-center justify-center text-white font-bold mr-3">
                                                        {index + 1}
                                                    </div>
                                                    <h3 className="text-lg font-semibold">{error.error_type}</h3>
                                                </div>
                                                <div className="grid grid-cols-2 gap-4 mb-4">
                                                    <div><span className="font-medium">Timestamp:</span> {new Date(error.timestamp).toLocaleString()}</div>
                                                    <div><span className="font-medium">Agent ID:</span> {error.agent_id || 'N/A'}</div>
                                                    <div><span className="font-medium">Tool Call ID:</span> {error.tool_call_id || 'N/A'}</div>
                                                    <div><span className="font-medium">LLM Call ID:</span> {error.llm_call_id || 'N/A'}</div>
                                                </div>
                                                <div className="mt-4">
                                                    <h4 className="font-medium mb-2">Error Message:</h4>
                                                    <ScrollArea className="h-40 w-full">
                                                        <pre className="bg-gray-100 p-2 rounded whitespace-pre-wrap">{error.error_message}</pre>
                                                    </ScrollArea>
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            </TabsContent>
                        </div>
                    </ScrollArea>
                </div>
            </Tabs>
        </div>
    );
};

export default TraceDetails;