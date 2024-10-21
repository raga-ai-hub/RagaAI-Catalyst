import { TimelineData } from '../types/timeline';

import initSqlJs from 'sql.js';

let SQL;
let db;

export const initDatabase = async () => {
    if (!SQL) {
        console.log('Initializing SQL.js');
        SQL = await initSqlJs({ locateFile: file => `/${file}` });
    }
    if (!db) {
        console.log('Loading trace_data.db');
        const response = await fetch('/trace_data.db');
        const arrayBuffer = await response.arrayBuffer();
        db = new SQL.Database(new Uint8Array(arrayBuffer));
        console.log('Database loaded successfully');
    }
    return db;
};

export const getDatabase = () => {
    if (!db) {
        throw new Error('Database not initialized. Call initDatabase() first.');
    }
    return db;
};

export const fetchAllProjects = async () => {
    await initDatabase();

    console.log('Fetching all projects');

    const result = db.exec(`
      SELECT id, project_name
      FROM project_info
      ORDER BY id
    `);

    console.log('All projects query result:', result);

    if (result[0] && result[0].values) {
        return result[0].values.map(([id, project_name]) => ({
            id,
            name: project_name,
        }));
    }

    console.log('No projects found');
    return [];
};

export const fetchProjectInfo = async (projectId: number) => {
    await initDatabase();

    console.log(`Fetching project info for project ID: ${projectId}`);

    // Fetch basic project info
    const projectInfoResult = db.exec(`
        SELECT id, project_name, start_time, end_time, duration
        FROM project_info
        WHERE id = ?
    `, [projectId]);

    console.log('Project info result:', projectInfoResult);

    if (!projectInfoResult[0] || !projectInfoResult[0].values[0]) {
        console.log('No project info found');
        return null;
    }

    const [id, project_name, start_time, end_time, duration] = projectInfoResult[0].values[0];

    // Fetch LLM calls for the project
    const llmCallsResult = db.exec(`
        SELECT token_usage, cost
        FROM llm_call
        WHERE project_id = ?
    `, [projectId]);

    console.log('LLM calls result:', llmCallsResult);

    let total_tokens = 0;
    let total_cost = 0;

    if (llmCallsResult[0] && llmCallsResult[0].values) {
        llmCallsResult[0].values.forEach(([token_usage, cost]) => {
            console.log('Processing LLM call:', { token_usage, cost });

            try {
                // Parse the double-encoded JSON strings
                const tokenUsageObj = JSON.parse(JSON.parse(token_usage));
                const costObj = JSON.parse(JSON.parse(cost));

                total_tokens += (tokenUsageObj.input || 0) + (tokenUsageObj.completion || 0) + (tokenUsageObj.reasoning || 0);
                total_cost += (costObj.input || 0) + (costObj.output || 0) + (costObj.reasoning || 0);

                console.log('Updated totals:', { total_tokens, total_cost });
            } catch (error) {
                console.error('Error parsing JSON:', error);
            }
        });
    }

    // Format total_cost
    const formattedTotalCost = total_cost < 0.001 ? total_cost.toExponential(3) : total_cost.toFixed(3);

    const result = {
        id,
        project_name,
        start_time,
        end_time,
        duration: duration !== null ? Number(duration) : null,
        total_tokens,
        total_cost: formattedTotalCost
    };

    console.log('Final project info:', result);

    return result;
};

export const fetchTraceData = async (projectId: number, traceId: string) => {
    await initDatabase();

    try {
        const traceResult = db.exec(`
            SELECT id, start_time, end_time
            FROM traces
            WHERE project_id = ? AND id = ?
            LIMIT 1
        `, [projectId, traceId]);

        if (!traceResult[0] || !traceResult[0].values[0]) {
            throw new Error(`No trace found for project ID ${projectId} and trace ID ${traceId}`);
        }

        const [resultTraceId, traceStartTime, traceEndTime] = traceResult[0].values[0];

        const llmCallsResult = db.exec(`
            SELECT *
            FROM llm_call
            WHERE trace_id = ?
        `, [traceId]);

        const toolCallsResult = db.exec(`
            SELECT *
            FROM tool_call
            WHERE trace_id = ?
        `, [traceId]);

        const agentCallsResult = db.exec(`
            SELECT *
            FROM agent_call
            WHERE trace_id = ?
        `, [traceId]);

        const errorsResult = db.exec(`
            SELECT *
            FROM errors
            WHERE trace_id = ?
        `, [traceId]);

        const formatResults = (result) => {
            if (!result[0]) return [];
            const columns = result[0].columns;
            return result[0].values.map(row =>
                columns.reduce((obj, col, index) => {
                    obj[col] = row[index];
                    return obj;
                }, {})
            );
        };

        return {
            id: traceId,
            start_time: traceStartTime,
            end_time: traceEndTime,
            llm_calls: formatResults(llmCallsResult),
            tool_calls: formatResults(toolCallsResult),
            agent_calls: formatResults(agentCallsResult),
            errors: formatResults(errorsResult)
        };
    } catch (error) {
        console.error('Error fetching trace data:', error);
        throw error;
    }
};

export const fetchSystemInfo = async (projectId) => {
    await initDatabase();

    const result = db.exec(`
      SELECT os_name, os_version, python_version, cpu_info, gpu_info, memory_total, disk_info
      FROM system_info
      WHERE project_id = ${projectId}
      ORDER BY id DESC
      LIMIT 1
    `);

    if (result[0] && result[0].values[0]) {
        const [os_name, os_version, python_version, cpu_info, gpu_info, memory_total, disk_info] = result[0].values[0];
        return {
            os: `${os_name} ${os_version}`,
            pythonVersion: python_version,
            cpu: cpu_info,
            gpu: gpu_info || 'N/A',
            totalMemory: `${(memory_total).toFixed(2)} GB`,
            diskSpace: disk_info
        };
    }
    return null;
};

export const fetchInstalledPackages = async (projectId) => {
    await initDatabase();

    const result = db.exec(`
      SELECT installed_packages
      FROM system_info
      WHERE project_id = ${projectId}
      ORDER BY id DESC
      LIMIT 1
    `);

    if (result[0] && result[0].values[0]) {
        const installedPackages = JSON.parse(result[0].values[0][0]);
        return Object.entries(installedPackages).map(([name, version]) => ({ name, version }));
    }
    return [];
};

export const fetchAgentCalls = async (projectId) => {
    await initDatabase();

    const result = db.exec(`
      SELECT 
        name,
        COUNT(*) as count,
        AVG(JULIANDAY(end_time) - JULIANDAY(start_time)) * 86400 as avg_duration,
        AVG(JSON_ARRAY_LENGTH(tool_call_ids)) as avg_tool_calls,
        AVG(JSON_ARRAY_LENGTH(llm_call_ids)) as avg_llm_calls
      FROM agent_call
      WHERE project_id = ${projectId}
      GROUP BY name
    `);

    if (result[0] && result[0].values) {
        return result[0].values.map(([name, count, avgDuration, avgToolCalls, avgLLMCalls]) => ({
            name,
            count,
            avgDuration: parseFloat(avgDuration).toFixed(2),
            avgToolCalls: parseFloat(avgToolCalls).toFixed(2),
            avgLLMCalls: parseFloat(avgLLMCalls).toFixed(2)
        }));
    }
    return [];
};


const typeColors = {
    'LLM': '#34A853',
    'Tool': '#FBBC05',
    'Action': '#4285F4',
    'Error': '#EA4335',
    // Add more types and colors as needed
};

export const fetchTimelineData = async (projectId: number, traceId: string): Promise<TimelineData[]> => {
    await initDatabase();

    const result = db.exec(`
      SELECT 
        CASE
          WHEN ac.id IS NOT NULL THEN ac.name
          WHEN lc.id IS NOT NULL THEN lc.name
          WHEN tc.id IS NOT NULL THEN tc.name
          WHEN e.id IS NOT NULL THEN e.error_type
          WHEN ui.id IS NOT NULL THEN ui.interaction_type
          ELSE 'Unknown'
        END as name,
        COALESCE(ac.start_time, lc.start_time, tc.start_time, e.timestamp, ui.timestamp) as start_time,
        COALESCE(ac.end_time, lc.end_time, tc.end_time, e.timestamp, ui.timestamp) as end_time,
        CASE
          WHEN ac.id IS NOT NULL THEN 'Action'
          WHEN lc.id IS NOT NULL THEN 'LLM'
          WHEN tc.id IS NOT NULL THEN 'Tool'
          WHEN e.id IS NOT NULL THEN 'Error'
          WHEN ui.id IS NOT NULL THEN 'UserInteraction'
          ELSE 'Unknown'
        END as type,
        COALESCE(ac.id, lc.id, tc.id, e.id, ui.id) as id,
        ac.name as agent_name,
        ac.llm_call_ids,
        ac.tool_call_ids,
        lc.input_prompt,
        lc.output as llm_output,
        lc.model,
        lc.token_usage,
        lc.cost,
        tc.name as tool_name,
        tc.input_parameters,
        tc.output as tool_output,
        tc.network_calls,
        e.error_type,
        e.error_message,
        ui.interaction_type,
        ui.content
      FROM traces t
      LEFT JOIN agent_call ac ON t.id = ac.trace_id
      LEFT JOIN llm_call lc ON t.id = lc.trace_id
      LEFT JOIN tool_call tc ON t.id = tc.trace_id
      LEFT JOIN errors e ON t.id = e.trace_id
      LEFT JOIN user_interactions ui ON t.id = ui.trace_id
      WHERE t.project_id = ? AND t.id = ?
      ORDER BY start_time
    `, [projectId, traceId]);

    if (result[0] && result[0].values) {
        return result[0].values.map(([
            name, start_time, end_time, type, id,
            agent_name, llm_call_ids, tool_call_ids,
            input_prompt, llm_output, model, token_usage, cost,
            tool_name, input_parameters, tool_output, network_calls,
            error_type, error_message, interaction_type, content
        ]) => ({
            name,
            start_time,
            end_time,
            type,
            id,
            details: {
                agent: agent_name,
                llm_call_ids,
                tool_call_ids,
                input: input_prompt || input_parameters,
                output: llm_output || tool_output,
                model,
                token_usage,
                cost,
                function: tool_name,
                network_calls,
                error_type,
                error_message,
                interaction_type,
                content
            }
        }));
    }

    return [];
};

// export const fetchTracesForProject = async (projectId: number): Promise<{ id: string; name: string }[]> => {
//     await initDatabase();

//     const result = db.exec(`
//       SELECT id, start_time
//       FROM traces
//       WHERE project_id = ${projectId}
//       ORDER BY start_time DESC
//     `);

//     if (result[0] && result[0].values) {
//         return result[0].values.map(([id, start_time]) => ({
//             id: id.toString(),
//             name: `Trace ${id} (${new Date(start_time).toLocaleString()})`
//         }));
//     }

//     return [];
// };

// For Trace History Page
export interface TraceHistoryItem {
    id: string;
    start_time: string;
    duration: number;
    llm_call_count: number;
    tool_call_count: number;
    agent_call_count: number;
    error_count: number;
}

export const fetchTraceHistory = async (projectId: number): Promise<TraceHistoryItem[]> => {
    await initDatabase();

    const result = db.exec(`
      SELECT 
        t.id,
        t.start_time,
        t.duration,
        (SELECT COUNT(*) FROM llm_call WHERE trace_id = t.id) as llm_call_count,
        (SELECT COUNT(*) FROM tool_call WHERE trace_id = t.id) as tool_call_count,
        (SELECT COUNT(*) FROM agent_call WHERE trace_id = t.id) as agent_call_count,
        (SELECT COUNT(*) FROM errors WHERE trace_id = t.id) as error_count
\      FROM traces t
      WHERE t.project_id = ?
      ORDER BY t.start_time DESC
    `, [projectId]);

    if (result[0] && result[0].values) {
        return result[0].values.map(([id, start_time, duration, llm_call_count, tool_call_count, agent_call_count, error_count, user_interaction_count]) => ({
            id: id.toString(),
            start_time,
            duration: duration || 0,
            llm_call_count: llm_call_count || 0,
            tool_call_count: tool_call_count || 0,
            agent_call_count: agent_call_count || 0,
            error_count: error_count || 0,
        }));
    }

    return [];
};


// For Detailed Trace in Trace History Page
export interface DetailedTraceComponents {
    agents: any[];
    tools: any[];
    llmCalls: any[];
    errors: any[];
}

export const fetchDetailedTraceComponents = async (traceId: string): Promise<DetailedTraceComponents> => {
    await initDatabase();

    const agentResult = db.exec(`
        SELECT *
        FROM agent_call
        WHERE trace_id = ?
    `, [traceId]);

    const toolResult = db.exec(`
        SELECT *
        FROM tool_call
        WHERE trace_id = ?
    `, [traceId]);

    const llmResult = db.exec(`
        SELECT *
        FROM llm_call
        WHERE trace_id = ?
    `, [traceId]);

    const errorResult = db.exec(`
        SELECT *
        FROM errors
        WHERE trace_id = ?
    `, [traceId]);

    return {
        agents: agentResult[0]?.values.map(row => {
            const columns = agentResult[0].columns;
            return columns.reduce((obj, col, index) => {
                obj[col] = row[index];
                return obj;
            }, {});
        }) || [],
        tools: toolResult[0]?.values.map(row => {
            const columns = toolResult[0].columns;
            return columns.reduce((obj, col, index) => {
                obj[col] = row[index];
                return obj;
            }, {});
        }) || [],
        llmCalls: llmResult[0]?.values.map(row => {
            const columns = llmResult[0].columns;
            return columns.reduce((obj, col, index) => {
                obj[col] = row[index];
                return obj;
            }, {});
        }) || [],
        errors: errorResult[0]?.values.map(row => {
            const columns = errorResult[0].columns;
            return columns.reduce((obj, col, index) => {
                obj[col] = row[index];
                return obj;
            }, {});
        }) || [],
    };
};

export const fetchEvaluationData = async (projectId: number, traceId: string | null): Promise<any[]> => {
    await initDatabase();

    let query = `
      SELECT 
        m.id,
        t.id as trace_id,
        m.metric_name,
        m.score,
        m.reason,
        m.result_detail,
        m.config,
        t.start_time,
        t.end_time,
        (julianday(t.end_time) - julianday(t.start_time)) * 86400 as duration
      FROM metrics m
      JOIN traces t ON m.trace_id = t.id
      WHERE t.project_id = ?
    `;

    const params: (number | string)[] = [projectId];

    if (traceId && traceId !== 'all') {
        query += ' AND t.id = ?';
        params.push(traceId);
    }

    query += ' ORDER BY t.start_time DESC';

    const result = db.exec(query, params);

    if (result[0] && result[0].values) {
        return result[0].values.map(([id, trace_id, metric_name, score, reason, result_detail, config, start_time, end_time, duration]) => ({
            id,
            trace_id,
            type: metric_name,
            [metric_name.toLowerCase().replace(/ /g, '_')]: {
                score: parseFloat(score),
                reason,
                result_detail: JSON.parse(result_detail),
                config: JSON.parse(config),
            },
            start_time,
            end_time,
            duration: parseFloat(duration)
        }));
    }

    return [];
};

export const fetchTracesForProject = async (projectId: number): Promise<{ id: string; name: string }[]> => {
    await initDatabase();

    const result = db.exec(`
        SELECT id, start_time
        FROM traces
        WHERE project_id = ${projectId}
        ORDER BY start_time DESC
      `);

    if (result[0] && result[0].values) {
        return result[0].values.map(([id, start_time]) => ({
            id: id.toString(),
            name: `Trace ${id} (${new Date(start_time).toLocaleString()})`
        }));
    }

    return [];
};


// For the Execution graph in the Overview Page:
export interface GraphNode {
    id: string;
    type: 'agent' | 'llm' | 'tool' | 'user_interaction' | 'error';
    data: {
        name: string;
        start_time: string;
        end_time: string;
        duration: number;
        memory_used?: number;
        token_usage?: { input: number; completion: number };
        error_message?: string;
        content?: string;
        llm_call_ids?: string[];
        tool_call_ids?: string[];
    };
}

export interface GraphEdge {
    id: string;
    source: string;
    target: string;
}

export const fetchExecutionGraphData = async (projectId: number, traceId: string): Promise<{ nodes: GraphNode[], edges: GraphEdge[] }> => {
    await initDatabase();

    const result = db.exec(`
      WITH RECURSIVE
      agent_calls AS (
        SELECT id, name, start_time, end_time, llm_call_ids, tool_call_ids
        FROM agent_call
        WHERE project_id = ? AND trace_id = ?
      ),
      llm_calls AS (
        SELECT id, name, model, start_time, end_time, agent_id, token_usage, cost, input_prompt, output
        FROM llm_call
        WHERE project_id = ? AND trace_id = ?
      ),
      tool_calls AS (
        SELECT id, name, start_time, end_time, agent_id, network_calls, input_parameters, output
        FROM tool_call
        WHERE project_id = ? AND trace_id = ?
      ),
      user_interaction_data AS (
        SELECT id, interaction_type, timestamp as start_time, timestamp as end_time, content, agent_id
        FROM user_interactions
        WHERE project_id = ? AND trace_id = ?
      ),
      all_nodes AS (
        SELECT 'agent' as type, id, name, start_time, end_time, NULL as parent_id,
               json_object('llm_call_ids', llm_call_ids, 'tool_call_ids', tool_call_ids) as extra_data
        FROM agent_calls
        UNION ALL
        SELECT 'llm' as type, id, name, start_time, end_time, agent_id as parent_id,
               json_object('model', model, 'token_usage', token_usage, 'cost', cost, 'input_prompt', input_prompt, 'output', output) as extra_data
        FROM llm_calls
        UNION ALL
        SELECT 'tool' as type, id, name, start_time, end_time, agent_id as parent_id,
               json_object('network_calls', network_calls, 'input_parameters', input_parameters, 'output', output) as extra_data
        FROM tool_calls
        UNION ALL
        SELECT 'user_interaction' as type, id, 'User Interaction' as name, start_time, end_time, agent_id as parent_id,
               json_object('interaction_type', interaction_type, 'content', content) as extra_data
        FROM user_interaction_data
      )
      SELECT * FROM all_nodes
      ORDER BY start_time
    `, [projectId, traceId, projectId, traceId, projectId, traceId, projectId, traceId]);

    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];
    const nodeMap: { [key: string]: GraphNode } = {};

    if (result[0] && result[0].values) {
        result[0].values.forEach(([type, id, name, start_time, end_time, parent_id, extra_data]) => {
            const nodeId = `${type}_${id}`;
            let parsedExtraData;
            try {
                parsedExtraData = JSON.parse(extra_data);
            } catch (error) {
                console.error(`Failed to parse extra_data for node ${nodeId}:`, extra_data);
                parsedExtraData = {};
            }
            const duration = new Date(end_time).getTime() - new Date(start_time).getTime();

            const node: GraphNode = {
                id: nodeId,
                type: type as GraphNode['type'],
                data: {
                    name,
                    start_time,
                    end_time,
                    duration,
                    ...parsedExtraData,
                }
            };

            nodes.push(node);
            nodeMap[nodeId] = node;

            if (parent_id) {
                const parentNodeId = `agent_${parent_id}`;
                const edgeId = `${parentNodeId}_to_${nodeId}`;
                if (!edges.some(edge => edge.id === edgeId)) {
                    edges.push({
                        id: edgeId,
                        source: parentNodeId,
                        target: nodeId
                    });
                }
            }
        });

        // Add edges for agent's llm_calls and tool_calls
        nodes.forEach(node => {
            if (node.type === 'agent') {
                const llmCallIds = Array.isArray(node.data.llm_call_ids) ? node.data.llm_call_ids : JSON.parse(node.data.llm_call_ids || '[]');
                const toolCallIds = Array.isArray(node.data.tool_call_ids) ? node.data.tool_call_ids : JSON.parse(node.data.tool_call_ids || '[]');

                [...llmCallIds, ...toolCallIds].forEach(callId => {
                    const callNodeId = `${callId.startsWith('llm') ? 'llm' : 'tool'}_${callId}`;
                    const edgeId = `${node.id}_to_${callNodeId}`;
                    if (!edges.some(edge => edge.id === edgeId)) {
                        edges.push({
                            id: edgeId,
                            source: node.id,
                            target: callNodeId
                        });
                    }
                });
            }
        });

        // Add edges based on execution order for nodes without parents
        const rootNodes = nodes.filter(node => !edges.some(edge => edge.target === node.id));
        for (let i = 1; i < rootNodes.length; i++) {
            const prevNode = rootNodes[i - 1];
            const currentNode = rootNodes[i];
            const edgeId = `${prevNode.id}_to_${currentNode.id}`;

            if (!edges.some(edge => edge.id === edgeId)) {
                edges.push({
                    id: edgeId,
                    source: prevNode.id,
                    target: currentNode.id
                });
            }
        }
    }

    console.log('Fetched nodes:', nodes.map(node => ({ id: node.id, type: node.type, data: node.data })));
    console.log('Fetched edges:', edges);

    return { nodes, edges };
};
