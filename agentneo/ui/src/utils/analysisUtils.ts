import { initDatabase, getDatabase } from './databaseUtils';

interface LLMUsageData {
    model: string;
    input: number;
    completion: number;
    reasoning: number;
    cost: number;
}

export const fetchLLMUsageData = async (projectId: number): Promise<LLMUsageData[]> => {
    await initDatabase();
    const db = getDatabase();

    const result = db.exec(`
        SELECT name, token_usage, cost
        FROM llm_call
        WHERE project_id = ?
    `, [projectId]);

    console.log('LLM usage result:', result);

    const usageData: { [key: string]: LLMUsageData } = {};

    if (result[0] && result[0].values) {
        result[0].values.forEach(([name, token_usage, cost]) => {
            const model = name;
            let usage;
            let costData;
            try {
                usage = JSON.parse(token_usage);
                costData = JSON.parse(cost);

                if (typeof usage === 'string') {
                    usage = JSON.parse(usage);
                }
                if (typeof costData === 'string') {
                    costData = JSON.parse(costData);
                }
            } catch (error) {
                console.error('Error parsing data:', error);
                usage = { input: 0, completion: 0, reasoning: 0 };
                costData = { total: 0 };
            }

            // Ensure usage and cost are objects with number properties
            usage = {
                input: parseInt(usage.input) || 0,
                completion: parseInt(usage.completion) || 0,
                reasoning: parseInt(usage.reasoning) || 0
            };
            const totalCost = parseFloat(costData.total) || 0;

            if (!usageData[model]) {
                usageData[model] = { model, input: 0, completion: 0, reasoning: 0, cost: 0 };
            }

            usageData[model].input += usage.input;
            usageData[model].completion += usage.completion;
            usageData[model].reasoning += usage.reasoning;
            usageData[model].cost += totalCost;
        });
    }

    const finalData = Object.values(usageData);
    console.log('Final LLM usage data:', finalData);
    return finalData;
};