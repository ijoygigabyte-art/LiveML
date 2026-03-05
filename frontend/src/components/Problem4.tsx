import { useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface Props {
    filename: string;
    targetCol: string;
    featureCols: string[];
}

export default function Problem4({ filename, targetCol, featureCols }: Props) {
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<any>(null);
    const [iterations] = useState(1000);

    const runAnalysis = async () => {
        setLoading(true);
        try {
            const req = {
                filename,
                target_col: targetCol,
                feature_cols: featureCols,
                learning_rates: [0.1, 0.01, 0.001],
                iterations
            };

            const apiUrl = import.meta.env.PROD ? '/api' : 'http://localhost:8000/api';
            const res = await axios.post(`${apiUrl}/problem4`, req);

            // Transform data for line chart
            const plotData = [];
            for (let i = 0; i < iterations; i += 10) { // sampling every 10 for performance
                plotData.push({
                    iteration: i,
                    alpha1: res.data['0.1'].cost_history[i],
                    alpha2: res.data['0.01'].cost_history[i],
                    alpha3: res.data['0.001'].cost_history[i],
                });
            }

            setResults({ raw: res.data, plotData });
        } catch (err) {
            console.error(err);
            alert('Analysis failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-surface border border-border rounded-xl p-8 shadow-xl">
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h2 className="text-2xl font-bold">⚙️ Problem 4: Gradient Descent</h2>
                    <p className="text-gray-400 mt-1">Cost vs. Iterations for Multiple Learning Rates</p>
                </div>
                <button
                    onClick={runAnalysis}
                    disabled={loading}
                    className="bg-[#2ecc71] hover:bg-[#27ae60] text-white font-bold py-2 px-6 rounded-lg transition-colors disabled:opacity-50"
                >
                    {loading ? 'Running...' : 'Run Analysis'}
                </button>
            </div>

            {results && (
                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">

                    <div className="grid grid-cols-3 gap-6">
                        <div className="bg-background border border-border rounded-xl p-4 text-center">
                            <p className="text-gray-400 text-sm">α = 0.1 Final Cost</p>
                            <p className="text-xl font-mono text-[#e74c3c] mt-1">
                                {results.raw['0.1'].final_cost !== null ? results.raw['0.1'].final_cost.toFixed(6) : 'Diverged (NaN)'}
                            </p>
                        </div>
                        <div className="bg-background border border-border rounded-xl p-4 text-center">
                            <p className="text-gray-400 text-sm">α = 0.01 Final Cost</p>
                            <p className="text-xl font-mono text-[#2ecc71] mt-1">
                                {results.raw['0.01'].final_cost !== null ? results.raw['0.01'].final_cost.toFixed(6) : 'Diverged (NaN)'}
                            </p>
                        </div>
                        <div className="bg-background border border-border rounded-xl p-4 text-center">
                            <p className="text-gray-400 text-sm">α = 0.001 Final Cost</p>
                            <p className="text-xl font-mono text-[#3498db] mt-1">
                                {results.raw['0.001'].final_cost !== null ? results.raw['0.001'].final_cost.toFixed(6) : 'Diverged (NaN)'}
                            </p>
                        </div>
                    </div>

                    <div className="bg-background border border-border rounded-xl p-6 h-96">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={results.plotData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#2D3348" />
                                <XAxis dataKey="iteration" stroke="#888" label={{ value: 'Iterations', position: 'insideBottom', offset: -5 }} />
                                <YAxis stroke="#888" label={{ value: 'Cost (MSE/2)', angle: -90, position: 'insideLeft' }} />
                                <Tooltip contentStyle={{ backgroundColor: '#1A1F2E', borderColor: '#2D3348', color: '#fff' }} />
                                <Legend verticalAlign="top" height={36} />
                                <Line type="monotone" dataKey="alpha1" name="α = 0.1" stroke="#e74c3c" strokeWidth={2} dot={false} />
                                <Line type="monotone" dataKey="alpha2" name="α = 0.01" stroke="#2ecc71" strokeWidth={2} dot={false} />
                                <Line type="monotone" dataKey="alpha3" name="α = 0.001" stroke="#3498db" strokeWidth={2} dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                </div>
            )}
        </div>
    );
}
