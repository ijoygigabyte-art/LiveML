import { useState } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface Props {
    filename: string;
    targetCol: string;
    featureCols: string[];
}

export default function Problem5({ filename, targetCol, featureCols }: Props) {
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<any>(null);

    const runAnalysis = async () => {
        setLoading(true);
        try {
            const reduced = featureCols.filter(c => c !== targetCol).slice(0, 3);
            if (reduced.length < 2) throw new Error("Need at least 2 features");

            const req = {
                filename,
                target_col: targetCol,
                feature_cols: featureCols,
                reduced_features: reduced,
                interact_a: reduced[0],
                interact_b: reduced[1]
            };

            const res = await axios.post(`${import.meta.env.VITE_API_URL || 'http://localhost:8000/api'}/problem5`, req);
            setResults(res.data);
        } catch (err) {
            console.error(err);
            alert('Analysis failed. Make sure you have at least 2 features.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-surface border border-border rounded-xl p-8 shadow-xl">
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h2 className="text-2xl font-bold">🏆 Problem 5: Model Selection</h2>
                    <p className="text-gray-400 mt-1">Comparing Full, Reduced, and Interaction Models via AIC/BIC</p>
                </div>
                <button
                    onClick={runAnalysis}
                    disabled={loading}
                    className="bg-[#9b59b6] hover:bg-[#8e44ad] text-white font-bold py-2 px-6 rounded-lg transition-colors disabled:opacity-50"
                >
                    {loading ? 'Running...' : 'Run Analysis'}
                </button>
            </div>

            {results && (
                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">

                    <div className="overflow-x-auto">
                        <table className="w-full text-left bg-background rounded-xl overflow-hidden shadow">
                            <thead className="bg-[#2D3348] text-gray-300">
                                <tr>
                                    <th className="p-4 rounded-tl-xl text-sm font-semibold">Model</th>
                                    <th className="p-4 text-sm font-semibold"># Features</th>
                                    <th className="p-4 text-sm font-semibold">Adj R²</th>
                                    <th className="p-4 text-sm font-semibold">AIC</th>
                                    <th className="p-4 rounded-tr-xl text-sm font-semibold">BIC</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border">
                                {results.comparison.map((m: any, i: number) => (
                                    <tr key={i} className="hover:bg-surface/50 transition-colors">
                                        <td className="p-4 font-medium">{m.Model}</td>
                                        <td className="p-4 font-mono">{m['# Features']}</td>
                                        <td className="p-4 font-mono">{m['Adj R²'].toFixed(4)}</td>
                                        <td className="p-4 font-mono">{m.AIC.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                        <td className="p-4 font-mono">{m.BIC.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    <div className="bg-background border border-border rounded-xl p-6 h-96">
                        <h3 className="text-sm font-semibold text-gray-300 mb-6 text-center">AIC vs BIC Comparison</h3>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={results.comparison} margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#2D3348" vertical={false} />
                                <XAxis dataKey="Model" stroke="#888" tick={{ fill: '#888' }} />
                                <YAxis stroke="#888" tick={{ fill: '#888' }} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1A1F2E', borderColor: '#2D3348', color: '#fff', borderRadius: '8px' }}
                                    cursor={{ fill: '#2D3348', opacity: 0.4 }}
                                />
                                <Legend verticalAlign="top" height={36} />
                                <Bar dataKey="AIC" fill="#3498db" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="BIC" fill="#e74c3c" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                </div>
            )}
        </div>
    );
}
