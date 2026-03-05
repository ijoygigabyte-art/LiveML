import { useState } from 'react';
import axios from 'axios';

interface Props {
    filename: string;
    featureCols: string[];
}

export default function Problem2({ filename, featureCols }: Props) {
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<any>(null);

    const runAnalysis = async () => {
        setLoading(true);
        try {
            const req = {
                filename,
                num_cols: featureCols.slice(0, 10) // Just limit to 10 for speed
            };
            const res = await axios.post(`${import.meta.env.VITE_API_URL || 'http://localhost:8000/api'}/problem2`, req);
            setResults(res.data);
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
                    <h2 className="text-2xl font-bold">🔗 Problem 2: Correlation Analysis</h2>
                    <p className="text-gray-400 mt-1">Analyzes top numerical features</p>
                </div>
                <button
                    onClick={runAnalysis}
                    disabled={loading}
                    className="bg-accent text-background hover:bg-accent/90 font-bold py-2 px-6 rounded-lg transition-colors disabled:opacity-50"
                >
                    {loading ? 'Running...' : 'Run Analysis'}
                </button>
            </div>

            {results && (
                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">

                    <div className="grid grid-cols-2 gap-6">
                        <div className="bg-background border border-border rounded-xl p-6">
                            <h3 className="text-lg font-semibold text-green-400 flex items-center mb-4">
                                <span className="w-2 h-2 rounded-full bg-green-400 mr-2"></span>
                                Top Positive Correlations
                            </h3>
                            <div className="space-y-3">
                                {results.top_pos.map((p: any, i: number) => (
                                    <div key={i} className="flex justify-between items-center bg-surface p-3 rounded-lg border border-border/50">
                                        <span className="text-sm font-medium">{p['Feature 1']} ↔ {p['Feature 2']}</span>
                                        <span className="font-mono text-green-400">+{p.Correlation.toFixed(4)}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="bg-background border border-border rounded-xl p-6">
                            <h3 className="text-lg font-semibold text-red-400 flex items-center mb-4">
                                <span className="w-2 h-2 rounded-full bg-red-400 mr-2"></span>
                                Top Negative Correlations
                            </h3>
                            <div className="space-y-3">
                                {results.top_neg.map((p: any, i: number) => (
                                    <div key={i} className="flex justify-between items-center bg-surface p-3 rounded-lg border border-border/50">
                                        <span className="text-sm font-medium">{p['Feature 1']} ↔ {p['Feature 2']}</span>
                                        <span className="font-mono text-red-400">{p.Correlation.toFixed(4)}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                </div>
            )}
        </div>
    );
}
