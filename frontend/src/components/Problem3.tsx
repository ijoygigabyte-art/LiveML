import { useState } from 'react';
import axios from 'axios';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface Props {
    filename: string;
    targetCol: string;
    featureCols: string[];
}

export default function Problem3({ filename, targetCol, featureCols }: Props) {
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<any>(null);

    const runAnalysis = async () => {
        setLoading(true);
        try {
            const req = {
                filename,
                target_col: targetCol,
                feature_cols: featureCols,
                num_cols: featureCols.slice(0, 4) // Example: Select first 4 as numerical to standardize
            };
            const apiUrl = import.meta.env.PROD ? '/api' : 'http://localhost:8000/api';
            const res = await axios.post(`${apiUrl}/problem3`, req);

            // Transform data for recharts
            const plotData = res.data.plots.y_sampled.map((val: number, i: number) => ({
                observed: val,
                predicted: res.data.plots.y_pred_sampled[i],
                residual: res.data.plots.residuals_sampled[i]
            }));

            setResults({ ...res.data, plotData });
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
                    <h2 className="text-2xl font-bold">📐 Problem 3: Multiple Linear Regression</h2>
                    <p className="text-gray-400 mt-1">Target: <code className="text-primary">{targetCol}</code></p>
                </div>
                <button
                    onClick={runAnalysis}
                    disabled={loading}
                    className="bg-[#3498db] text-white hover:bg-[#2980b9] font-bold py-2 px-6 rounded-lg transition-colors disabled:opacity-50"
                >
                    {loading ? 'Running...' : 'Run Analysis'}
                </button>
            </div>

            {results && (
                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">

                    <div className="grid grid-cols-4 gap-4">
                        <div className="bg-background border border-border rounded-xl p-4 text-center">
                            <p className="text-gray-400 text-sm">R²</p>
                            <p className="text-2xl font-mono text-[#3498db] mt-1">{results.metrics.rsquared.toFixed(4)}</p>
                        </div>
                        <div className="bg-background border border-border rounded-xl p-4 text-center">
                            <p className="text-gray-400 text-sm">Adj R²</p>
                            <p className="text-2xl font-mono text-[#3498db] mt-1">{results.metrics.rsquared_adj.toFixed(4)}</p>
                        </div>
                        <div className="bg-background border border-border rounded-xl p-4 text-center">
                            <p className="text-gray-400 text-sm">MSE</p>
                            <p className="text-2xl font-mono text-accent mt-1">{results.metrics.mse.toLocaleString(undefined, { maximumFractionDigits: 0 })}</p>
                        </div>
                        <div className="bg-background border border-border rounded-xl p-4 text-center">
                            <p className="text-gray-400 text-sm">RMSE</p>
                            <p className="text-2xl font-mono text-accent mt-1">{results.metrics.rmse.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-6">
                        <div className="bg-background border border-border rounded-xl p-6 h-80">
                            <h3 className="text-sm font-semibold text-gray-300 mb-4 text-center">Linearity (Observed vs Predicted)</h3>
                            <ResponsiveContainer width="100%" height="100%">
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#2D3348" />
                                    <XAxis type="number" dataKey="predicted" name="Predicted" stroke="#888" />
                                    <YAxis type="number" dataKey="observed" name="Observed" stroke="#888" />
                                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1A1F2E', borderColor: '#2D3348', color: '#fff' }} />
                                    <Scatter name="Data" data={results.plotData} fill="#3498db" opacity={0.5} />
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="bg-background border border-border rounded-xl p-6 h-80">
                            <h3 className="text-sm font-semibold text-gray-300 mb-4 text-center">Homoscedasticity (Residuals vs Fitted)</h3>
                            <ResponsiveContainer width="100%" height="100%">
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#2D3348" />
                                    <XAxis type="number" dataKey="predicted" name="Fitted" stroke="#888" />
                                    <YAxis type="number" dataKey="residual" name="Residual" stroke="#888" />
                                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1A1F2E', borderColor: '#2D3348', color: '#fff' }} />
                                    <Scatter name="Data" data={results.plotData} fill="#e74c3c" opacity={0.5} />
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                </div>
            )}
        </div>
    );
}
