import { useState } from 'react';
import axios from 'axios';

interface Props {
    filename: string;
    targetCol: string;
    featureCols: string[];
}

export default function Problem1({ filename, targetCol, featureCols }: Props) {
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<any>(null);
    const [threshold, setThreshold] = useState<number | ''>('');

    const runAnalysis = async () => {
        setLoading(true);
        try {
            // For P1, we assume all string cols might be categorical
            const req = {
                filename,
                target_col: targetCol,
                target_threshold: threshold !== '' ? threshold : undefined,
                feature_cols: featureCols,
                cat_cols: [] // For simplicity, rely on pandas doing it or not passing it
            };

            const apiUrl = import.meta.env.PROD ? '/api' : 'http://localhost:8000/api';
            const res = await axios.post(`${apiUrl}/problem1`, req);
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
                    <h2 className="text-2xl font-bold">🛡️ Problem 1: Imbalanced Classification</h2>
                    <p className="text-gray-400 mt-1">Target: <code className="text-primary">{targetCol}</code></p>
                    <div className="mt-4 flex items-center space-x-3 bg-background border border-border rounded-lg p-3 inline-block">
                        <label className="text-sm font-medium text-gray-300">Continuous Target Threshold (Value ≥ Threshold = 1):</label>
                        <input
                            type="number"
                            className="bg-surface border border-border rounded px-3 py-1.5 text-sm w-28 text-white focus:outline-none focus:border-primary transition-colors"
                            value={threshold}
                            onChange={(e) => setThreshold(e.target.value === '' ? '' : Number(e.target.value))}
                            placeholder="e.g. 0.5"
                        />
                    </div>
                </div>
                <button
                    onClick={runAnalysis}
                    disabled={loading}
                    className="bg-primary hover:bg-primary/90 text-white font-semibold py-2 px-6 rounded-lg transition-colors disabled:opacity-50"
                >
                    {loading ? 'Running...' : 'Run Analysis'}
                </button>
            </div>

            {results && (
                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">

                    <div className="grid grid-cols-2 gap-6">
                        <div className="bg-background border border-border rounded-xl p-6">
                            <h3 className="text-lg font-semibold text-accent mb-4">Baseline Random Forest</h3>
                            <div className="space-y-4">
                                <div className="flex justify-between items-center border-b border-border pb-2">
                                    <span className="text-gray-400">Accuracy</span>
                                    <span className="font-mono text-xl">{results.base_metrics.Accuracy.toFixed(4)}</span>
                                </div>
                                <div className="flex justify-between items-center border-b border-border pb-2">
                                    <span className="text-gray-400">Precision</span>
                                    <span className="font-mono text-xl">{results.base_metrics.Precision.toFixed(4)}</span>
                                </div>
                                <div className="flex justify-between items-center border-b border-border pb-2">
                                    <span className="text-gray-400">Recall</span>
                                    <span className="font-mono text-xl">{results.base_metrics.Recall.toFixed(4)}</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-gray-400">F1-score</span>
                                    <span className="font-mono text-xl">{results.base_metrics['F1-score'].toFixed(4)}</span>
                                </div>
                            </div>
                        </div>

                        <div className="bg-background border border-border rounded-xl p-6 relative overflow-hidden">
                            <div className="absolute top-0 right-0 w-32 h-32 bg-green-500/10 rounded-bl-full -mr-8 -mt-8" />
                            <h3 className="text-lg font-semibold text-green-400 mb-4">After SMOTETomek</h3>
                            <div className="space-y-4">
                                <div className="flex justify-between items-center border-b border-border pb-2">
                                    <span className="text-gray-400">Accuracy</span>
                                    <div className="text-right">
                                        <span className="font-mono text-xl block">{results.res_metrics.Accuracy.toFixed(4)}</span>
                                        <span className={`text-xs ${results.res_metrics.Accuracy > results.base_metrics.Accuracy ? 'text-green-400' : 'text-red-400'}`}>
                                            {((results.res_metrics.Accuracy - results.base_metrics.Accuracy) * 100).toFixed(2)}%
                                        </span>
                                    </div>
                                </div>
                                <div className="flex justify-between items-center border-b border-border pb-2">
                                    <span className="text-gray-400">Precision</span>
                                    <div className="text-right">
                                        <span className="font-mono text-xl block">{results.res_metrics.Precision.toFixed(4)}</span>
                                        <span className={`text-xs ${results.res_metrics.Precision > results.base_metrics.Precision ? 'text-green-400' : 'text-red-400'}`}>
                                            {((results.res_metrics.Precision - results.base_metrics.Precision) * 100).toFixed(2)}%
                                        </span>
                                    </div>
                                </div>
                                <div className="flex justify-between items-center border-b border-border pb-2">
                                    <span className="text-gray-400">Recall</span>
                                    <div className="text-right">
                                        <span className="font-mono text-xl block">{results.res_metrics.Recall.toFixed(4)}</span>
                                        <span className={`text-xs ${results.res_metrics.Recall > results.base_metrics.Recall ? 'text-green-400' : 'text-red-400'}`}>
                                            {((results.res_metrics.Recall - results.base_metrics.Recall) * 100).toFixed(2)}%
                                        </span>
                                    </div>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-gray-400">F1-score</span>
                                    <div className="text-right">
                                        <span className="font-mono text-xl block">{results.res_metrics['F1-score'].toFixed(4)}</span>
                                        <span className={`text-xs ${results.res_metrics['F1-score'] > results.base_metrics['F1-score'] ? 'text-green-400' : 'text-red-400'}`}>
                                            {((results.res_metrics['F1-score'] - results.base_metrics['F1-score']) * 100).toFixed(2)}%
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="bg-primary/10 border border-primary/20 rounded-xl p-6">
                        <h4 className="font-medium text-primary mb-2">Analysis Insight</h4>
                        <p className="text-gray-300">
                            Resampling the dataset using SMOTETomek significantly improves the recall of the minority class, meaning the model becomes better at identifying actual high-demand instances. However, this often comes at the cost of precision, resulting in more false positives.
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
}
