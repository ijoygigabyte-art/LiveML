import { useState } from 'react';
import axios from 'axios';
import { UploadCloud, Activity, TrendingUp, ScatterChart, Shield, Calculator } from 'lucide-react';

// We'll import these components next
import Problem1 from './components/Problem1.tsx';
import Problem2 from './components/Problem2.tsx';
import Problem3 from './components/Problem3.tsx';
import Problem4 from './components/Problem4.tsx';
import Problem5 from './components/Problem5.tsx';

function App() {
  const [fileData, setFileData] = useState<any>(null);
  const [activeProblem, setActiveProblem] = useState<number>(1);
  const [targetCol, setTargetCol] = useState<string>('');
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [colsToDrop, setColsToDrop] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);

  const handleFileUpload = async (e: any) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = import.meta.env.PROD ? '/api' : 'http://localhost:8000/api';
      const res = await axios.post(`${apiUrl}/upload`, formData);
      setFileData(res.data);
      // Auto-select features (exclude target inside the problem components initially, but provide all for now)
      setFeatureCols(res.data.columns);
      setColsToDrop([]);
    } catch (err) {
      console.error(err);
      alert('Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const menuItems = [
    { id: 1, label: 'Problem 1: Classification', icon: <Shield size={20} /> },
    { id: 2, label: 'Problem 2: Correlation', icon: <Activity size={20} /> },
    { id: 3, label: 'Problem 3: Regression', icon: <ScatterChart size={20} /> },
    { id: 4, label: 'Problem 4: Gradient Descent', icon: <TrendingUp size={20} /> },
    { id: 5, label: 'Problem 5: Model Selection', icon: <Calculator size={20} /> },
  ];

  return (
    <div className="min-h-screen flex bg-background text-gray-100 font-sans">
      {/* Sidebar */}
      <div className="w-80 bg-gradient-to-b from-[#0E1117] to-surface border-r border-border p-6 flex flex-col h-screen fixed">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent mb-8">
          🤖 Smart ML
        </h1>

        <div className="mb-8">
          <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-border rounded-xl cursor-pointer hover:bg-surface/50 transition-colors">
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <UploadCloud className="w-8 h-8 mb-3 text-gray-400" />
              <p className="text-sm text-gray-400">{uploading ? 'Uploading...' : 'Upload CSV Dataset'}</p>
            </div>
            <input type="file" className="hidden" accept=".csv" onChange={handleFileUpload} />
          </label>

          {fileData && (
            <div className="mt-4 p-4 rounded-lg bg-surface border border-border">
              <p className="text-sm font-medium text-green-400 mb-1">✅ Loaded: {fileData.filename}</p>
              <p className="text-xs text-gray-400">{fileData.rows.toLocaleString()} rows × {fileData.columns.length} cols</p>

              <div className="mt-4">
                <label className="text-xs font-semibold text-gray-300">Target Column (Optional)</label>
                <select
                  className="w-full mt-1 bg-background border border-border rounded p-2 text-sm text-white"
                  value={targetCol}
                  onChange={(e) => setTargetCol(e.target.value)}
                >
                  <option value="">-- Select Target --</option>
                  {fileData.columns.map((c: string) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>

              <div className="mt-4">
                <label className="text-xs font-semibold text-gray-300">Columns to Drop</label>
                <div className="mt-2 max-h-32 overflow-y-auto space-y-1 bg-background border border-border rounded p-2 custom-scrollbar">
                  {fileData.columns.map((c: string) => (
                    <label key={c} className="flex items-center space-x-2 text-sm text-gray-300 hover:text-white cursor-pointer select-none">
                      <input
                        type="checkbox"
                        className="rounded border-gray-600 bg-surface text-primary focus:ring-primary h-3 w-3"
                        checked={colsToDrop.includes(c)}
                        onChange={(e) => {
                          if (e.target.checked) setColsToDrop([...colsToDrop, c]);
                          else setColsToDrop(colsToDrop.filter(col => col !== c));
                        }}
                      />
                      <span>{c}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="mt-4">
                <label className="text-xs font-semibold text-gray-300">Feature Columns</label>
                <div className="mt-2 max-h-32 overflow-y-auto space-y-1 bg-background border border-border rounded p-2 custom-scrollbar">
                  {fileData.columns.filter((c: string) => !colsToDrop.includes(c) && c !== targetCol).map((c: string) => (
                    <label key={`feat-${c}`} className="flex items-center space-x-2 text-sm text-gray-300 hover:text-white cursor-pointer select-none">
                      <input
                        type="checkbox"
                        className="rounded border-gray-600 bg-surface text-primary focus:ring-primary h-3 w-3"
                        checked={featureCols.includes(c)}
                        onChange={(e) => {
                          if (e.target.checked) setFeatureCols([...featureCols, c]);
                          else setFeatureCols(featureCols.filter(col => col !== c));
                        }}
                      />
                      <span>{c}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto space-y-2">
          {menuItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveProblem(item.id)}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${activeProblem === item.id
                ? 'bg-primary/10 text-primary border border-primary/20'
                : 'text-gray-400 hover:bg-surface hover:text-gray-200'
                }`}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 ml-80 p-10">
        {!fileData ? (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-lg mx-auto">
            <h2 className="text-3xl font-bold mb-4">Welcome to Smart ML Dashboard</h2>
            <p className="text-gray-400 text-lg">
              To get started, please upload your dataset using the upload box in the sidebar.
              This dashboard provides interactive machine learning analytics.
            </p>
          </div>
        ) : (
          <div className="max-w-6xl mx-auto space-y-6">
            {activeProblem === 1 && <Problem1 filename={fileData.filename} targetCol={targetCol || 'cnt'} featureCols={featureCols.filter(c => !colsToDrop.includes(c) && c !== (targetCol || 'cnt'))} />}
            {activeProblem === 2 && <Problem2 filename={fileData.filename} featureCols={featureCols.filter(c => !colsToDrop.includes(c) && c !== (targetCol || 'cnt'))} />}
            {activeProblem === 3 && <Problem3 filename={fileData.filename} targetCol={targetCol || 'cnt'} featureCols={featureCols.filter(c => !colsToDrop.includes(c) && c !== (targetCol || 'cnt'))} />}
            {activeProblem === 4 && <Problem4 filename={fileData.filename} targetCol={targetCol || 'cnt'} featureCols={featureCols.filter(c => !colsToDrop.includes(c) && c !== (targetCol || 'cnt'))} />}
            {activeProblem === 5 && <Problem5 filename={fileData.filename} targetCol={targetCol || 'cnt'} featureCols={featureCols.filter(c => !colsToDrop.includes(c) && c !== (targetCol || 'cnt'))} />}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
