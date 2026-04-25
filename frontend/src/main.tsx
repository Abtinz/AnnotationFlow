import React from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type JobSummary = {
  processed_images?: number;
  duplicate_images?: number;
  failed_images?: number;
  skipped_images?: number;
  classes?: string[];
  train_count?: number;
  valid_count?: number;
  test_count?: number;
};

type LogLine = {
  timestamp: string;
  level: string;
  message: string;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

function App() {
  const [files, setFiles] = React.useState<File[]>([]);
  const [jobId, setJobId] = React.useState<string | null>(null);
  const [status, setStatus] = React.useState("idle");
  const [summary, setSummary] = React.useState<JobSummary | null>(null);
  const [logs, setLogs] = React.useState<LogLine[]>([]);
  const [error, setError] = React.useState<string | null>(null);

  const canRun = files.length > 0 && status !== "uploading" && status !== "processing";
  const datasetUrl = jobId ? `${API_BASE_URL}/jobs/${jobId}/dataset` : null;

  React.useEffect(() => {
    if (!jobId || status === "completed" || status === "failed") {
      return;
    }

    const interval = window.setInterval(async () => {
      try {
        const [jobResponse, logResponse] = await Promise.all([
          fetch(`${API_BASE_URL}/jobs/${jobId}`),
          fetch(`${API_BASE_URL}/jobs/${jobId}/logs`),
        ]);
        if (jobResponse.ok) {
          const jobPayload = await jobResponse.json();
          setStatus(jobPayload.status);
          setSummary(jobPayload.summary);
        }
        if (logResponse.ok) {
          const logPayload = await logResponse.json();
          setLogs(logPayload.logs ?? []);
        }
      } catch (pollError) {
        setError(pollError instanceof Error ? pollError.message : "Could not poll job status");
      }
    }, 1000);

    return () => window.clearInterval(interval);
  }, [jobId, status]);

  async function startJob() {
    if (!canRun) {
      return;
    }

    setError(null);
    setLogs([]);
    setSummary(null);
    setStatus("uploading");

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    try {
      const response = await fetch(`${API_BASE_URL}/jobs`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Upload failed with ${response.status}`);
      }
      const payload = await response.json();
      setJobId(payload.job_id);
      setStatus(payload.status);
      setSummary(payload.summary);
    } catch (uploadError) {
      setStatus("failed");
      setError(uploadError instanceof Error ? uploadError.message : "Upload failed");
    }
  }

  return (
    <main className="shell">
      <header className="hero">
        <div className="brand">
          <span className="mark" />
          <span>AnnotationFlow</span>
        </div>
        <h1>Object Detection Dataset Builder</h1>
        <p>Clean raw images, run Roboflow Workflow inference, and export a ready-to-train YOLO dataset.</p>
      </header>

      <section className="grid">
        <div className="panel input-panel">
          <div className="panel-top">
            <span className="panel-label">Input</span>
            <span className="pill">{status}</span>
          </div>
          <div className="count">{files.length}</div>
          <p className="muted">selected image(s)</p>

          <label className="file-drop">
            <input
              type="file"
              accept="image/*,.heic,.heif"
              multiple
              onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
            />
            <span>Select Images</span>
          </label>

          <button className="run-button" disabled={!canRun} onClick={startJob}>
            Run Pipeline
          </button>

          <div className="file-list">
            {files.length === 0 ? <span className="empty">No images selected.</span> : null}
            {files.slice(0, 9).map((file) => (
              <span className="file-name" key={`${file.name}-${file.size}`}>
                {file.name}
              </span>
            ))}
          </div>
        </div>

        <div className="panel dataset-panel">
          <span className="panel-label">Dataset</span>
          <div className="stats">
            <Stat label="processed" value={summary?.processed_images ?? 0} />
            <Stat label="duplicates" value={summary?.duplicate_images ?? 0} />
            <Stat label="failed" value={summary?.failed_images ?? 0} />
            <Stat label="skipped" value={summary?.skipped_images ?? 0} />
            <Stat label="train" value={summary?.train_count ?? 0} />
            <Stat label="valid" value={summary?.valid_count ?? 0} />
          </div>
          <p className="dataset-copy">
            Classes: {(summary?.classes ?? []).join(", ") || "none yet"} · Test: {summary?.test_count ?? 0}
          </p>
          {datasetUrl ? (
            <a className="dataset-link" href={datasetUrl}>
              Download dataset zip
            </a>
          ) : null}
          {error ? <p className="error">{error}</p> : null}
        </div>
      </section>

      <section className="logs">
        <div className="panel-top">
          <span className="log-title">System Logs</span>
          <span className="log-count">{logs.length} lines</span>
        </div>
        {logs.length === 0 ? <p className="empty-log">No logs yet.</p> : null}
        {logs.map((log, index) => (
          <p className="log-line" key={`${log.timestamp}-${index}`}>
            [{log.level}] {log.message}
          </p>
        ))}
      </section>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="stat">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
