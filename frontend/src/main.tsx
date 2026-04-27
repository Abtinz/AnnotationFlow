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

type RuntimeConfig = {
  ROBOFLOW_API_KEY: string;
  ROBOFLOW_WORKSPACE_NAME: string;
  ROBOFLOW_WORKFLOW_ID: string;
  ROBOFLOW_USE_CACHE: string;
  ROBOFLOW_CONFIDENCE: string;
  TRAIN_RATIO: string;
  VAL_RATIO: string;
  TEST_RATIO: string;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const DEFAULT_RUNTIME_CONFIG: RuntimeConfig = {
  ROBOFLOW_API_KEY: "",
  ROBOFLOW_WORKSPACE_NAME: "",
  ROBOFLOW_WORKFLOW_ID: "",
  ROBOFLOW_USE_CACHE: "",
  ROBOFLOW_CONFIDENCE: "",
  TRAIN_RATIO: "",
  VAL_RATIO: "",
  TEST_RATIO: "",
};

function App() {
  const [files, setFiles] = React.useState<File[]>([]);
  const [runtimeConfig, setRuntimeConfig] = React.useState<RuntimeConfig>(DEFAULT_RUNTIME_CONFIG);
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

  function updateRuntimeConfig(key: keyof RuntimeConfig, value: string) {
    setRuntimeConfig((current) => ({ ...current, [key]: value }));
  }

  function appendRuntimeConfig(formData: FormData) {
    const payload = Object.fromEntries(
      Object.entries(runtimeConfig).filter(([, value]) => value.trim() !== ""),
    );
    if (Object.keys(payload).length > 0) {
      formData.append("config", JSON.stringify(payload));
    }
  }

  async function loadLogs(nextJobId: string) {
    const logResponse = await fetch(`${API_BASE_URL}/jobs/${nextJobId}/logs`);
    if (logResponse.ok) {
      const logPayload = await logResponse.json();
      setLogs(logPayload.logs ?? []);
    }
  }

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
    appendRuntimeConfig(formData);

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
      await loadLogs(payload.job_id);
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

      <section className="input-grid">
        <div className="panel input-panel">
          <div className="panel-top">
            <span className="panel-label">Images</span>
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

        <div className="panel config-panel">
          <span className="panel-label">Pipeline Inputs</span>
          <div className="form-grid">
            <ConfigField
              label="ROBOFLOW_API_KEY"
              type="password"
              value={runtimeConfig.ROBOFLOW_API_KEY}
              onChange={(value) => updateRuntimeConfig("ROBOFLOW_API_KEY", value)}
            />
            <ConfigField
              label="ROBOFLOW_WORKSPACE_NAME"
              value={runtimeConfig.ROBOFLOW_WORKSPACE_NAME}
              onChange={(value) => updateRuntimeConfig("ROBOFLOW_WORKSPACE_NAME", value)}
            />
            <ConfigField
              label="ROBOFLOW_WORKFLOW_ID"
              value={runtimeConfig.ROBOFLOW_WORKFLOW_ID}
              onChange={(value) => updateRuntimeConfig("ROBOFLOW_WORKFLOW_ID", value)}
            />
            <label className="field">
              <span>ROBOFLOW_USE_CACHE</span>
              <select
                value={runtimeConfig.ROBOFLOW_USE_CACHE}
                onChange={(event) => updateRuntimeConfig("ROBOFLOW_USE_CACHE", event.target.value)}
              >
                <option value="">Use .env</option>
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </label>
            <ConfigField
              label="ROBOFLOW_CONFIDENCE"
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={runtimeConfig.ROBOFLOW_CONFIDENCE}
              onChange={(value) => updateRuntimeConfig("ROBOFLOW_CONFIDENCE", value)}
            />
            <div className="ratio-row">
              <ConfigField
                label="TRAIN_RATIO"
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={runtimeConfig.TRAIN_RATIO}
                onChange={(value) => updateRuntimeConfig("TRAIN_RATIO", value)}
              />
              <ConfigField
                label="VAL_RATIO"
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={runtimeConfig.VAL_RATIO}
                onChange={(value) => updateRuntimeConfig("VAL_RATIO", value)}
              />
              <ConfigField
                label="TEST_RATIO"
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={runtimeConfig.TEST_RATIO}
                onChange={(value) => updateRuntimeConfig("TEST_RATIO", value)}
              />
            </div>
          </div>
        </div>
      </section>

      <section className="monitor-grid">
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
      </section>
    </main>
  );
}

function ConfigField({
  label,
  value,
  onChange,
  type = "text",
  step,
  min,
  max,
}: {
  label: keyof RuntimeConfig;
  value: string;
  onChange: (value: string) => void;
  type?: string;
  step?: string;
  min?: string;
  max?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type={type}
        value={value}
        step={step}
        min={min}
        max={max}
        placeholder="Use .env"
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
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
