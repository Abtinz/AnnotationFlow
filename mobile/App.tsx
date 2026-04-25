import { StatusBar } from "expo-status-bar";
import * as DocumentPicker from "expo-document-picker";
import { useEffect, useMemo, useState } from "react";
import { Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View } from "react-native";

type PickedAsset = {
  name: string;
  uri: string;
  mimeType?: string;
};

type JobSummary = {
  processed_images?: number;
  duplicate_images?: number;
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

const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function App() {
  const [assets, setAssets] = useState<PickedAsset[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [summary, setSummary] = useState<JobSummary | null>(null);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = assets.length > 0 && status !== "uploading" && status !== "processing";
  const datasetUrl = useMemo(() => (jobId ? `${API_BASE_URL}/jobs/${jobId}/dataset` : null), [jobId]);

  useEffect(() => {
    if (!jobId || status === "completed" || status === "failed") {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const [jobResponse, logsResponse] = await Promise.all([
          fetch(`${API_BASE_URL}/jobs/${jobId}`),
          fetch(`${API_BASE_URL}/jobs/${jobId}/logs`),
        ]);
        if (jobResponse.ok) {
          const jobPayload = await jobResponse.json();
          setStatus(jobPayload.status);
          setSummary(jobPayload.summary);
        }
        if (logsResponse.ok) {
          const logsPayload = await logsResponse.json();
          setLogs(logsPayload.logs ?? []);
        }
      } catch (pollError) {
        setError(pollError instanceof Error ? pollError.message : "Unable to poll job logs");
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [jobId, status]);

  const pickImages = async () => {
    setError(null);
    const result = await DocumentPicker.getDocumentAsync({
      type: "image/*",
      multiple: true,
      copyToCacheDirectory: true,
    });

    if (result.canceled) {
      return;
    }

    setAssets(
      result.assets.map((asset) => ({
        name: asset.name,
        uri: asset.uri,
        mimeType: asset.mimeType,
      })),
    );
  };

  const submitJob = async () => {
    if (!canSubmit) {
      return;
    }

    setError(null);
    setStatus("uploading");
    setLogs([]);
    setSummary(null);

    const formData = new FormData();
    assets.forEach((asset) => {
      formData.append("files", {
        uri: asset.uri,
        name: asset.name,
        type: asset.mimeType ?? "image/jpeg",
      } as unknown as Blob);
    });

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
  };

  return (
    <SafeAreaView style={styles.screen}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <Text style={styles.eyebrow}>AnnotationFlow</Text>
          <Text style={styles.title}>YOLO Dataset Builder</Text>
          <Text style={styles.subtitle}>
            Upload images, watch the pipeline logs, and export a Roboflow-powered object-detection dataset.
          </Text>
        </View>

        <View style={styles.toolbar}>
          <Pressable style={styles.primaryButton} onPress={pickImages}>
            <Text style={styles.primaryButtonText}>Select Images</Text>
          </Pressable>
          <Pressable
            style={[styles.secondaryButton, !canSubmit && styles.disabledButton]}
            disabled={!canSubmit}
            onPress={submitJob}
          >
            <Text style={styles.secondaryButtonText}>Start Job</Text>
          </Pressable>
        </View>

        <View style={styles.panel}>
          <Text style={styles.panelTitle}>Selected</Text>
          <Text style={styles.metric}>{assets.length} image(s)</Text>
          {assets.slice(0, 6).map((asset) => (
            <Text key={asset.uri} style={styles.fileName}>
              {asset.name}
            </Text>
          ))}
        </View>

        <View style={styles.panel}>
          <Text style={styles.panelTitle}>Status</Text>
          <Text style={styles.metric}>{status}</Text>
          {summary ? (
            <Text style={styles.summary}>
              Processed {summary.processed_images ?? 0}, duplicates {summary.duplicate_images ?? 0}, classes{" "}
              {(summary.classes ?? []).join(", ") || "none"}, split {summary.train_count ?? 0}/
              {summary.valid_count ?? 0}/{summary.test_count ?? 0}
            </Text>
          ) : null}
          {datasetUrl ? <Text style={styles.datasetUrl}>{datasetUrl}</Text> : null}
          {error ? <Text style={styles.error}>{error}</Text> : null}
        </View>

        <View style={styles.logPanel}>
          <Text style={styles.panelTitle}>Logs</Text>
          {logs.length === 0 ? <Text style={styles.emptyLog}>No logs yet.</Text> : null}
          {logs.map((log, index) => (
            <Text key={`${log.timestamp}-${index}`} style={styles.logLine}>
              [{log.level}] {log.message}
            </Text>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#f5f2ea",
  },
  content: {
    padding: 24,
    gap: 18,
  },
  header: {
    gap: 12,
    maxWidth: 720,
  },
  eyebrow: {
    color: "#37635b",
    fontSize: 13,
    fontWeight: "700",
    letterSpacing: 0,
    textTransform: "uppercase",
  },
  title: {
    color: "#171717",
    fontSize: 38,
    fontWeight: "800",
    letterSpacing: 0,
  },
  subtitle: {
    color: "#4b4b4b",
    fontSize: 17,
    lineHeight: 24,
  },
  toolbar: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  primaryButton: {
    backgroundColor: "#171717",
    borderRadius: 8,
    paddingHorizontal: 18,
    paddingVertical: 13,
  },
  primaryButtonText: {
    color: "#ffffff",
    fontSize: 15,
    fontWeight: "800",
  },
  secondaryButton: {
    backgroundColor: "#d9eadf",
    borderColor: "#315b50",
    borderRadius: 8,
    borderWidth: 1,
    paddingHorizontal: 18,
    paddingVertical: 13,
  },
  secondaryButtonText: {
    color: "#183c34",
    fontSize: 15,
    fontWeight: "800",
  },
  disabledButton: {
    opacity: 0.45,
  },
  panel: {
    backgroundColor: "#ffffff",
    borderColor: "#ded8cb",
    borderRadius: 8,
    borderWidth: 1,
    padding: 16,
    gap: 8,
  },
  logPanel: {
    backgroundColor: "#151713",
    borderRadius: 8,
    padding: 16,
    gap: 8,
    minHeight: 220,
  },
  panelTitle: {
    color: "#37635b",
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 0,
    textTransform: "uppercase",
  },
  metric: {
    color: "#171717",
    fontSize: 24,
    fontWeight: "800",
  },
  fileName: {
    color: "#4b4b4b",
    fontSize: 14,
  },
  summary: {
    color: "#343434",
    fontSize: 15,
    lineHeight: 22,
  },
  datasetUrl: {
    color: "#315b50",
    fontSize: 13,
  },
  error: {
    color: "#b42318",
    fontSize: 14,
    fontWeight: "700",
  },
  emptyLog: {
    color: "#8f978e",
    fontSize: 14,
  },
  logLine: {
    color: "#e8ede3",
    fontFamily: "monospace",
    fontSize: 13,
    lineHeight: 20,
  },
});
