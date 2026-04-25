import { StatusBar } from "expo-status-bar";
import { SafeAreaView, StyleSheet, Text, View } from "react-native";

export default function App() {
  return (
    <SafeAreaView style={styles.screen}>
      <StatusBar style="dark" />
      <View style={styles.header}>
        <Text style={styles.eyebrow}>AnnotationFlow</Text>
        <Text style={styles.title}>YOLO Dataset Builder</Text>
        <Text style={styles.subtitle}>
          Upload images, watch the pipeline logs, and export a Roboflow-powered object-detection dataset.
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#f7f5ef",
    padding: 24,
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
});

