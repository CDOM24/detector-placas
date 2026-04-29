import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Speech from 'expo-speech';
import React, { useRef, useState, useEffect } from 'react';
import {
  ActivityIndicator,
  Alert,
  Animated,
  Dimensions,
  Image,
  Platform,
  Pressable,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

const { width: SCREEN_W } = Dimensions.get('window');

// ─── Paleta de colores ────────────────────────────────────────────
const C = {
  bg:       '#0a0a0f',
  surface:  '#111118',
  card:     '#13131f',
  border:   '#1e1e2e',
  accent:   '#00e5b4',
  accent2:  '#7b61ff',
  warn:     '#ff6b35',
  text:     '#e8e8f0',
  muted:    '#5a5a7a',
};

// ─── Componente principal ─────────────────────────────────────────
export default function CameraScreen() {
  const cameraRef                         = useRef<any>(null);
  const [permission, requestPermission]   = useCameraPermissions();
  const [serverUrl, setServerUrl]         = useState('');
  const [image, setImage]                 = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [plates, setPlates]               = useState<string[]>([]);
  const [loading, setLoading]             = useState(false);
  const [lastMsg, setLastMsg]             = useState('');
  const fadeAnim                          = useRef(new Animated.Value(0)).current;
  const pulseAnim                         = useRef(new Animated.Value(1)).current;

  // Pulso del botón cuando hay cámara lista
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.04, duration: 900, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1.0,  duration: 900, useNativeDriver: true }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, []);

  const fadeIn = () => {
    fadeAnim.setValue(0);
    Animated.timing(fadeAnim, { toValue: 1, duration: 400, useNativeDriver: true }).start();
  };

  // ── Permisos ────────────────────────────────────────────────────
  useEffect(() => { if (!permission) requestPermission(); }, [permission]);

  if (!permission) {
    return (
      <View style={s.centered}>
        <Text style={s.mutedText}>Solicitando permisos de cámara…</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={s.centered}>
        <Text style={[s.mutedText, { marginBottom: 16 }]}>Se necesita acceso a la cámara.</Text>
        <Pressable style={s.btnPrimary} onPress={requestPermission}>
          <Text style={s.btnText}>Conceder permiso</Text>
        </Pressable>
      </View>
    );
  }

  // ── Captura y envío ─────────────────────────────────────────────
  const handleCapture = async () => {
    if (!cameraRef.current) return;

    const url = serverUrl.trim().replace(/\/$/, '');
    if (!url) {
      Alert.alert('Falta la URL', 'Ingresa la URL del servidor ngrok antes de tomar la foto.');
      return;
    }

    try {
      setLoading(true);
      setPlates([]);
      setProcessedImage(null);
      setLastMsg('');

      const photo = await cameraRef.current.takePictureAsync({ base64: true });
      setImage(photo.uri);

      const endpoint = `${url}/predict/`;
      const formBody = new URLSearchParams();
      formBody.append('image_base64', photo.base64 ?? '');

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          Accept: 'application/json',
        },
        body: formBody.toString(),
      });

      if (!response.ok) {
        Alert.alert('Error HTTP', `Código: ${response.status}`);
        if (Platform.OS !== 'web') Speech.speak('Error de conexión con el servidor.');
        return;
      }

      const data = await response.json();

      if (data?.placas && data.placas.length > 0) {
        setPlates(data.placas);
        setLastMsg(`${data.placas.length} placa${data.placas.length > 1 ? 's' : ''} detectada${data.placas.length > 1 ? 's' : ''}`);
        if (data.image) setProcessedImage(`data:image/jpeg;base64,${data.image}`);
        fadeIn();

        const txt =
          data.placas.length === 1
            ? `Placa detectada: ${data.placas[0].split('').join(' ')}`
            : `Se detectaron ${data.placas.length} placas: ${data.placas.join(', ')}`;
        if (Platform.OS !== 'web') Speech.speak(txt, { language: 'es-ES' });

      } else if (data?.error) {
        setLastMsg('Error: ' + data.error);
        Alert.alert('Error del servidor', data.error);
      } else {
        setLastMsg('No se detectaron placas');
        if (Platform.OS !== 'web') Speech.speak('No se detectaron placas.');
      }
    } catch (err: any) {
      Alert.alert('Sin conexión', 'No se pudo conectar al servidor. Verifica la URL.');
      if (Platform.OS !== 'web') Speech.speak('No se pudo conectar al servidor.');
    } finally {
      setLoading(false);
    }
  };

  // ── Render ──────────────────────────────────────────────────────
  return (
    <ScrollView style={s.scroll} contentContainerStyle={s.container} showsVerticalScrollIndicator={false}>
      <StatusBar barStyle="light-content" backgroundColor={C.bg} />

      {/* Header */}
      <View style={s.header}>
        <View style={s.logoMark}>
          <Text style={{ fontSize: 22 }}>🚗</Text>
        </View>
        <View>
          <Text style={s.title}>Detector de <Text style={{ color: C.accent }}>Placas</Text></Text>
          <Text style={s.subtitle}>YOLOv8 · EasyOCR · ngrok</Text>
        </View>
        <View style={s.badge}>
          <Text style={s.badgeText}>LIVE</Text>
        </View>
      </View>

      {/* URL input */}
      <View style={s.card}>
        <Text style={s.label}>URL DEL SERVIDOR</Text>
        <TextInput
          style={s.input}
          placeholder="https://xxxx.ngrok-free.app"
          placeholderTextColor={C.muted}
          value={serverUrl}
          onChangeText={setServerUrl}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
        />
        <Text style={s.hint}>Copia la URL que generó Colab con ngrok</Text>
      </View>

      {/* Camera */}
      <View style={s.cameraWrap}>
        <View style={s.cameraCorner} />
        <CameraView ref={cameraRef} style={s.camera} facing="back" />
        <View style={[s.cameraCorner, { right: 0, left: undefined }]} />
      </View>

      {/* Capture button */}
      <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
        <Pressable
          style={({ pressed }) => [s.btnCapture, pressed && { opacity: 0.85 }]}
          onPress={handleCapture}
          disabled={loading}
        >
          {loading
            ? <ActivityIndicator color={C.bg} size="small" />
            : <Text style={s.btnCaptureText}>⚡  DETECTAR PLACA</Text>
          }
        </Pressable>
      </Animated.View>

      {loading && (
        <View style={s.loaderRow}>
          <Text style={s.mutedText}>Procesando con YOLOv8 + OCR…</Text>
        </View>
      )}

      {/* Results */}
      {(plates.length > 0 || lastMsg) && (
        <Animated.View style={[s.resultsWrap, { opacity: fadeAnim }]}>

          {/* Status bar */}
          <View style={s.resultsHeader}>
            <Text style={s.resultsHeaderText}>RESULTADO</Text>
            {lastMsg ? <Text style={[s.badgeText, { color: C.accent }]}>{lastMsg}</Text> : null}
          </View>

          {/* Plate cards */}
          {plates.length > 0 && (
            <View style={s.platesRow}>
              {plates.map((p, i) => (
                <View key={i} style={s.plateCard}>
                  <Text style={s.plateMini}>PLACA DETECTADA</Text>
                  <Text style={s.plateNumber}>{p}</Text>
                </View>
              ))}
            </View>
          )}

          {plates.length === 0 && lastMsg && (
            <View style={s.noDetect}>
              <Text style={{ color: C.warn, fontFamily: 'monospace', fontSize: 13 }}>
                ⚠️ {lastMsg}
              </Text>
            </View>
          )}

          {/* Processed image */}
          {processedImage && (
            <View style={s.resultImgWrap}>
              <View style={s.resultImgLabel}>
                <View style={s.dot} />
                <Text style={s.mutedText}>IMAGEN PROCESADA</Text>
              </View>
              <Image
                source={{ uri: processedImage }}
                style={s.resultImg}
                resizeMode="contain"
              />
            </View>
          )}
        </Animated.View>
      )}

      {/* Captured image */}
      {image && !processedImage && (
        <View style={[s.resultImgWrap, { marginTop: 16 }]}>
          <View style={s.resultImgLabel}>
            <Text style={s.mutedText}>FOTO CAPTURADA</Text>
          </View>
          <Image source={{ uri: image }} style={s.resultImg} resizeMode="contain" />
        </View>
      )}

      <Text style={s.footer}>Desarrollado por <Text style={{ color: C.accent2 }}>Alfredo Díaz</Text></Text>
    </ScrollView>
  );
}

// ─── Estilos ──────────────────────────────────────────────────────
const s = StyleSheet.create({
  scroll:   { flex: 1, backgroundColor: C.bg },
  container:{ padding: 20, paddingTop: 52, paddingBottom: 60 },
  centered: { flex: 1, backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center', padding: 24 },

  // Header
  header: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 28 },
  logoMark: {
    width: 46, height: 46, borderRadius: 12,
    backgroundColor: C.accent2,
    alignItems: 'center', justifyContent: 'center',
  },
  title:    { fontSize: 20, fontWeight: '800', color: C.text, letterSpacing: -0.5 },
  subtitle: { fontSize: 11, color: C.muted, fontFamily: 'monospace', marginTop: 2 },
  badge: {
    marginLeft: 'auto',
    borderWidth: 1, borderColor: C.accent,
    borderRadius: 999, paddingHorizontal: 10, paddingVertical: 4,
  },
  badgeText: { fontSize: 10, color: C.accent, fontFamily: 'monospace', letterSpacing: 1 },

  // Card / input
  card: {
    backgroundColor: C.card, borderWidth: 1, borderColor: C.border,
    borderRadius: 16, padding: 16, marginBottom: 16,
  },
  label:  { fontSize: 10, color: C.muted, fontFamily: 'monospace', letterSpacing: 1.5, marginBottom: 8 },
  input:  {
    backgroundColor: C.surface, borderWidth: 1, borderColor: C.border,
    borderRadius: 10, paddingHorizontal: 14, paddingVertical: 11,
    color: C.text, fontSize: 14, fontFamily: 'monospace',
  },
  hint: { fontSize: 11, color: C.muted, marginTop: 8, fontFamily: 'monospace' },

  // Camera
  cameraWrap: { borderRadius: 16, overflow: 'hidden', marginBottom: 16, position: 'relative' },
  camera: { width: '100%', height: SCREEN_W * 0.75 },
  cameraCorner: {
    position: 'absolute', top: 12, left: 12,
    width: 24, height: 24,
    borderTopWidth: 2, borderLeftWidth: 2,
    borderColor: C.accent, borderRadius: 4, zIndex: 10,
  },

  // Buttons
  btnCapture: {
    borderRadius: 14, paddingVertical: 18,
    alignItems: 'center', justifyContent: 'center',
    marginBottom: 12,
    backgroundColor: C.accent,
  },
  btnCaptureText: { fontSize: 15, fontWeight: '800', color: C.bg, letterSpacing: 0.5 },
  btnPrimary: {
    backgroundColor: C.accent, borderRadius: 12,
    paddingHorizontal: 24, paddingVertical: 14,
  },
  btnText: { color: C.bg, fontWeight: '700', fontSize: 15 },

  // Loader
  loaderRow: { alignItems: 'center', marginBottom: 12 },
  mutedText: { color: C.muted, fontSize: 12, fontFamily: 'monospace' },

  // Results
  resultsWrap: { marginTop: 8 },
  resultsHeader: {
    flexDirection: 'row', alignItems: 'center',
    justifyContent: 'space-between', marginBottom: 14,
  },
  resultsHeaderText: { fontSize: 11, color: C.muted, fontFamily: 'monospace', letterSpacing: 1.5 },

  platesRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginBottom: 16 },
  plateCard: {
    flex: 1, minWidth: 140,
    backgroundColor: C.card,
    borderWidth: 1, borderColor: C.border,
    borderRadius: 14, padding: 20, alignItems: 'center',
    borderTopWidth: 3, borderTopColor: C.accent,
  },
  plateMini:   { fontSize: 9, color: C.muted, fontFamily: 'monospace', letterSpacing: 1.5, marginBottom: 8 },
  plateNumber: { fontSize: 30, fontWeight: '800', color: C.accent, letterSpacing: 2 },

  noDetect: {
    backgroundColor: C.card, borderWidth: 1, borderColor: C.border,
    borderRadius: 14, padding: 20, alignItems: 'center', marginBottom: 16,
  },

  resultImgWrap: {
    borderRadius: 14, overflow: 'hidden',
    borderWidth: 1, borderColor: C.border,
    marginBottom: 16,
  },
  resultImgLabel: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    backgroundColor: C.card, padding: 10, paddingHorizontal: 14,
  },
  dot: { width: 6, height: 6, borderRadius: 3, backgroundColor: C.accent },
  resultImg: { width: '100%', height: 220, backgroundColor: '#000' },

  footer: { textAlign: 'center', marginTop: 40, fontSize: 11, color: C.muted, fontFamily: 'monospace' },
});
