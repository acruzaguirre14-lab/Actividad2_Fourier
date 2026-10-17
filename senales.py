"""
Actividad formativa 2: Simulación y análisis de señales con la Transformada de Fourier

Estructura del código:
    1. Definición de las señales en el dominio del tiempo
    2. Cálculo de la Transformada de Fourier (np.fft.fft)
    3. Visualización de resultados (señal, magnitud y fase)
    4. Análisis de propiedades (linealidad, desplazamiento en el tiempo,
       escalamiento) y aplicación real (filtrado de ruido)

Uso:
    python fourier_senales.py
Las figuras se guardan en la carpeta ./figuras
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # permite generar figuras sin ventana
import matplotlib.pyplot as plt

os.makedirs("figuras", exist_ok=True)

# ----------------------------------------------------------------------
# DEFINICIÓN DE LAS SEÑALES EN EL DOMINIO DEL TIEMPO
# ----------------------------------------------------------------------
fs = 1000                      # frecuencia de muestreo (Hz)
T = 2.0                        # duración (s)
N = int(fs * T)                # número de muestras
t = np.arange(N) / fs - T / 2  # eje de tiempo centrado en 0: [-1, 1) s


def pulso_rectangular(t, ancho=0.2):
    """Pulso de amplitud 1 entre -ancho/2 y ancho/2."""
    return (np.abs(t) <= ancho / 2).astype(float)


def escalon(t):
    """Función escalón unitario u(t)."""
    return (t >= 0).astype(float)


def senoidal(t, f0=10.0, amp=1.0):
    """Señal senoidal de frecuencia f0 (Hz)."""
    return amp * np.sin(2 * np.pi * f0 * t)


x_rect = pulso_rectangular(t)
x_step = escalon(t)
x_sin = senoidal(t, f0=10)

# ----------------------------------------------------------------------
# CÁLCULO DE LA TRANSFORMADA DE FOURIER
# ----------------------------------------------------------------------
NFFT = 2 ** 15                 # zero-padding: más puntos en frecuencia -> espectro más suave
freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1 / fs))  # eje de frecuencias centrado en 0


def espectro(x):
    """Devuelve la FFT centrada en 0 y escalada por 1/fs (aproxima la TF continua).

    Detalles importantes:
    - Se rellena con ceros a ambos lados (zero-padding) hasta NFFT puntos.
    - ifftshift coloca la muestra t=0 en el índice 0 antes de la FFT. Sin esto,
      la FFT asume que la señal empieza en t=0 y aparece un término de fase
      lineal extra que no corresponde a la señal real.
    """
    pad = (NFFT - len(x)) // 2
    x_pad = np.pad(x, (pad, NFFT - len(x) - pad))
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x_pad))) / fs


def fase_limpia(X, umbral=1e-3):
    """Fase en radianes; se pone en 0 donde la magnitud es despreciable
    (ahí la fase es ruido numérico y ensucia la gráfica)."""
    ang = np.angle(X)
    ang[np.abs(X) < umbral * np.max(np.abs(X))] = 0
    return ang


# ----------------------------------------------------------------------
# VISUALIZACIÓN DE RESULTADOS
# ----------------------------------------------------------------------
def graficar_senal_y_espectro(x, titulo, archivo, xlim_f=60, xlim_t=None, umbral_fase=1e-3):
    """Una figura con 3 paneles: señal, magnitud y fase."""
    X = espectro(x)
    fig, ax = plt.subplots(3, 1, figsize=(9, 9))

    ax[0].plot(t, x, color="tab:blue")
    ax[0].set_title(f"{titulo} - dominio del tiempo")
    ax[0].set_xlabel("Tiempo (s)")
    ax[0].set_ylabel("Amplitud")
    if xlim_t:
        ax[0].set_xlim(*xlim_t)
    ax[0].grid(True)

    ax[1].plot(freqs, np.abs(X), color="tab:red")
    ax[1].set_title("Magnitud del espectro |X(f)|")
    ax[1].set_xlabel("Frecuencia (Hz)")
    ax[1].set_ylabel("Magnitud")
    ax[1].set_xlim(-xlim_f, xlim_f)
    ax[1].grid(True)

    ax[2].plot(freqs, fase_limpia(X, umbral_fase), color="tab:green")
    ax[2].set_title("Fase del espectro ∠X(f)")
    ax[2].set_xlabel("Frecuencia (Hz)")
    ax[2].set_ylabel("Fase (rad)")
    ax[2].set_xlim(-xlim_f, xlim_f)
    ax[2].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join("figuras", archivo), dpi=150)
    plt.close(fig)


graficar_senal_y_espectro(x_rect, "Pulso rectangular (ancho 0.2 s)", "01_pulso_rectangular.png",
                          xlim_f=40, xlim_t=(-0.5, 0.5))
graficar_senal_y_espectro(x_step, "Función escalón u(t)", "02_escalon.png", xlim_f=40, umbral_fase=0.05)
graficar_senal_y_espectro(x_sin, "Senoidal de 10 Hz", "03_senoidal.png", xlim_f=40, xlim_t=(-0.5, 0.5),
                          umbral_fase=0.05)

# Comparación cuantitativa con la teoría
# TF del pulso: X(f) = ancho * sinc(ancho * f)   (np.sinc(x) = sin(pi x)/(pi x))
X_rect = espectro(x_rect)
X_teorica = 0.2 * np.sinc(0.2 * freqs)
error_rect = np.max(np.abs(np.abs(X_rect) - np.abs(X_teorica)))
print(f"[Pulso] error máximo vs sinc teórica: {error_rect:.4f}")

# Frecuencia dominante de la senoidal
f_pico = freqs[np.argmax(np.abs(espectro(x_sin)))]
print(f"[Senoidal] frecuencia dominante detectada: {abs(f_pico):.2f} Hz (esperada: 10 Hz)")

# ----------------------------------------------------------------------
# ANÁLISIS DE PROPIEDADES
# ----------------------------------------------------------------------

# 4.1 LINEALIDAD:  F{a x1 + b x2} = a F{x1} + b F{x2}
a, b = 2.0, 3.0
x1, x2 = x_rect, x_sin
X_combinada = espectro(a * x1 + b * x2)
X_suma = a * espectro(x1) + b * espectro(x2)
err_lin = np.max(np.abs(X_combinada - X_suma))
print(f"[Linealidad] error máximo: {err_lin:.2e}")

fig, ax = plt.subplots(2, 1, figsize=(9, 7))
ax[0].plot(t, a * x1 + b * x2)
ax[0].set_title("Linealidad: y(t) = 2·pulso(t) + 3·sen(2π·10t)")
ax[0].set_xlabel("Tiempo (s)")
ax[0].set_xlim(-0.5, 0.5)
ax[0].grid(True)
ax[1].plot(freqs, np.abs(X_combinada), label="F{a·x1 + b·x2}", linewidth=3, alpha=0.6)
ax[1].plot(freqs, np.abs(X_suma), "--", label="a·F{x1} + b·F{x2}", color="k")
ax[1].set_xlim(-40, 40)
ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].set_ylabel("Magnitud")
ax[1].set_title(f"Ambos espectros coinciden (error máx = {err_lin:.1e})")
ax[1].legend()
ax[1].grid(True)
fig.tight_layout()
fig.savefig("figuras/04_linealidad.png", dpi=150)
plt.close(fig)

# DESPLAZAMIENTO EN EL TIEMPO:  x(t - t0)  <->  X(f) e^{-j 2π f t0}
# La magnitud NO cambia; la fase varía linealmente con la frecuencia.
t0 = 0.1
x_desp = pulso_rectangular(t - t0)
X_orig = espectro(x_rect)
X_desp = espectro(x_desp)
X_pred = X_orig * np.exp(-1j * 2 * np.pi * freqs * t0)
print(f"[Desplazamiento] diferencia de magnitudes: {np.max(np.abs(np.abs(X_desp) - np.abs(X_orig))):.2e}")
print(f"[Desplazamiento] error vs. predicción teórica: {np.max(np.abs(X_desp - X_pred)):.2e}")

fig, ax = plt.subplots(3, 1, figsize=(9, 9))
ax[0].plot(t, x_rect, label="x(t)")
ax[0].plot(t, x_desp, "--", label=f"x(t - {t0})")
ax[0].set_xlim(-0.5, 0.5)
ax[0].set_title("Desplazamiento en el tiempo")
ax[0].set_xlabel("Tiempo (s)")
ax[0].legend()
ax[0].grid(True)
ax[1].plot(freqs, np.abs(X_orig), label="original", linewidth=3, alpha=0.6)
ax[1].plot(freqs, np.abs(X_desp), "--", label="desplazada", color="k")
ax[1].set_xlim(-40, 40)
ax[1].set_title("Magnitud: idéntica en ambas señales")
ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].legend()
ax[1].grid(True)
ax[2].plot(freqs, fase_limpia(X_orig), label="original")
ax[2].plot(freqs, fase_limpia(X_desp), label="desplazada")
ax[2].set_xlim(-40, 40)
ax[2].set_title("Fase: el desplazamiento agrega un término lineal -2π f t0")
ax[2].set_xlabel("Frecuencia (Hz)")
ax[2].set_ylabel("Fase (rad)")
ax[2].legend()
ax[2].grid(True)
fig.tight_layout()
fig.savefig("figuras/05_desplazamiento_tiempo.png", dpi=150)
plt.close(fig)

# ESCALAMIENTO:  x(a t)  <->  (1/|a|) X(f/a)
# Comprimir en el tiempo (a>1) ensancha el espectro; expandir lo estrecha.
a_esc = 2.0
x_comp = pulso_rectangular(a_esc * t)       # pulso de ancho 0.1 s
X_comp = espectro(x_comp)
X_esc_teo = (1 / a_esc) * 0.2 * np.sinc(0.2 * freqs / a_esc)
print(f"[Escalamiento] error máx. vs teoría: {np.max(np.abs(np.abs(X_comp) - np.abs(X_esc_teo))):.4f}")

fig, ax = plt.subplots(2, 1, figsize=(9, 7))
ax[0].plot(t, x_rect, label="x(t): ancho 0.2 s")
ax[0].plot(t, x_comp, label="x(2t): ancho 0.1 s")
ax[0].set_xlim(-0.5, 0.5)
ax[0].set_title("Escalamiento: señal comprimida en el tiempo")
ax[0].set_xlabel("Tiempo (s)")
ax[0].legend()
ax[0].grid(True)
ax[1].plot(freqs, np.abs(espectro(x_rect)), label="|X(f)|")
ax[1].plot(freqs, np.abs(X_comp), label="|F{x(2t)}|")
ax[1].set_xlim(-60, 60)
ax[1].set_title("Más corta en el tiempo -> espectro más ancho y de menor amplitud")
ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].legend()
ax[1].grid(True)
fig.tight_layout()
fig.savefig("figuras/06_escalamiento.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# APLICACIÓN REAL: ELIMINAR RUIDO DE UNA SEÑAL CON FILTRO EN FRECUENCIA
# ----------------------------------------------------------------------
# Escenario: un sensor mide una vibración de 5 Hz + 12 Hz contaminada con
# ruido de alta frecuencia (p. ej. ruido eléctrico de 120 Hz y ruido aleatorio).
rng = np.random.default_rng(42)  # semilla fija para reproducibilidad
limpia = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
ruidosa = limpia + 0.8 * np.sin(2 * np.pi * 120 * t) + 0.3 * rng.standard_normal(N)

F = np.fft.fft(ruidosa)
f_raw = np.fft.fftfreq(N, d=1 / fs)
F_filtrada = F.copy()
F_filtrada[np.abs(f_raw) > 30] = 0           # filtro pasa-bajas ideal con corte en 30 Hz
recuperada = np.real(np.fft.ifft(F_filtrada))

ecm_antes = np.mean((ruidosa - limpia) ** 2)
ecm_despues = np.mean((recuperada - limpia) ** 2)
print(f"[Aplicación] error cuadrático medio antes: {ecm_antes:.4f}  después: {ecm_despues:.4f}")

fig, ax = plt.subplots(3, 1, figsize=(9, 9))
ax[0].plot(t, ruidosa, color="gray", label="con ruido")
ax[0].plot(t, limpia, color="tab:blue", label="original")
ax[0].set_xlim(0, 0.6)
ax[0].set_title("Señal del sensor con ruido (dominio del tiempo)")
ax[0].set_xlabel("Tiempo (s)")
ax[0].legend()
ax[0].grid(True)
ax[1].plot(np.fft.fftshift(f_raw), np.fft.fftshift(np.abs(F)) / N)
ax[1].axvline(30, color="r", linestyle="--")
ax[1].axvline(-30, color="r", linestyle="--", label="corte 30 Hz")
ax[1].set_xlim(-200, 200)
ax[1].set_title("Espectro: picos en 5 y 12 Hz (señal) y 120 Hz (interferencia)")
ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].legend()
ax[1].grid(True)
ax[2].plot(t, recuperada, color="tab:green", label="filtrada")
ax[2].plot(t, limpia, "k--", alpha=0.6, label="original")
ax[2].set_xlim(0, 0.6)
ax[2].set_title("Señal recuperada tras filtrar en frecuencia")
ax[2].set_xlabel("Tiempo (s)")
ax[2].legend()
ax[2].grid(True)
fig.tight_layout()
fig.savefig("figuras/07_aplicacion_filtrado.png", dpi=150)
plt.close(fig)

print("Listo. Figuras guardadas en ./figuras")
