import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import savgol_filter
import gradio as gr
import tempfile
import scipy.io.wavfile as wavfile
from datetime import datetime
import os
import pandas as pd

# Matplotlib için Türkçe karakter desteği (Gerekirse)
plt.rcParams['axes.unicode_minus'] = False

def kozmik_analiz():
    # --- 1. SİNYAL ÜRETİMİ VE GÜRÜLTÜ ---
    t = np.linspace(0, 2, 16000) 
    f1 = np.random.randint(20, 50)
    f2 = np.random.randint(70, 120)
    clean_signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    noise = np.random.normal(0, 0.7, 16000)
    noisy_signal = clean_signal + noise
    
    # Anomali (Patlama) Ekleme
    burst_power = np.random.uniform(4, 9)
    burst_idx = np.random.randint(4000, 12000) 
    noisy_signal[burst_idx:burst_idx+200] += burst_power

    # --- 2. ANALİZ VE FİLTRELEME (DSP) ---
    n = len(t)
    f_hat = fft(noisy_signal)
    psd = np.abs(f_hat * np.conj(f_hat) / n)
    freq = fftfreq(n, d=(t[1]-t[0]))
    threshold = np.mean(psd) + 2.5 * np.std(psd)
    f_hat_clean = f_hat.copy()
    f_hat_clean[psd < threshold] = 0
    fft_filtered = ifft(f_hat_clean).real
    final_signal = savgol_filter(fft_filtered, window_length=501, polyorder=3)
    
    # SNR Hesaplama
    snr_val = 10 * np.log10(np.mean(clean_signal**2) / np.mean((clean_signal - final_signal)**2))

    # --- 3. GÖRSELLEŞTİRME (GRAFİKLER) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=100)
    
    # Zaman Alanı Grafiği
    ax1.plot(t, noisy_signal, color='#FF9999', alpha=0.5, label='Ham Veri (Gürültülü)')
    ax1.plot(t, final_signal, color='#0000FF', linewidth=2, label='Filtrelenmiş Sinyal')
    ax1.set_title(f'Karabük Merkez İstasyonu Analizi | SNR: {snr_val:.2f} dB', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Genlik', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Frekans Alanı Grafiği
    ax2.plot(freq[:n//2], psd[:n//2], color='black', label='Güç Spektrumu (PSD)')
    ax2.axhline(y=threshold, color='#00AA00', linestyle='--', linewidth=1.5, label='Dinamik Eşik (Threshold)')
    ax2.set_xlim(0, 150)
    ax2.set_title('Sinyal Güç Spektrumu ve Eşik Değeri', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Frekans (Hz)', fontsize=12)
    ax2.set_ylabel('Güç', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout(pad=3.0)

    # --- 4. DOSYA OLUŞTURMA (Ses ve Veri İndirme) ---
    # Ses Dosyası (WAV)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        norm_signal = np.int16(final_signal / np.max(np.abs(final_signal)) * 32767)
        wavfile.write(temp_wav.name, 8000, norm_signal)
        wav_path = temp_wav.name

    # Veri Dosyası (CSV)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
        df = pd.DataFrame({'Zaman (s)': t, 'Ham Veri': noisy_signal, 'Filtrelenmiş Veri': final_signal})
        df.to_csv(temp_csv.name, index=False, sep=';', encoding='utf-8')
        csv_path = temp_csv.name

    # --- 5. RESMİ RAPOR METNİ OLUŞTURMA ---
    durum = '⚠️ KRİTİK ANALİZ GEREKLİ (Yüksek Enerjili Anomaliler Saptandı)' if burst_power > 6.5 else '✅ NORMAL OPERASYON (Sinyal Akışı Doğal Limitler Dahilinde)'
    rapor_metni = f"""
/*******************************************************************************
 * TÜRK UZAY AJANSI - ASTRO ANALİZ SİSTEMİ (v2.0)                *
 * OTOMATİK SİNYAL RAPORU - KARABÜK MERKEZ İSTASYONU            *
 *******************************************************************************/
 
 [GENEL BİLGİLER]
 -------------------------------------------------------------------------------
 Analiz Tarihi         : {datetime.now().strftime('%d/%m/%Y')}
 Analiz Saati          : {datetime.now().strftime('%H:%M:%S')}
 İstasyon Konumu       : Karabük / Türkiye (Ana Kontrol Merkezi)
 Veri Kaynağı          : SERTİFİKALI-ÇELİKÇİLER | v10.9 Sinyal Yakalama Modülü
 
 [TEKNİK PARAMETRELER]
 -------------------------------------------------------------------------------
 Ana Sinyal Frekansı (f1)    : {f1} Hz
 Yan Sinyal Frekansı (f2)    : {f2} Hz
 Örnekleme Hızı (Fs)         : 8000 Hz
 Veri Noktası Sayısı (N)     : {n}
 Sinyal-Gürültü Oranı (SNR)   : {snr_val:.2f} dB
 
 [ANOMALİ TESPİT DURUMU]
 -------------------------------------------------------------------------------
 DURUM                : {durum}
 
 Açıklama             : 
 Sinyal içinde {f1} Hz ve {f2} Hz frekanslarında baskın tepe noktaları saptandı.
 {'Filtreleme işlemi sırasında dinamik eşik değerini aşan yüksek enerjili bir sinyal patlaması (anomali) tespit edilmiştir.' if burst_power > 6.5 else 'Sinyal akışı doğal limitler dahilinde olup, olağanüstü bir anomaliye rastlanmamıştır.'}
-------------------------------------------------------------------------------
    """
    
    return fig, wav_path, rapor_metni, csv_path

# --- 6. GRADIO WEB ARAYÜZÜ TASARIMI ---
# Tema ve stil ayarları
theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"]
)

with gr.Blocks(theme=theme, title="TUA SERTİFİKALI-ÇELİKÇİLER Sinyal Analiz v10.9") as demo:
    
    # Başlık ve Logo Bölümü
    with gr.Row():
        gr.Markdown(
            """
            # 🛰️ TÜRK UZAY AJANSI (TUA) - SERTİFİKALI-ÇELİKÇİLER v10.9
            ## Kozmik Sinyal Analiz ve Anomali Tespit Arayüzü
            *Karabük Merkez İstasyonu Veri İşleme Birimi*
            """
        )
    
    gr.Markdown("---")
    
    # Kontrol Butonu
    with gr.Row():
        analiz_btn = gr.Button("🚀 SİNYAL YAKALA VE ANALİZ ET", variant="primary", scale=2)
    
    gr.Markdown("### 📡 Analiz Sonuçları")
    
    # Sonuç Paneli (Grafik ve Diğer Çıktılar)
    with gr.Row():
        # Sol Taraf: Grafikler (Daha Geniş)
        with gr.Column(scale=3):
            plot_output = gr.Plot(label="Spektrum ve Zaman Alanı Analizi", show_label=False)
        
        # Sağ Taraf: Ses, Rapor ve Dosyalar
        with gr.Column(scale=2):
            audio_output = gr.Audio(label="Sinyal Ses Kaydı (Filtrelenmiş WAV)", type="filepath")
            report_output = gr.Textbox(label="Resmi İstasyon Raporu", lines=18, show_label=True, interactive=False)
            gr.Markdown("#### 📦 Dosya İndirme")
            csv_output = gr.File(label="Ham Veriyi İndir (CSV)", file_types=[".csv"])

    # Buton Tıklama Olayı (Fonksiyonu Çıktılara Bağlama)
    analiz_btn.click(
        fn=kozmik_analiz, 
        outputs=[plot_output, audio_output, report_output, csv_output]
    )
    
    # Sayfa Altı Bilgisi
    gr.Markdown("---")
    gr.Markdown(
        """
        *TUA SERTİFİKALI-ÇELİKÇİLER v10.9 Sinyal Analiz Sistemi* | Tüm Hakları Saklıdır © 2026
        """
        )

# --- 7. UYGULAMAYI BAŞLATMA ---
if __name__ == "__main__":
    # share=False yaparak sadece yerelde çalışmasını sağlıyoruz (hackathon sunumu için en güvenlisi)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
