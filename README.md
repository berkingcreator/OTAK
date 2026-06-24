# OTAK (Öz Türkçe Bilge) Dil Modeli Ailesi

OTAK, Türkiye'nin yapay zeka alanındaki teknolojik bağımsızlık vizyonuna katkı sağlamak amacıyla sıfırdan Llama mimarisiyle kurgulanmış, açık kaynaklı yerli bir dil modeli ailesi projesidir. Projenin temel amacı, kısıtlı donanımlarda (4-bit kuantizasyon ile 3 GB VRAM kapasiteli kartlarda) dahi yüksek kararlılıkla çalışabilecek küçük ve etkili dil modelleri (SLM) üretmektir.

## 🚀 Otak Mini-1 (Konsept Kanıtı - PoC)

Büyük model mimarileri öncesinde altyapıyı test etmek amacıyla eğitilen 270 Milyon parametreli prototip sürümdür.

* **Mimari:** LlamaForCausalLM (Sıfırdan Scratch Eğitim)
* **Parametre Sayısı:** 269.89 Milyon
* **Token Sayısı:** 3.2 Milyar Token
* **Kapanış Loss Değeri:** 3.06
* **Altyapı:** Google Colab T4 GPU
* **Sözlük (Tokenizer):** `ytu-ce-cosmos/turkish-gpt2`

### 📊 Model Konfigürasyonu
```python
LlamaConfig(
    vocab_size=50257,
    hidden_size=896,               
    intermediate_size=3584,
    num_hidden_layers=14,
    num_attention_heads=14,
    max_position_embeddings=256,   
)
```

## 🛠️ Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak veya eğitimi başlatmak için gerekli bağımlılıkları kurun:

```bash
pip install transformers datasets accelerate torch
```

Eğitimi başlatmak için `train.py` dosyasını çalıştırmanız yeterlidir:

```bash
python train.py
```

## 📌 Yol Haritası (Roadmap)
- [x] Otak Mini-1 (270M) ön eğitim sürecinin tamamlanması ve bağlam optimizasyonu.
- [ ] Veri seti kalitesinin artırılması ve tamamen Türkçe kodlama talimatlarının entegre edilmesi.
- [ ] **OTAK 2.5 Flash (1B Parametre)** modelinin TRUBA süper bilgisayar altyapısı üzerinde eğitilmesi.
- [ ] GGUF kuantizasyon formatı ile düşük donanımlı sistemler için optimize edilmesi.

## 📄 Lisans
Bu proje **MIT Lisansı** altında açık kaynak olarak dağıtılmaktadır. Detaylar için `LICENSE` dosyasına göz atabilirsiniz.
