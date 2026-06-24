# 🇹🇷 OTAK (Yerli ve Milli Yapay Zeka Dil Modeli Ailesi)

OTAK, Türkiye'nin yazılım ve yapay zeka dünyasındaki tam bağımsızlık vizyonunun bir parçası olarak sıfırdan kurgulanmış, tamamen açık kaynaklı milli bir dil modeli projesidir. Savunma sanayimizde yakaladığımız yerlilik ve bağımsızlık ruhunu yapay zeka ekosistemine taşımayı kendimize vizyon edindik.

Projenin temel amacı, dışa bağımlılığı bitirerek Türkiye'deki kısıtlı donanım imkanlarına sahip kullanıcıların bile (4-bit kuantizasyon teknolojisi ile 3 GB VRAM kapasiteli eski ekran kartlarında) kendi bilgisayarlarında güvenle çalıştırabileceği, yüksek performanslı Küçük Dil Modelleri (SLM) üretmektir.

---

## 🎯 Proje Hedefleri ve Vizyonumuz

*   **Tam Bağımsızlık:** Küresel teknoloji devlerinin modellerine olan bağımlılığı azaltmak ve Türkçe dil yapısına tamamen hakim yerli bir yapay zeka altyapısı kurmak.
*   **Düşük Donanım Optimizasyonu:** Milyarlarca liralık sunuculara ihtiyaç duymadan, okullardaki ve evlerdeki bilgisayarlarda çalışabilecek dinamik bir mimari geliştirmek.
*   **Açık Kaynak Kültürü:** Tüm eğitim kodlarını ve süreçlerini şeffaf bir şekilde paylaşarak Türkiye'deki genç geliştiricilere ve akademik çalışmalara zemin hazırlamak.

---

## 🚀 Otak Mini-1 (Konsept Kanıtı - PoC)

Büyük model mimarileri öncesinde altyapıyı ve Türkçe dil yeteneklerini test etmek amacıyla geliştirilen ilk prototip sürümdür. Google Colab T4 üzerinde yürütülen 10.000 adımlık ön eğitim (pre-training) sürecinde, modelin Loss değeri **3.06** seviyesine indirilerek kararlı bir dil yapısı elde edilmiştir.

*   **Model Mimarisi:** Sıfırdan Llama Mimarisi (`LlamaForCausalLM`)
*   **Parametre Boyutu:** 269.89 Milyon (270M)
*   **Eğitilen Veri Boyutu:** 3.2 Milyar Token
*   **Kullanılan Kelime Dağarcığı:** `ytu-ce-cosmos/turkish-gpt2`

### 💻 Teknik Model Yapılandırması
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

---

## 📈 Gelecek Yol Haritası (Roadmap)

- [x] **Aşama 1:** Otak Mini-1 (270M) ile Llama mimarisinin sıfırdan Türkçe ön eğitim başarısının kanıtlanması.
- [ ] **Aşama 2:** Veri setinin genişletilmesi ve İngilizce kodlama komut baskısının tamamen Türkçe talimatlarla temizlenmesi.
- [ ] **Aşama 3:** **OTAK 2.5 Flash (1B Parametre)** modelinin TRUBA süper bilgisayar altyapısı ve akademik mentorluk destekleriyle 20 Milyar token ile eğitilmesi.
- [ ] **Aşama 4:** Modelin GGUF formatına sıkıştırılarak 3 GB VRAM'li sistemlerde çalıştırılabilir hale getirilmesi.

---

## 📄 Lisans
Bu proje, yerli teknolojinin özgürce büyümesi ve geliştirilmesi adına **MIT Lisansı** ile korunmaktadır. Telif hakkı kuralları dahilinde herkes tarafından geliştirilmeye açıktır.
