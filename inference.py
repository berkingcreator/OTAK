from transformers import LlamaForCausalLM, AutoTokenizer
import torch

print("OTAK Mini-1 yükleniyor, lütfen bekleyin...")
model_yolu = "./OTAK-Mini-1-270m-Final"

tokenizer = AutoTokenizer.from_pretrained(model_yolu)
model = LlamaForCausalLM.from_pretrained(model_yolu).to("cuda")

print("\n" + "="*50)
print(" OTAK Mini-1 (270m) Test Arayüzü")
print(" Çıkmak için 'çıkış' yazın.")
print("="*50 + "\n")

while True:
    user_input = input("[Sen]: ")
    
    if user_input.lower() == "çıkış":
        print("\n[OTAK Mini-1]: Görüşmek üzere!")
        break
        
    if not user_input.strip():
        continue
        
    inputs = tokenizer(user_input, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    
    # Üretim ayarları
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=256,              
        do_sample=True,              
        top_k=50,                    
        top_p=0.90,                  
        temperature=0.75,            
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Çıktıyı çözme ve formatlama
    cevap_listesi = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    if isinstance(cevap_listesi, list) and len(cevap_listesi) > 0:
        cevap = cevap_listesi[0]
    else:
        cevap = str(cevap_listesi)
    
    if cevap.startswith(user_input):
        cevap = cevap[len(user_input):].strip()
        
    print(f"[OTAK Mini-1]: {cevap}")
    print("-" * 50)
