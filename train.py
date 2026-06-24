!pip install -q transformers datasets accelerate

from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import load_dataset, concatenate_datasets
import torch

print("OTAK Mini-1 başlatılıyor...")

# ytu tokenizer
tok = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2")
tok.pad_token = tok.eos_token

ayarlar = LlamaConfig(
    vocab_size=tok.vocab_size,
    hidden_size=896,               
    intermediate_size=3584,
    num_hidden_layers=14,
    num_attention_heads=14,
    max_position_embeddings=256,   
)
model = LlamaForCausalLM(ayarlar)
print(f"OTAK Mini-1. Parametre Sayısı: {model.num_parameters() / 1e6:.2f} Milyon")

print("Veriler çekiliyor.")

wiki_veri = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train[:40000]")
wiki_veri = wiki_veri.remove_columns([c for c in wiki_veri.column_names if c != "text"])

def wiki_duzenle(satir):
    return {"text": f"{satir['text']}{tok.eos_token}"}

wiki_veri = wiki_veri.map(wiki_duzenle)

kodlar = load_dataset("sahil2801/CodeAlpaca-20k", split="train[:10000]")

def kod_duzenle(ornek):
    return {"text": f"Görev: {ornek['instruction']} \nKod:\n{ornek['output']}{tok.eos_token}"}

kodlar = kodlar.map(kod_duzenle, remove_columns=kodlar.column_names)

hepsi = concatenate_datasets([wiki_veri, kodlar]).shuffle(seed=42)

def tokenla(veri):
    return tok(veri["text"], truncation=True, max_length=256) 

print("Tokenize işlemi başlıyor.")
hazir_veri = hepsi.map(tokenla, batched=True, remove_columns=["text"], num_proc=2)
toplayici = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

class EkranaYaz(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"Adım: {state.global_step} - Loss: {logs['loss']:.4f}")

print("Otak Mini-1 eğitimi başlıyor hadi hayırlısı.")
egitim_ayarlari = TrainingArguments(
    output_dir="./otak_mini_1_cikti",  
    per_device_train_batch_size=8,   
    gradient_accumulation_steps=2,
    max_steps=10000,                 
    save_steps=2000,                 
    logging_steps=10,                
    learning_rate=5e-4,              
    fp16=torch.cuda.is_available(),  
    optim="adamw_torch",
    weight_decay=0.01,
    report_to="none",
    disable_tqdm=False               
)

trainer = Trainer(
    model=model,
    args=egitim_ayarlari,
    data_collator=toplayici,
    train_dataset=hazir_veri,
    callbacks=[EkranaYaz()]       
)

trainer.train()

trainer.save_model("./OTAK-Mini-1-270m-Final")
tok.save_pretrained("./OTAK-Mini-1-270m-Final")

print("Eğitim tamamlandı! OTAK Mini-1 (270m) diske kaydedildi.")
