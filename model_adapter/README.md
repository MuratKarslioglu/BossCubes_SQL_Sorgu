# Eğitilmiş LoRA adapter

Bu klasör, uygulamanın varsayılan olarak kullandığı eğitilmiş adapter'ı içerir.
`adapter_config.json` ve `adapter_model.safetensors` uygulamanın çalışması için
gereklidir; klasör yapısını değiştirmeyin.

Temel model (`Qwen/Qwen2.5-0.5B-Instruct`) ilk çalıştırmada Hugging Face'ten
otomatik indirilir ve kullanıcının yerel model önbelleğinde tutulur.
