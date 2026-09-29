# Jetson Orin Nano çalışma rehberi

Hedef: tek kullanıcının bir prompt için beklediği süreyi düşürmek; JSON ve semantic doğruluğu korumak. Yerel macOS sonuçları Jetson ölçümünün yerine geçmez.

## 1. Önce donanım/yazılım eşleşmesi

`python main.py diagnose` Python, platform, L4T, torch sürümü, CUDA erişimi, compute capability, derlenmiş CUDA mimarileri, GPU bellek ve salt-okunur `nvpmodel -q` bilgisi verir. Şunları Jetson üzerinde kaydedin:

```bash
python main.py diagnose
cat /etc/nv_tegra_release
python -m pip show torch transformers peft
```

PyTorch wheel'i kurulu JetPack/L4T, Python ABI ve CUDA sürümüne göre seçilmelidir. Bir başka makineden kopyalanmış venv kullanmayın. Genel Linux/aarch64 wheel'i otomatik olarak Jetson uyumlu sayılmaz. NVIDIA'nın [kurulum rehberi](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) ve [uyumluluk tablosu](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html) kullanılmalıdır; evrensel bir CUDA index URL'si önerilmiyor.

Önce uyumlu CUDA PyTorch'u kurun, sonra:

```bash
python -m pip install -r requirements/inference.txt -c requirements/constraints.txt
python main.py serve --env-file configs/jetson_orin_nano.env
```

Projede geliştirme Python'u 3.12'dir. Hedef JetPack wheel'inin Python sürümü farklıysa bu fark ayrıca doğrulanmalı; zorla 3.12 wheel'i kurulması önerilmez. `requirements/constraints.txt` torch'u bilerek pinlemez. Diğer paketler de hedef mimaride kurulum/test gerektirir.

Bu repodaki Transformers 5.17, PyTorch >= 2.5 ister. Eski bir Jetson wheel'iyle bu yığın doğrudan çalıştırılmamalıdır. Geliştirme requirements'ı torch >= 2.6 kullanır; bazı eski pickle ağırlıkları için Transformers'ın güvenlik kontrolü de en az 2.6 ister. Modelin safetensors ağırlıkları tercih edilmelidir.

`sm_87` uyarısı tek başına yavaşlığın sebebini veya bellek bant genişliği darboğazını kanıtlamaz. CUDA kernel/PTX uyumluluğu, sürümler, güç modu, sıcaklık ve bellek ölçülmeden böyle bir sonuç çıkarılmamalıdır.

## 2. Başlangıç ayarı

`configs/jetson_orin_nano.env`: CUDA + float16, merged LoRA, SDPA, eğitimle uyumlu chat prompt, 192 token, retry kapalı, bir ısınma isteği, tek worker. Bu hedef için **aday ayardır**, hedefte doğrulanmış optimum değildir. SDPA hata veya doğruluk farkı yaratırsa `ATTENTION_IMPLEMENTATION=auto` ile karşılaştırın.

Model süreç içinde bir kez yüklenir. Merge orijinal adapter dosyalarına yazmaz. FP16/SDPA ve merge davranışı gerçek sorgularla kontrol edilmelidir. HF'nin [LoRA merge açıklaması](https://huggingface.co/docs/peft/developer_guides/checkpoint) ve PyTorch'un [inference mode sözleşmesi](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html) kullanılan yöntemleri açıklar.

## 3. Aynı örneklerle karşılaştırma

```bash
python main.py optimize \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path model_adapter \
  --test_file storage/datasets/sql_json_dataset_v2_professional/test.jsonl \
  --device cuda --dtype float16 --limit 100 --sample_seed 42
```

Her senaryo ayrı süreçte, aynı rastgele örneklerde, batch=1 ve warmup sonrası ölçülür. Karşılaştırma JSON'u exact/semantic/şema düşüşünü kontrol eder. Tam test ve challenge için `benchmark` komutunda limit'i kaldırın:

```bash
python main.py benchmark \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path model_adapter \
  --test_file storage/datasets/sql_json_dataset_v2_professional/challenge_test.jsonl \
  --device cuda --dtype float16 --merge_adapter --attention sdpa \
  --batch_size 1 --warmup_runs 1 \
  --output_dir artifacts/benchmarks/jetson_challenge \
  --dump_predictions artifacts/benchmarks/jetson_challenge/predictions.jsonl
```

Test, challenge ve işverenin beklenen cevapları etiketlenmiş gerçek sorgularında aynı ayarı kontrol edin. Model yükleme, soğuk ilk istek, sıcak ortalama/medyan/p95, tokenlar, aşama süreleri, bellek ve status hatalarını birlikte inceleyin.

`--profile_tokens` ayrı bir profiler koşusudur; ilk token süresi streamer ile ölçülür ve ilave maliyet oluşturabilir. Batch>1 amortize throughput ölçümüdür; etkileşimli gecikme değildir.

12 yerel CPU örneğinde 128-token ayarı bir JSON'u kestiği için mevcut varsayılan 192'dir. Daha düşük token sınırını yalnızca tam doğruluk kontrolünden sonra seçin.

## 4. Güç, ısı ve daha ileri backend

Bu çalışma güç modu, clock, swap veya sistem servisi değiştirmedi. Benchmark sırasında mevcut güç modu, sıcaklık/termal throttling ve bellek durumu kaydedilmeli; `tegrastats` hedef üzerinde izlenebilir. Sistem ayarlarını değiştirmek ölçüm karşılaştırmasını da değiştirir; başlangıç koşullarını sabitleyin.

GGUF/llama.cpp ve TensorRT alternatifleri bu repoda uygulanmadı. Önce birleştirilmiş ağırlığı `main.py merge` ile ayrı dizine çıkarmak mümkündür. Alternatif backend için model/JetPack uyumluluğu doğrulanmalı, aynı prompt sözleşmesi ve JSON parsing korunmalı, quantization kaybı test/challenge/domain verisiyle ölçülmelidir. Hız artışı ölçülmeden garanti edilmez.
