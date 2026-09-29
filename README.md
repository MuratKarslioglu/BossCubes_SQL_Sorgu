# Sorgu Atölyesi

Türkçe doğal dil isteklerini doğrulanmış bir sorgu JSON'una çeviren yerel model uygulaması. Web arayüzünde prompt çalıştırma, beklenen JSON ile karşılaştırma, doğru/hatalı değerlendirme ve CSV/JSONL dışa aktarımı bulunur.

**Uygulama SQL üretmez ve veritabanında sorgu çalıştırmaz.** Model satış domain'ine göre eğitilmiştir; farklı veritabanı şemalarını otomatik öğrenmez.

## Klasör yapısı

```text
app/                  Uygulama, API, web arayüzü ve CLI komutları
training/             Eğitim, veri yükleme ve doğruluk metrikleri
scripts/data/         Veri doğrulama ve harici veri dönüştürme araçları
requirements/         core / web / inference / training / dev bağımlılıkları
configs/              Geliştirme ve Jetson ayar profilleri
docs/                 Teknik belgeler ve veri seti metadata'sı
tests/                Regresyon ve isteğe bağlı model entegrasyon testleri
model_adapter/        Çalışma için gerekli eğitilmiş LoRA adapter
storage/              Yerel model, veri seti ve değerlendirmeler (Git'e girmez)
  models/             İsteğe bağlı yerel model denemeleri
  datasets/           Özel JSONL verileri; OOD ayrı tutulur
  feedback/           Kalıcı değerlendirme veritabanı
artifacts/            Üretilen dosyalar (Git'e girmez)
  benchmarks/         Performans raporları ve tahminler
  releases/           Kaynak ve adapter teslim ZIP'leri
main.py               Tüm komutlar için ortak giriş noktası
```

`requirements.txt`, standart bir bilgisayarda uygulamayı çalıştırmak için gereken
PyTorch ve inference bağımlılıklarını kurar. Eğitim için `requirements/training.txt`,
geliştirme/test için `requirements/dev.txt` kullanılır.

## Kurulum

Python 3.12 kullanın. macOS, Windows veya standart Linux bilgisayarda:

```bash
git clone https://github.com/MuratKarslioglu/Cuba_Boss_SQL_Sorgu.git
cd Cuba_Boss_SQL_Sorgu
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c requirements/constraints.txt
python main.py serve
```

Ardından http://127.0.0.1:8000 adresini açın. `.env` oluşturmak zorunlu değildir;
repodaki geliştirme profili otomatik kullanılır.

Windows PowerShell karşılığı:

```powershell
git clone https://github.com/MuratKarslioglu/Cuba_Boss_SQL_Sorgu.git
cd Cuba_Boss_SQL_Sorgu
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c requirements/constraints.txt
.\.venv\Scripts\python.exe main.py serve
```

**Jetson Orin Nano:** genel PyPI PyTorch'u yerine JetPack/Python/CUDA sürümleriyle uyumlu NVIDIA wheel'i kurulmalıdır. Sonrasında inference bağımlılıklarını kurun. [Jetson kurulumu ve ölçüm rehberi](docs/jetson_optimization.md).

### Model dosyaları

Eğitilmiş LoRA adapter kod deposundaki `model_adapter/` klasöründe bulunur; ayrıca
dosya kopyalamak gerekmez. Beklenen dosyalar:

```text
model_adapter/
  adapter_config.json
  adapter_model.safetensors
```

Temel model `Qwen/Qwen2.5-0.5B-Instruct` ilk çalıştırmada Hugging Face'ten
otomatik indirilir; ilk açılışta internet bağlantısı ve yeterli disk alanı gerekir.
Sonraki açılışlarda yerel önbellek kullanılır. Başka model/adapter kullanacaksanız
`.env.example` dosyasını `.env` olarak kopyalayıp `BASE_MODEL` ve `ADAPTER_PATH`
değerlerini değiştirebilirsiniz. Eksik adapter durumunda uygulama sessizce başka
modele geçmez.

### Başlatma

```bash
python main.py serve
```

Bu bilgisayardaki hazır ortamla: `.venv-dev/bin/python main.py serve`.

Arayüz: http://127.0.0.1:8000 · API belgeleri: http://127.0.0.1:8000/docs

Model süreç başına bir kez yüklenir ve başlangıçta bir ısınma sorgusu çalışır. Tamamlanmış model önbelleğiyle `HF_HUB_OFFLINE=1` kullanılarak ağsız çalıştırılabilir.

## Kullanım

1. Türkçe isteği yazın ve “Sorguyu çalıştır” düğmesine basın.
2. Modelin işlem, hedef, filtre ve belirsizlik yorumunu kontrol edin.
3. İsterseniz beklenen JSON'u girerek alan bazlı karşılaştırma yapın.
4. Sonucu doğru/hatalı işaretleyin; hata türü ve açıklama ekleyin.
5. Değerlendirmeleri JSONL veya CSV olarak indirin.

Kayıtlar `storage/feedback/evaluation.sqlite3` dosyasında kalıcıdır. Yeniden puanlama aynı kaydı günceller. İnsan doğruluğu yalnızca puanlanan denemeler üzerinden hesaplanır. **Geçerli JSON/şema, doğru yorumlama anlamına gelmez.**

Servis varsayılan olarak localhost'a açıktır ve kimlik doğrulama içermez. İnternete doğrudan açmayın. Promptlar, çıktılar ve geri bildirimler cihazda saklanır; hassas veri kullanımı kurumun izinlerine bağlıdır.

## Yapılandırma ve API

Temel ayarlar `.env.example` içindedir. `configs/development.env` standart geliştirme, `configs/jetson_orin_nano.env` CUDA/FP16 aday ayarlarıdır:

`main.py serve` ve `infer`, proje kökündeki `.env` dosyasını okur; yoksa geliştirme
profilini kullanır. Ortam değişkenleri dosyadaki ayarları geçersiz kılar. Yapılandırmadaki
göreli model/adapter/kayıt yolları ve `serve --env-file` yolu proje köküne göre çözülür;
CLI'ye açıkça verilen veri/çıktı/model yolları ise komutu çalıştırdığınız klasöre göredir.
Bu nedenle `python /projenin/yolu/main.py serve` başka klasörden de çalışır.

```bash
python main.py serve --env-file configs/jetson_orin_nano.env
```

Varsayılan: merged LoRA, deterministik generation, KV cache, 192 output token, sıfır retry, 4000 karakter/1024 input token sınırı, tek worker. Eşzamanlı ikinci inference 429 döner.

| Endpoint | İşlev |
|---|---|
| `GET /health` | Model durumu ve değerlendirme özeti |
| `POST /extract` | `{"text":"...", "expected":null}` |
| `POST /feedback` | `request_id`, `correct`, isteğe bağlı `category/note/expected` |
| `GET /export?format=jsonl` | Denemeleri JSONL indirme |
| `GET /export?format=csv` | CSV indirme |

Yanıtta `success/data/error`, süre aşamaları, runtime, karşılaştırma ve kayıt kimliği bulunur. Geçersiz girdi 422, meşgul model 429, hazır olmayan model 503 döner.

## CLI ve performans ölçümü

```bash
python main.py diagnose
python main.py infer --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path model_adapter --interactive --no_retry

python main.py benchmark --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path model_adapter --test_file /veri/test.jsonl \
  --merge_adapter --batch_size 1 --warmup_runs 1 \
  --dump_predictions artifacts/benchmarks/predictions.jsonl

python main.py optimize --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path model_adapter --test_file /veri/test.jsonl \
  --device cpu --dtype float32 --limit 100 --sample_seed 42
```

Benchmark; exact/semantic eşleşme, şema, alan/status doğruluğu, sıcak ortalama/medyan/p95 süre, soğuk ilk istek, token ve bellek bilgilerini kaydeder. `optimize` aynı örneklerle dört ayarı ayrı süreçlerde karşılaştırır; hız uğruna exact/semantic/şema oranını düşüren adayları önermez.

**Batch>1 amortize üretim süresidir, tek kullanıcının bekleme süresi değildir.** `--profile_tokens` profiler ek maliyetiyle ayrı ilk-token ölçümü yapar; normal benchmark ile karıştırmayın. Jetson performansı hedef cihazda doğrulanmalıdır.

Yerel kontrol: rastgele 100 sentetik satış sorgusunda exact/semantic %100; satış şeması dışındaki 136 OOD sorguda %0. Bu, gerçek alan sözlüğü ve kabul sorguları olmadan genel veritabanı doğruluğu iddia edilemeyeceğini gösterir.

## Geliştirme ve test

```bash
python -m pip install -r requirements/dev.txt -c requirements/constraints.txt
pytest
ruff check .
ruff format --check .

# İsteğe bağlı: önbellekteki tiny model ile entegrasyon kontrolleri
RUN_MODEL_INTEGRATION_TESTS=1 HF_HUB_OFFLINE=1 pytest
```

Şema, parsing, doğruluk metrikleri, generation, API, eşzamanlılık ve kalıcı kayıt regresyon testleri repoda tutulur. Varsayılan testler model indirmez. GitHub Actions CPU test/lint workflow'u içerir.

Eğitim ve veri doğrulama araçları `training/` ve `scripts/data/` altındadır. Eğitim bağımlılıkları inference kurulumundan ayrıdır. Dataset'ler ayrıca temin edilir; challenge verisi eğitimde kullanılmaz. JSON sözleşmesi ve veri kuralları: [proje spesifikasyonu](docs/text_to_sql_json_project_spec.md). [Veri dosyaları ve metadata](docs/datasets/README.md).

```bash
python main.py validate \
  --train storage/datasets/sql_json_dataset_v2_professional/train.jsonl \
  --validation storage/datasets/sql_json_dataset_v2_professional/validation.jsonl \
  --test storage/datasets/sql_json_dataset_v2_professional/test.jsonl

python main.py import-external --help
python main.py import-kaggle --help
python main.py train --help
```

Komutlar ayrıca paket modülü olarak çalıştırılabilir: `python -m app.cli.infer --help`
ve `python -m scripts.data.validate --help`. Alt dosyaları doğrudan çalıştırmayın;
ortak giriş noktası veya modül komutlarını kullanın.

### Docker (CPU)

```bash
docker build --target api -t sorgu-atolyesi .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$PWD/storage/feedback:/workspace/storage/feedback" sorgu-atolyesi

docker build --target test -t sorgu-test .
docker run --rm sorgu-test
```

Bu imaj Jetson CUDA imajı değildir; Docker Desktop Apple MPS'e erişmez. Model önbelleği kalıcı tutulacaksa ayrıca Hugging Face cache volume'u bağlayın.

## GitHub teslimi

- Repo kodu, yapılandırma örneklerini, gerekli testleri ve çalıştırma adapter'ını içerir.
- Çalıştırma için gereken LoRA adapter repodaki `model_adapter/` klasöründedir.
  Dataset'ler, loglar, SQLite kayıtları, sanal ortamlar ve gerçek `.env` paylaşılmaz.
- Repo model ağırlığı içerdiği için erişimini şirket politikasına uygun tutun; açık
  kaynak lisansı atanmadı.
- Eski Git geçmişinde ham CSV/log commit'leri bulunur. Git takibinden çıkarmak geçmişi temizlemez. Geçmişi paylaşmak istemiyorsanız temiz kaynak ZIP'ini **yeni bir private repo** başlangıcı olarak kullanın; eski history bu pakete dahil değildir.
