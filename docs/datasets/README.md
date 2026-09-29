# Veri setleri

JSONL ve ham veri dosyaları kod deposunda paylaşılmaz. Yerel dosyalar:

```text
storage/datasets/
  sql_json_dataset_v2_professional/
    train.jsonl
    validation.jsonl
    test.jsonl
    challenge_test.jsonl
  ood_eval/
    kaggle_turkish_real_nl_ood.jsonl
```

`v2/` altındaki dataset card, manifest ve JSON şeması ilk üretimin metadata'sıdır.
Sonraki veri temizlemeleri nedeniyle manifest'teki sayılar mevcut dosyalardan farklı
olabilir; güncel sayı ve bütünlük için `python main.py validate --json ...` kullanın.
Challenge ve OOD dosyaları eğitime katılmaz.

Eğitim verileri ve kaynakların kullanım/paylaşım izinleri ayrıca doğrulanmalıdır.
GitHub teslimi veri içermediğinden, eğitim ve benchmark için JSONL dosyaları ayrıca
temin edilmelidir. Normal inference için bu dosyalar gerekmez.
