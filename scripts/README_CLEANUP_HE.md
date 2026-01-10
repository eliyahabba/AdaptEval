# מדריך ניקוי קבצים - Chain Linking

## 🎯 מה הבעיה?

אחרי הרצת ניסוי chain linking, התיקייה תופסת **~11GB** במקום ~100MB שצריך!

**הסיבה**: קבצים זמניים שלא נמחקו (`.temp/`, `chain_cache/`)

---

## ✅ הפתרון המהיר

```bash
# בדיקה מה יימחק (לא באמת מוחק):
./scripts/batch_cleanup.sh /cslab/gabis_lab/ehabba/AdaptEval/data/v3

# מחיקה אמיתית:
./scripts/batch_cleanup.sh /cslab/gabis_lab/ehabba/AdaptEval/data/v3 --force
```

**תוצאה**: 11GB → 100MB (חיסכון של 99%)

---

## 📂 מה יש בתיקיית ניסוי?

### ✅ קבצים חשובים (לשמור!)
```
all_results.csv          # תוצאות כל המרחקים
all_results.json
config.json              # הגדרות הניסוי
dist_X_*/
  ├── results.json       # מטריקות למרחק X
  ├── validation_*.csv   # תוצאות פר-מודל
  └── random_*.csv       # baselines
```
**גודל**: ~100MB
**למה צריך**: ניתוח, גרפים, טבלאות למאמר

---

### 🗑️ קבצים למחיקה

#### 1. `.temp/` (~7GB)
קבצים זמניים לתקשורת בין workers:
- DataFrames גדולים (`final_df_*.pkl`)
- פרמטרי IRT (`*_irt.pkl`, `*_A.npy`, `*_B.npy`)

**למחוק תמיד!** (הקוד אמור למחוק אוטומטית אבל לפעמים נכשל)

#### 2. `chain_cache/` (~2.7GB)
Cache לחידוש ניסוי שנכשל:
- תיקיות `after_*/` עם outputs של אימון
- `checkpoint.pkl`

**למחוק אחרי שהניסוי הצליח!**

#### 3. `irt_base/` (~119MB) + `dist_*/irt_*/` (~500MB)
מודלי IRT מאומנים.

**אופציונלי**: רק אם צריך לטעון מחדש מודלים.

---

## 🛠️ איך להשתמש?

### אופציה 1: ניקוי מלא (מומלץ!)

```bash
# תחילה - רק תראה מה יקרה:
./scripts/batch_cleanup.sh /path/to/data/v3

# אם נראה טוב - מחק באמת:
./scripts/batch_cleanup.sh /path/to/data/v3 --force
```

---

### אופציה 2: שמירת IRT params + מחיקת training datasets (מומלץ מאוד!)

**חדש!** מחק רק את ה-training datasets (`.jsonlines`) תוך שמירת הפרמטרים:

```bash
# מחיקת training datasets בלבד (חיסכון של ~88% מ-IRT directories):
./scripts/clean_training_datasets.sh /path/to/experiment --force

# או דרך Python:
python scripts/cleanup_chain_results.py /path/to/experiment --models --keep-params

# דחיסת metadata נוסף (חיסכון של ~70%):
python scripts/compress_irt_metadata.py /path/to/experiment --force --recursive
```

**מה נשמר**:
- ✅ `item_params.parquet` - פרמטרי המודל (נחוץ!)
- ✅ `item_params.meta.json.gz` - metadata דחוס
- 🗑️ `irt_dataset_final.jsonlines` - training data (לא נחוץ!)
- 🗑️ `irt_val_dataset_dim5.jsonlines` - validation data (לא נחוץ!)

**חיסכון**: 17MB → ~1.5MB per IRT directory (91%)

---

### אופציה 3: ניקוי ידני

```bash
cd /cslab/gabis_lab/ehabba/AdaptEval/data/v3/full_chain_classic_seed_26_anchors_100_target_TruthfulQA

# מחק temp:
rm -rf .temp/

# מחק cache:
rm -rf chain_cache/

# מחק רק training datasets (שמור params!):
find . -type f -path "*/irt_*/*.jsonlines" -delete

# מחק מודלי IRT לגמרי:
rm -rf irt_base/
find . -type d -name "irt_*" -exec rm -rf {} +
```

---

### אופציה 4: Python script (יותר אופציות)

```bash
# ניסוי בודד:
python scripts/cleanup_chain_results.py /path/to/experiment --all

# כל הניסויים בתיקייה:
python scripts/cleanup_chain_results.py /path/to/data --all --recursive

# רק temp ו-cache:
python scripts/cleanup_chain_results.py /path/to/experiment --temp --cache

# מחק models אבל שמור params:
python scripts/cleanup_chain_results.py /path/to/experiment --models --keep-params
```

---

## 🚀 הרצות חדשות - ניקוי אוטומטי

```bash
python src/experiments/chain_linking/chain_linking_parallel.py \
  --data-source-mode helm_classic \
  --cleanup-cache \      # מוחק chain_cache אוטומטית (ברירת מחדל: כן)
  --cleanup-models       # מוחק גם מודלי IRT (ברירת מחדל: לא)
```

**ברירות מחדל**:
- ✅ `.temp/` - **תמיד נמחק אוטומטית**
- ✅ `chain_cache/` - **נמחק אוטומטית** (אלא אם `--no-cleanup-cache`)
- ❌ `dist_*/irt_*/` - **לא נמחק** (אלא אם `--cleanup-models`)

---

## 📊 דוגמה מהמקרה שלך

```bash
# לפני:
eliyahabba@phoenix-gw-02> du -h --max-depth=1
7.1G    ./.temp
2.7G    ./chain_cache
119M    ./irt_base
267M    ./dist_0_direct
179M    ./dist_1_HellaSwag
195M    ./dist_2_HellaSwag_GSM8K
11G     .

# אחרי ניקוי:
eliyahabba@phoenix-gw-02> du -h --max-depth=1
267M    ./dist_0_direct        # רק results + CSV
179M    ./dist_1_HellaSwag
195M    ./dist_2_HellaSwag_GSM8K
641M    .                      # חיסכון של 10.4GB!
```

---

## ⚠️ חשוב לדעת

1. **תמיד הרץ dry-run לפני מחיקה**:
   ```bash
   ./scripts/batch_cleanup.sh /path --dry-run  # או בלי --force
   ```

2. **ודא שהניסוי הסתיים בהצלחה** לפני מחיקת cache

3. **שמור עותק גיבוי** עד שוידאת שהתוצאות נכונות

4. **לעולם אל תמחק**:
   - `all_results.*`
   - `config.json`
   - `dist_*/results.json`
   - `dist_*/validation_*.csv`

---

## 🔍 בדיקה מהירה

```bash
# הצג גודל כל תת-תיקייה:
du -h --max-depth=1

# מצא כל ה-.temp directories:
find . -name ".temp" -type d -exec du -sh {} \;

# מצא כל ה-cache directories:
find . -name "chain_cache" -type d -exec du -sh {} \;
```

---

## 📚 מסמכים נוספים

- [מדריך מלא באנגלית](README_CLEANUP.md)
- [תיעוד מפורט](../docs/chain_linking_storage.md)
- [דוגמאות שימוש](cleanup_examples.sh)

