# HELM Processing - Configurable Dataset Mapping

מערכת עיבוד HELM עם אפשרות לבחור בין שני מצבי פעולה:

## מצבי פעולה

### 1. מצב מתקדם (Advanced Mapping Mode)
- משתמש בקבצי JSON עם מיפוי מדויק של שאלות
- מאפשר חיפוש מדויק ומהיר של שאלות
- דרוש זמינות של קבצי mapping

### 2. מצב דיפולטי (Generic Fallback Mode) 
- פועל ללא קבצי JSON נוספים
- משתמש בפירוק ID מהמבנה הסטנדרטי של HELM
- גנרי וניתן לשיתוף

## איך להשתמש

### שימוש עם מיפוי מתקדם

```python
from pathlib import Path
from converter_utils.dataset_utils import create_instance_section

# הגדרת נתיב לקבצי המיפוי
mapping_dir = Path("/path/to/your/mapping/files")

# שימוש עם מיפוי מתקדם
result = create_instance_section(
    instance=instance_data,
    display_request={},
    dataset_name="mmlu.geography",
    map_dir=mapping_dir,
    use_mapping=True
)
```

### שימוש במצב דיפולטי (לשיתוף)

```python
from converter_utils.dataset_utils import create_instance_section

# שימוש במצב דיפולטי - ID parsing בלבד
result = create_instance_section(
    instance=instance_data,
    display_request={},
    dataset_name="mmlu.geography",
    use_mapping=False
)

# או פשוט אל תעביר שום הגדרה (auto-detect)
result = create_instance_section(
    instance=instance_data,
    display_request={},
    dataset_name="mmlu.geography"
)
```

### קונפיגורציה פשוטה ונקייה

הקונפיגורציה מועברת כארגומנטים לפונקציות - אין קבצי config נפרדים:
- `use_mapping`: האם להשתמש במיפוי מתקדם
- `map_dir`: נתיב לקבצי המיפוי (אם נדרש)

## דוגמת שימוש מלאה

```python
from converter_utils.dataset_utils import create_instance_section, normalize_dataset_name

# עיבוד instance
instance = {
    "id": "id123",
    "split": "test", 
    "input": {"text": "What is the capital of France?"},
    "references": [
        {"output": {"text": "Paris"}},
        {"output": {"text": "London"}},
        {"output": {"text": "Berlin"}}
    ]
}

# יצירת section (מצב דיפולטי)
result = create_instance_section(
    instance=instance,
    display_request={},
    dataset_name="mmlu.geography",
    use_mapping=False
)

print(result)
# Output: {
#     "raw_input": "What is the capital of France?",
#     "dataset_name": "mmlu.geography", 
#     "hf_split": "test",
#     "hf_index": 123
# }
```

## קבצים במערכת

- `converter_utils/dataset_utils.py` - כל הפונקציות העיקריות כולל validation
- `converter_utils/advanced_mapping.py` - לוגיקת מיפוי מתקדמת (נדרשת רק במצב מתקדם)
- `example_usage.py` - דוגמאות שימוש

## יתרונות

### מצב מתקדם
✅ חיפוש מדויק של שאלות  
✅ תמיכה במקרי קצה מורכבים  
✅ ביצועים מהירים עם cache  

### מצב דיפולטי  
✅ לא דורש קבצי JSON נוספים  
✅ פשוט לשיתוף עם אחרים  
✅ עובד out-of-the-box  
✅ טיפול חכם במקרי קצה  

## הערות חשובות

1. **מינימליסטי**: כל מה שנדרש ב-2 קבצים בלבד - אין קבצי config נפרדים
2. **ארגומנטים ברורים**: כל הקונפיגורציה מועברת כארגומנטים
3. **תאימות לאחור**: הקוד הקיים ימשיך לעבוד ללא שינויים  
4. **בטיחות**: מיפוי כושל עובר אוטומטית למצב דיפולטי

