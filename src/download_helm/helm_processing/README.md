# HELM Processing - Flexible Dataset Support

מערכת עיבוד HELM גנרית שתומכת בכל דאטה סט חדש ללא שינוי קוד:

## מצבי פעולה

### 1. מצב מתקדם (Advanced Mapping Mode)
- משתמש בקבצי JSON עם מיפוי מדויק של שאלות
- מאפשר חיפוש מדויק ומהיר של שאלות
- תומך בכל דאטה סט עם קבצי mapping מתאימים

### 2. מצב דיפולטי (Generic Fallback Mode) 
- פועל ללא קבצי JSON נוספים
- משתמש בפירוק ID מהמבנה הסטנדרטי של HELM
- **תומך בכל דאטה סט חדש אוטומטית**

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

### קונפיגורציה גמישה

- **מיפויי שמות**: בקובץ JSON נפרד (`dataset_mappings.json`)
- **ארגומנטים ברורים**: כל הקונפיגורציה מועברת כארגומנטים  
- **אין global state**: אין משתנים גלובליים או config מורכב

## דוגמת שימוש מלאה

```python
from converter_utils.dataset_utils import create_instance_section, normalize_dataset_name

# עיבוד instance
instance = {
    "id": "id123",
    "split": "test", 
    "input": {"text": "Sample question text"},
    "references": [
        {"output": {"text": "Option A"}},
        {"output": {"text": "Option B"}},
        {"output": {"text": "Option C"}}
    ]
}

# השתמש בכל שם דאטה סט שאתה רוצה
dataset_name = "your_dataset_name.your_subject"  # החלף עם השם שלך

# יצירת section (מצב דיפולטי) - עובד עם כל דאטה סט
result = create_instance_section(
    instance=instance,
    display_request={},
    dataset_name=dataset_name,
    use_mapping=False
)

print(result)
# Output: {
#     "raw_input": "Sample question text",
#     "dataset_name": "your_dataset_name.your_subject", 
#     "hf_split": "test",
#     "hf_index": 123
# }
```

## קבצים במערכת

- `converter_utils/dataset_utils.py` - כל הפונקציות העיקריות כולל validation
- `converter_utils/advanced_mapping.py` - לוגיקת מיפוי מתקדמת (נדרשת רק במצב מתקדם)
- `dataset_mappings.json` - מיפויי שמות דאטה סטים (ניתן לעריכה)
- `example_usage.py` - דוגמאות שימוש

## יתרונות החדשים

### גמישות מלאה
✅ **תומך בכל דאטה סט חדש** ללא שינוי קוד  
✅ **ללא שמות קודקדים** - עובד עם כל שם דאטה סט  
✅ **מיפויים מותאמים אישית** - רק כשנדרש  
✅ **חילוץ אוטומטי מ-HELM** - מבין את מבנה HELM ללא הגבלות  

### מצב מתקדם
✅ חיפוש מדויק של שאלות  
✅ תמיכה במקרי קצה מורכבים  
✅ ביצועים מהירים עם cache  

### מצב דיפולטי  
✅ לא דורש קבצי JSON נוספים  
✅ פשוט לשיתוף עם אחרים  
✅ עובד out-of-the-box עם כל דאטה סט  
✅ טיפול חכם במקרי קצה  

## דוגמאות להוספת דאטה סטים חדשים

```python
# דוגמה 1: כל דאטה סט חדש עובד מיד
dataset_name = "company_evaluation.financial_analysis"
result = create_instance_section(instance, {}, dataset_name)
# עובד מיד! ✅

# דוגמה 2: מיפויים אוטומטיים מקובץ JSON
normalized = normalize_dataset_name("gsm.math")  # → "gsm8k.math" (מ-dataset_mappings.json)

# דוגמה 3: מיפוי מותאם (עוקף את ה-JSON)
custom_mappings = {"legacy_name": "modern_name"} 
normalized = normalize_dataset_name("legacy_name.category", custom_mappings)
# → "modern_name.category"

# דוגמה 4: HELM חדש יחלץ כל שם אוטומטית
run_spec = {
    "scenario_spec": {
        "class_name": "custom_research_scenario",
        "args": {"domain": "healthcare"}
    }
}
extracted = extract_dataset_name_from_run_spec(run_spec, {})
# → "custom_research.healthcare" ✅
```

## הערות חשובות

1. **גנרי לחלוטין**: אין שמות דאטה סטים קודקדים בקוד
2. **מיפויים ב-JSON**: קל לעריכה ללא שינוי קוד Python
3. **ארגומנטים ברורים**: כל הקונפיגורציה מועברת כארגומנטים
4. **תאימות לאחור**: הקוד הקיים ימשיך לעבוד ללא שינויים  
5. **הוספה קלה**: דאטה סטים חדשים עובדים מיד ללא שינוי קוד
6. **עריכה נוחה**: פשוט לערוך את `dataset_mappings.json` להוסיף מיפויים

