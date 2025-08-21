import json
import os
import re
import pandas as pd
import argparse
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# יצירת מבנה נתונים לאחסון כל הדוגמאות
all_examples = []

# מיפוי בין שמות דאטה-סטים לנתיבי מאגרים ב-Hugging Face
DATASET_TO_HF_REPO = {
    "med_qa": "bigbio/med_qa",
    "openbook_qa": "allenai/openbookqa",
    "mmlu": "cais/mmlu",
    # ניתן להוסיף עוד מיפויים כאן
}

# מיפוי בין שמות דאטה-סטים לתבניות הנחיות
DATASET_TO_INSTRUCTION = {
    "med_qa": {
        "name": "MultipleChoiceTemplatesInstructionsWithTopicHelm",
        "text": "The following are multiple choice questions (with answers) about medicine.\n\nQuestion: {question}\n{choices}\nAnswer:"
    },
    "openbook_qa": {
        "name": "MultipleChoiceTemplatesInstructionsWithTopicHelm",
        "text": "The following are multiple choice questions (with answers) about common sense.\n\nQuestion: {question}\n{choices}\nAnswer:"
    },
    "mmlu": {
        "name": "MultipleChoiceTemplatesInstructionsWithTopicHelm",
        "text": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {question}\n{choices}\nAnswer:"
    },
    # ניתן להוסיף עוד מיפויים כאן
}

# מטמון למיפויי אינדקסים של דאטה-סטים
INDEX_MAP_CACHE = {}

# פונקציה לקבלת אינדקס ומחיצה של שאלה מתוך קובץ מיפוי
def get_question_index(dataset_name: str, question: str, choices: List[str]) -> Tuple[int, str]:
    """
    מחפשת את האינדקס והמחיצה של שאלה בדאטה-סט המקורי
    
    Args:
        dataset_name: שם הדאטה-סט
        question: טקסט השאלה
        choices: רשימת אפשרויות התשובה
        
    Returns:
        tuple: (hf_index, hf_split)
    """
    try:
        # נרמול שם הדאטה-סט
        map_file_name = dataset_name.split('.')[0] if '.' in dataset_name else dataset_name
        
        # טיפול במקרים מיוחדים
        if map_file_name == "ai2_arc":
            map_file_name = "ai2_arc"
        elif map_file_name.startswith("global_mmlu"):
            # הוספת השפה לשם הקובץ
            map_file_name = f"{map_file_name}.{dataset_name.split('.')[1]}"

        # בניית נתיב לקובץ המיפוי
        current_dir = Path(__file__).parent
        json_path = Path(r'/Users/ehabba/PycharmProjects/LLM-Evaluation/src/experiments/dataset_scheme/conversions/hf_map_data') / f"{map_file_name}_samples.json"
        
        # בדיקה אם הקובץ קיים
        if not json_path.exists():
            print(f"Warning: Map file {json_path} not found. Using default values.")
            return None, "test"
            
        # טעינת קובץ המיפוי אם לא נטען כבר
        if map_file_name not in INDEX_MAP_CACHE:
            with open(json_path, 'r', encoding='utf-8') as f:
                INDEX_MAP_CACHE[map_file_name] = json.load(f)
                
        index_map = INDEX_MAP_CACHE[map_file_name]
        
        # ניקוי ונרמול הבחירות
        clean_choices = [choice.strip() for choice in choices]
        
        # יצירת מפתח חיפוש בפורמט זהה לזה שבקובץ המיפוי
        key = f"{question}|||{'|||'.join(sorted(clean_choices))}"
        
        # חיפוש ישיר במיפוי
        if key in index_map:
            metadata = index_map[key]
            print(f"Match found - index: {metadata['index']}, source: {metadata['source']}")
            return metadata['index'], metadata['source']
            
        # אם לא נמצאה התאמה, רישום פרטים לצורך דיבוג
        print(f"No match found for key in {map_file_name}_samples.json")
        print(f"Question: {question}")
        print(f"Choices: {clean_choices}")
        
        return None, "test"
        
    except Exception as e:
        print(f"Error in get_question_index: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None, "test"

# פונקציה לקבלת נתיב המאגר ב-Hugging Face לפי שם הדאטה-סט
def get_hf_repo_for_dataset(dataset_name: str) -> Optional[str]:
    """מחזירה את נתיב המאגר ב-Hugging Face לפי שם הדאטה-סט, או None אם אין מיפוי"""
    if not dataset_name:
        return None
        
    # בדיקה אם יש התאמה מדויקת
    if dataset_name in DATASET_TO_HF_REPO:
        return DATASET_TO_HF_REPO[dataset_name]
    
    # בדיקה אם יש התאמה לחלק הראשון של השם (לפני הנקודה)
    base_name = dataset_name.split('.')[0] if '.' in dataset_name else dataset_name
    if base_name in DATASET_TO_HF_REPO:
        return DATASET_TO_HF_REPO[base_name]
    
    return None

# פונקציה לקבלת תבנית ההנחיות לפי שם הדאטה-סט
def get_instruction_for_dataset(dataset_name: str) -> Optional[Dict[str, str]]:
    """מחזירה את תבנית ההנחיות לפי שם הדאטה-סט, או None אם אין מיפוי"""
    if not dataset_name:
        return None
        
    # בדיקה אם יש התאמה מדויקת
    if dataset_name in DATASET_TO_INSTRUCTION:
        return DATASET_TO_INSTRUCTION[dataset_name]
    
    # בדיקה אם יש התאמה לחלק הראשון של השם (לפני הנקודה)
    base_name = dataset_name.split('.')[0] if '.' in dataset_name else dataset_name
    if base_name in DATASET_TO_INSTRUCTION:
        instruction = DATASET_TO_INSTRUCTION[base_name].copy()
        
        # אם זה MMLU ויש נושא ספציפי, נחליף את התבנית {topic} בנושא האמיתי
        if base_name == "mmlu" and '.' in dataset_name:
            topic = dataset_name.split('.')[1]
            instruction["text"] = instruction["text"].replace("{topic}", topic)
            
        return instruction
    
    return None

# קריאת מידע על מודלים מקובץ CSV
def load_model_metadata():
    """טעינת מידע על מודלים מקובץ CSV"""
    csv_path = os.path.join(os.path.dirname(__file__), "model_metadata.csv")
    try:
        df = pd.read_csv(csv_path)
        # המרת הדאטה-פריים למילון
        model_info = {}
        for _, row in df.iterrows():
            model_name = row['name']
            model_info[model_name] = {
                "family": row['family'],
                "architecture": row['architecture'],
                "parameters": row['parameters'],
                "context_window": row['context_window'],
                "is_instruct": row['is_instruct'] == 'true',
                "hf_path": row['hf_path'] if pd.notna(row['hf_path']) else None,
                "revision": row['revision'] if pd.notna(row['revision']) else None,
                "quantization": {
                    "bit_precision": row['quantization_bit_precision'],
                    "method": row['quantization_method']
                }
            }
        print(f"Loaded metadata for {len(model_info)} models from {csv_path}")
        return model_info
    except Exception as e:
        print(f"Error loading model metadata from CSV: {e}")
        # Return an empty dictionary as fallback
        return {}

# טעינת מידע על מודלים פעם אחת בתחילת הריצה
MODEL_INFO = load_model_metadata()


def load_json_file(filepath: str) -> dict:
    """טעינת קובץ JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_evaluation_id(run_spec: dict, instance_id: str) -> str:
    """יצירת מזהה ייחודי להערכה"""
    # שילוב של שם הריצה ומזהה הדוגמה
    run_name = run_spec.get("name")
    if not run_name:
        run_name = "unknown"
    evaluation_id = f"{run_name}-{instance_id}"
    import hashlib
    hashed_id = hashlib.sha256(evaluation_id.encode()).hexdigest()
    return hashed_id


def get_model_info(model_name: str) -> dict:
    """השגת מידע על המודל"""
    # בדיקה אם המודל מוכר במיפוי הידוע מראש
    if model_name in MODEL_INFO:
        model_data = MODEL_INFO[model_name]
        return {
            "model_info": {
                "name": model_name,
                "family": model_data["family"]
            },
            "configuration": {
                "context_window": model_data["context_window"],
                "architecture": model_data["architecture"],
                "parameters": model_data["parameters"],
                "is_instruct": model_data["is_instruct"],
                "hf_path": model_data["hf_path"],
                "revision": model_data["revision"]
            },
            "inference_settings": {
                "quantization": model_data["quantization"]
            }
        }
    else:
        # אם המודל לא מוכר, נחזיר רק את שם המודל ללא מידע נוסף
        print(f"Warning: Model '{model_name}' not found in metadata. Using minimal default values.")
        return {
            "model_info": {
                "name": model_name,
                "family": None
            },
            "configuration": {
                "context_window": None,
                "architecture": None,
                "parameters": None,
                "is_instruct": None,
                "hf_path": None,
                "revision": None
            },
            "inference_settings": {
                "quantization": {
                    "bit_precision": "none",
                    "method": "None"
                }
            }
        }


def create_model_section(run_spec: dict, display_request: dict) -> dict:
    """יצירת חלק המודל בסכמה החדשה"""
    model_name = run_spec.get("adapter_spec", {}).get("model")
    model_info = get_model_info(model_name)

    # הוספת פרמטרי היצירה מתוך נתוני ה-display_request
    # כאן אנחנו לוקחים את הערכים אך ורק מהנתונים עצמם, ללא ערכי ברירת מחדל
    model_info["inference_settings"]["generation_args"] = {
        "use_vllm": None,  # מידע שלא קיים ב-HELM
        "temperature": display_request.get("request", {}).get("temperature"),
        "top_p": display_request.get("request", {}).get("top_p"),
        "max_tokens": display_request.get("request", {}).get("max_tokens"),
        "stop_sequences": display_request.get("request", {}).get("stop_sequences")
    }

    return model_info['model_info']


def detect_prompt_format(prompt: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    מזהה את פורמט הפרומפט מתוך הטקסט - אילו אותיות משמשות למספור ואיזה תו משמש להפרדה
    מחזיר: (enumerator_type, separator, instruction_name)
    """
    if not prompt:
        return None, None, None

    # בדיקה יותר מדויקת של הפורמט של האופציות
    # מחפשת A., B., C., D. ברצף בתוך הטקסט

    # בדיקת אותיות גדולות (A., B., C., D.)
    if re.search(r'A\. .+\s+B\. .+\s+C\. .+\s+D\.', prompt, re.DOTALL):
        enumerator = "capitals"
    # בדיקת אותיות קטנות (a., b., c., d.)
    elif re.search(r'a\. .+\s+b\. .+\s+c\. .+\s+d\.', prompt, re.DOTALL):
        enumerator = "lowercase"
    # בדיקת מספרים (1., 2., 3., 4.)
    elif re.search(r'1\. .+\s+2\. .+\s+3\. .+\s+4\.', prompt, re.DOTALL):
        enumerator = "numbers"
    # בדיקת ספרות רומיות (i., ii., iii., iv.)
    elif re.search(r'i\. .+\s+ii\. .+\s+iii\. .+\s+iv\.', prompt, re.DOTALL) or \
            re.search(r'I\. .+\s+II\. .+\s+III\. .+\s+IV\.', prompt, re.DOTALL):
        enumerator = "roman"
    # בדיקת מקשי מקלדת (@, #, $, %)
    elif re.search(r'@\. .+\s+#\. .+\s+\$\. .+\s+%\.', prompt, re.DOTALL):
        enumerator = "keyboard"
    else:
        enumerator = None

    # בדיקת המפריד בין אפשרויות
    # מחפשת את התבנית שבה האופציות מופרדות

    # בדיקה האם יש שורות חדשות בין האופציות
    if re.search(r'A\. .+\nB\. ', prompt) or re.search(r'1\. .+\n2\. ', prompt):
        separator = "\\n"
    # בדיקה האם יש פסיקים בין האופציות
    elif re.search(r'A\. .+, B\. ', prompt) or re.search(r'1\. .+, 2\. ', prompt):
        separator = ", "
    # בדיקה האם יש נקודה-פסיק בין האופציות
    elif re.search(r'A\. .+; B\. ', prompt) or re.search(r'1\. .+; 2\. ', prompt):
        separator = "; "
    # בדיקה האם יש מקף אנכי בין האופציות
    elif re.search(r'A\. .+\| B\. ', prompt) or re.search(r'1\. .+\| 2\. ', prompt):
        separator = " | "
    # בדיקה האם יש "OR" בין האופציות
    elif re.search(r'A\. .+ OR B\. ', prompt) or re.search(r'1\. .+ OR 2\. ', prompt):
        separator = " OR "
    # בדיקה האם יש "or" בין האופציות
    elif re.search(r'A\. .+ or B\. ', prompt) or re.search(r'1\. .+ or 2\. ', prompt):
        separator = " or "
    else:
        separator = None

    # זיהוי סוג ההנחיות (אם יש)
    instruction_name = None


    return enumerator, separator, instruction_name


def extract_dataset_name(run_spec: dict, scenario: dict) -> Optional[str]:
    """חילוץ שם הדאטה-סט מתוך ה-run_spec וה-scenario"""
    dataset_base = None
    subject = None

    # נסיון לחלץ את ה-dataset והנושא מה-run_spec
    if run_spec and "scenario_spec" in run_spec:
        spec = run_spec.get("scenario_spec", {})

        # בדיקה אם יש מחלקה שמכילה את שם הדאטה-סט
        class_name = spec.get("class_name", "")
        if class_name:
            # בדיקה למקרה של OpenBookQA בתוך commonsense_scenario
            if "commonsense_scenario.OpenBookQA" in class_name:
                dataset_base = "openbook_qa"
            else:
                # מחלץ את שם הדאטה-סט מתוך שם המחלקה
                for part in class_name.split('.'):
                    part_lower = part.lower()
                    if "_scenario" in part_lower:
                        dataset_base = part_lower.replace("_scenario", "")
                        break

        # חילוץ הנושא מתוך הארגומנטים
        if "args" in spec and "subject" in spec["args"] and spec["args"]["subject"]:
            subject = spec["args"]["subject"]

    # אם לא מצאנו בסיס דאטה-סט מה-class_name, נבדוק אם יש פרמטר dataset בשם ה-run
    if not dataset_base and run_spec and "name" in run_spec:
        run_name = run_spec["name"]

        # בדיקה ספציפית לפרמטר dataset= בשם ה-run
        if "dataset=" in run_name:
            dataset_parts = [p for p in run_name.split(",") if "dataset=" in p]
            if dataset_parts:
                dataset_value = dataset_parts[0].split("=")[1]
                if dataset_value.lower() == "openbookqa":
                    dataset_base = "openbook_qa"
                else:
                    dataset_base = dataset_value
        # אם אין dataset=, נסה לקחת את החלק לפני ה:
        elif ":" in run_name and not dataset_base:
            dataset_base = run_name.split(":")[0]

        # בודק אם יש subject בשם ה-run
        if not subject and "subject=" in run_name:
            subject_part = [p for p in run_name.split(",") if "subject=" in p]
            if subject_part:
                subject = subject_part[0].split("=")[1]

    # אם עדיין אין לנו בסיס דאטה-סט, ננסה לקחת משם ה-scenario
    if not dataset_base and scenario and "name" in scenario:
        scenario_name = scenario["name"]
        if scenario_name.lower() == "openbookqa":
            dataset_base = "openbook_qa"
        else:
            dataset_base = scenario_name

    # בניית שם הדאטה-סט המלא
    if dataset_base:
        if subject:
            dataset_name = f"{dataset_base}.{subject}"
        else:
            dataset_name = dataset_base

        return dataset_name

    return None  # אם לא הצלחנו לחלץ מידע מספק

def extract_dataset_name1(run_spec: dict, scenario: dict) -> Optional[str]:
    """חילוץ שם הדאטה-סט מתוך ה-run_spec וה-scenario"""
    dataset_base = None
    subject = None

    # נסיון לחלץ את ה-dataset והנושא מה-run_spec
    if run_spec and "scenario_spec" in run_spec:
        spec = run_spec.get("scenario_spec", {})
        # בדיקה אם יש מחלקה שמכילה את שם הדאטה-סט
        class_name = spec.get("class_name", "")
        if class_name:
            # מחלץ את שם הדאטה-סט מתוך שם המחלקה (למשל, mmlu מתוך MMLUScenario)
            for part in class_name.split('.'):
                if "_scenario" in part.lower():
                    dataset_base = part.lower().replace("_scenario", "")
                    break

        # חילוץ הנושא מתוך הארגומנטים
        if "args" in spec and "subject" in spec["args"]:
            subject = spec["args"]["subject"]

    # נסיון לחלץ מהשם של ה-run אם לא מצאנו ב-scenario_spec
    if not dataset_base and run_spec and "name" in run_spec:
        name_parts = run_spec["name"].split(":")
        if len(name_parts) > 0:
            first_part = name_parts[0].split(",")[0]  # למקרה שיש פרמטרים אחרי שם הדאטה-סט
            dataset_base = first_part

    # אם אין לנו את אחד מהרכיבים או אם הם ריקים, ננסה לקחת מידע משם ה-scenario
    if (not dataset_base or not subject) and scenario and "name" in scenario:
        scenario_name_parts = scenario["name"].split(".")
        if len(scenario_name_parts) > 1:
            # אם שם ה-scenario כבר מכיל נקודה, ננסה לפרק אותו
            dataset_base = dataset_base or scenario_name_parts[0]
            # אם אין לנו נושא, ננסה לקחת את החלק השני
            subject = subject or scenario_name_parts[1]
        else:
            # אם אין נקודה, נשתמש בשם המלא כבסיס אם אין לנו
            dataset_base = dataset_base or scenario["name"]

    # בניית שם הדאטה-סט המלא
    if dataset_base:
        if subject:
            dataset_name = f"{dataset_base}.{subject}"
        else:
            dataset_name = dataset_base
            
        # נרמול שם הדאטה-סט
        return normalize_dataset_name(dataset_name)

    return None  # אם לא הצלחנו לחלץ מידע מספק

def create_prompt_config(run_spec: dict, first_request: dict, dataset_name: str = None) -> dict:
    """יצירת חלק הגדרות הפרומפט בסכמה החדשה"""
    adapter_spec = run_spec.get("adapter_spec", {})

    # זיהוי הפורמט מתוך הפרומפט של הבקשה הראשונה
    prompt = first_request.get("request", {}).get("prompt") if first_request else None
    enumerator, separator, _ = detect_prompt_format(prompt)
    
    # קבלת תבנית ההנחיות לפי שם הדאטה-סט
    instruction_phrasing = None
    if dataset_name:
        instruction_phrasing = get_instruction_for_dataset(dataset_name)
    
    # אם לא מצאנו תבנית ספציפית, נשתמש בהנחיות מה-adapter_spec
    if not instruction_phrasing:
        instruction_phrasing = {
            "name": None,
            "text": adapter_spec.get("instructions")
        }

    return {
        "prompt_class": "MultipleChoice",  # לפי הסכמה שלך, זה חייב להיות אחד מהערכים המוגדרים
        "dimensions": {
            "choices_order": {
                "method": "fixed",
                "description": "Fixed order as provided in dataset"
            },
            "enumerator": enumerator,  # כפי שזוהה מהנתונים
            "instruction_phrasing": instruction_phrasing,
            "separator": separator,  # כפי שזוהה מהנתונים
            "shots": adapter_spec.get("max_train_instances")
        }
    }


def extract_correct_answer(references: List[dict]) -> Optional[str]:
    """מציאת התשובה הנכונה מתוך רשימת האפשרויות"""
    for ref in references:
        if "correct" in ref.get("tags", []):
            return ref.get("output", {}).get("text")
    return None


def create_instance_section(instance: dict, display_request: dict, dataset_name: str) -> dict:
    """יצירת חלק הדוגמה בסכמה החדשה"""
    question_text = instance.get("input", {}).get("text")
    instance_id = instance.get("id")

    # יצירת המיפוי בין האותיות לטקסט התשובות
    references = instance.get("references", [])
    choices = []
    correct_text = extract_correct_answer(references)
    correct_id = None

    # רשימת טקסטים של כל האפשרויות לצורך חיפוש במיפוי
    choice_texts = []

    for i, ref in enumerate(references):
        choice_id = chr(65 + i)  # A, B, C, D...
        choice_text = ref.get("output", {}).get("text")
        choices.append({"id": choice_id, "text": choice_text})
        choice_texts.append(choice_text)
        if choice_text == correct_text:
            correct_id = choice_id

    # חילוץ מספר מזהה מ-ID
    instance_num = int(instance_id.replace("id", "")) if instance_id and instance_id.startswith("id") else None

    # קבלת נתיב המאגר ב-Hugging Face לפי שם הדאטה-סט
    hf_repo = get_hf_repo_for_dataset(dataset_name)
    
    # חיפוש האינדקס והמחיצה של השאלה בדאטה-סט המקורי
    hf_index, hf_split = get_question_index(dataset_name, question_text, choice_texts)

    # אם נמצא אינדקס, נשתמש בו במקום באינדקס שחילצנו מה-ID
    if hf_index is not None:
        instance_num = hf_index
    else:
        a=1
    result = {
        # "task_type": "classification",  # הכרחי לפי הסכמה
        "raw_input": question_text,
        # "language": "en",  # אין מידע ישיר בקבצים
        # "sample_identifier": {
        "dataset_name": dataset_name,
        # "hf_repo": hf_repo,  # שימוש בנתיב המאגר שמצאנו
        "hf_split": hf_split,  # שימוש במחיצה שמצאנו
        "hf_index": instance_num
        # },
        # "classification_fields": {
        #     "full_input": display_request.get("request", {}).get("prompt"),
        #     "question": question_text,
        #     "choices": choices,
        #     "ground_truth": {
        #         "id": correct_id,
        #         "text": correct_text
        #     }
        # }
    }

    return result


def create_output_section(prediction: dict) -> dict:
    """יצירת חלק הפלט בסכמה החדשה"""
    predicted_text = prediction.get("predicted_text")
    if predicted_text:
        predicted_text = predicted_text.strip()
    return predicted_text


def create_evaluation_section(prediction: dict, instance: dict) -> dict:
    """יצירת חלק ההערכה בסכמה החדשה"""
    # מציאת התשובה הנכונה
    references = instance.get("references", [])
    correct_id = None

    # איתור האות (A,B,C,D) של התשובה הנכונה
    for i, ref in enumerate(references):
        if "correct" in ref.get("tags", []):
            correct_id = chr(65 + i)  # A, B, C, D...
            break

    # Determine score using supported metrics in priority order
    stats = prediction.get("stats", {}) or {}
    metric_priority = [
        ("exact_match", {
            "method_name": "label_only_match",
            "description": "Compares only the choice identifier/label to evaluate the response."
        }),
        ("quasi_exact_match", {
            "method_name": "quasi_label_only_match",
            "description": "Compares only the choice identifier/label to evaluate the response with a tolerance for minor differences."
        }),
        ("final_number_exact_match", {
            "method_name": "final_number_match",
            "description": "Compares the final number of the response with the expected answer."
        }),
    ]

    selected_key = None
    score = None
    evaluation_method = None

    for key, method in metric_priority:
        value = stats.get(key)
        if value is not None:
            selected_key = key
            score = value
            evaluation_method = method
            break

    if selected_key is None:
        raise ValueError(
            "No supported evaluation metric found in prediction stats. "
            "Expected one of: exact_match, quasi_exact_match, final_number_exact_match."
        )

    return {
        "ground_truth": correct_id,
        "method_name": evaluation_method["method_name"],
        "score": score,
    }


def process_helm_data(data_dir: str) -> List[dict]:
    """עיבוד נתוני HELM ויצירת דוגמאות בפורמט הסכמה החדשה"""
    # טעינת הקבצים העיקריים
    run_spec = load_json_file(os.path.join(data_dir, "run_spec.json"))
    scenario = load_json_file(os.path.join(data_dir, "scenario.json"))
    instances = load_json_file(os.path.join(data_dir, "instances.json"))
    display_requests = load_json_file(os.path.join(data_dir, "display_requests.json"))
    display_predictions = load_json_file(os.path.join(data_dir, "display_predictions.json"))

    # יצירת מיפוי בין מזהי דוגמאות לבקשות ותחזיות
    request_map = {req["instance_id"]: req for req in display_requests}
    prediction_map = {pred["instance_id"]: pred for pred in display_predictions}

    examples = []

    # חילוץ שם הדאטה-סט פעם אחת לפני הלולאה
    dataset_name = extract_dataset_name(run_spec, scenario)
    print(f"Dataset name extracted: {dataset_name}")

    # יצירת קונפיגורציה של הפרומפט על בסיס הבקשה הראשונה (אם יש)
    first_request = display_requests[0] if display_requests else None
    prompt_config = create_prompt_config(run_spec, first_request, dataset_name)

    # עיבוד כל דוגמה
    for instance in instances:
        instance_id = instance.get("id")

        # בדיקה אם קיימים נתוני בקשה ותחזית עבור דוגמה זו
        if instance_id in request_map and instance_id in prediction_map:
            request = request_map[instance_id]
            prediction = prediction_map[instance_id]

            # יצירת דוגמה בפורמט החדש
            example = {
                "evaluation_id": create_evaluation_id(run_spec, instance_id),
                "model": create_model_section(run_spec, request),
                # "prompt_config": prompt_config,  # אותו קונפיג לכל הדוגמאות
                "instance": create_instance_section(instance, request, dataset_name),
                "output": create_output_section(prediction),
                "evaluation": create_evaluation_section(prediction, instance)
            }
            # flatten the example keys, values from nested dict to top level
            example2 = {**example, **example["instance"], **example["model"], **example["evaluation"]}
            example2.pop("instance", None)
            example2.pop("model", None)
            example2.pop("evaluation", None)
            # change the "name" to "model_name" in the model section
            example2["model_name"] = example2.pop("name", None)
            example2["model_family"] = example2.pop("family", None)
            example2["evaluation_method_name"] = example2.pop("method_name", None)
            example2["evaluation_score"] = example2.pop("score", None)
            #order - ['evaluation_id', 'dataset_name', 'hf_split', 'hf_index',  'raw_input',  'ground_truth', 'model_name', 'model_family' , 'output', 'evaluation_method_name', 'evaluation_score'])
            #sort the dict to follow the order
            example2 = {k: example2[k] for k in sorted(example2.keys(), key=lambda x: ['evaluation_id', 'dataset_name', 'hf_split', 'hf_index',
                                                                                       'raw_input', 'ground_truth', 'model_name', 'model_family',
                                                                                       'output',
                                                                                       'evaluation_method_name', 'evaluation_score'].index(x) if
            x in ['evaluation_id', 'dataset_name', 'hf_split', 'hf_index', 'raw_input', 'ground_truth', 'model_name',
                  'model_family', 'output', 'evaluation_method_name', 'evaluation_score'] else
            len(['evaluation_id', 'dataset_name', 'hf_split', 'hf_index', 'raw_input',
                 'ground_truth', 'model_name', 'model_family', 'output', 'evaluation_method_name', 'evaluation_score']))}
            examples.append(example2)

    return examples


def convert_nan_to_null(obj):
    """עובר רקורסיבית על המבנה ומחליף ערכי NaN ב-None"""
    import math
    import numpy as np

    if isinstance(obj, dict):
        return {key: convert_nan_to_null(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_null(item) for item in obj]
    elif isinstance(obj, (float, np.float64, np.float32)) and (math.isnan(obj) or np.isnan(obj)):
        return None
    else:
        return obj

def nan_to_null(obj):
    """ממיר ערכי NaN ל-null להתאמה ל-JSON"""
    import math
    import numpy as np
    if isinstance(obj, (float, np.float64, np.float32)) and (np.isnan(obj) or math.isnan(obj)):
        return None
    return obj

def main(data_dir: str, output_file: str):
    """פונקציה ראשית לעיבוד והמרת הנתונים"""
    examples = process_helm_data(data_dir)
    examples_processed = convert_nan_to_null(examples)

    # Save results as CSV
    try:
        df = pd.DataFrame(examples_processed)
        df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error saving CSV: {e}")
        raise

    print(f"Converted {len(examples)} examples and saved to {output_file}")

    # Also print the first example in JSON format for quick inspection
    if examples:
        print("\nFirst example:")
        print(json.dumps(examples_processed[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert HELM data to the new schema format")
    parser.add_argument("--data-dir", required=True, help="Directory containing HELM data files")
    parser.add_argument("--output-file", required=True, help="Output CSV file path")
    args = parser.parse_args()

    main(args.data_dir, args.output_file)

def normalize_dataset_name(dataset_name: str) -> str:
    """
    מנרמל את שם הדאטה-סט לפורמט אחיד
    למשל, ממיר openbookqa ל-openbook_qa
    """
    if not dataset_name:
        return dataset_name
        
    # המרות ספציפיות
    if dataset_name == "openbookqa":
        return "openbook_qa"
    
    # אם יש נקודה בשם (כמו mmlu.anatomy), נרמל רק את החלק הראשון
    if '.' in dataset_name:
        base_name, subject = dataset_name.split('.', 1)
        if base_name == "openbookqa":
            return f"openbook_qa.{subject}"
    
    return dataset_name