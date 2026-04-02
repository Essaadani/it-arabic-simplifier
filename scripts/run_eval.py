from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Dict, List

MODEL_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_ADAPTER = "./outputs/adapters/qwen-it-ar-simplifier-v3"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUTPUT_CSV = OUTPUT_DIR / "eval_outputs.csv"
CLEAN_OUTPUT_CSV = OUTPUT_DIR / "eval_outputs_clean.csv"


PROMPTS: List[Dict[str, str]] = [
    {
        "id": "1",
        "topic": "Object-Oriented Programming",
        "reference_simplified": "في البرمجة كائنية التوجه يحدد الصنف الخصائص والدوال التي يمكن أن يملكها الكائن.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- classe = صنف
- objet = كائن
- méthode = دالة
- attribut = خاصية

Text:
Une classe définit les propriétés et les méthodes qu’un objet peut posséder.""",
    },
    {
        "id": "2",
        "topic": "Object-Oriented Programming",
        "reference_simplified": "الوراثة تسمح لصنف جديد بإعادة استخدام الخصائص والدوال الموجودة في صنف آخر.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- héritage = وراثة
- classe = صنف
- méthode = دالة

Text:
L’héritage permet à une classe de réutiliser les propriétés et les méthodes d’une autre classe.""",
    },
    {
        "id": "3",
        "topic": "Object-Oriented Programming",
        "reference_simplified": "التغليف يحمي بيانات الكائن عبر منع الوصول المباشر إلى خصائصه.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- encapsulation = تغليف
- classe = صنف
- données = بيانات

Text:
L’encapsulation permet de protéger les données d’un objet en limitant l’accès direct à ses attributs.""",
    },
    {
        "id": "4",
        "topic": "Algorithms",
        "reference_simplified": "الخوارزمية هي مجموعة تعليمات مرتبة لحل مشكلة خطوة بخطوة.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- algorithme = خوارزمية

Text:
Un algorithme est une suite d’instructions permettant de résoudre un problème étape par étape.""",
    },
    {
        "id": "5",
        "topic": "Databases",
        "reference_simplified": "قاعدة البيانات نظام يُستخدم لتخزين البيانات وتنظيمها.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- base de données = قاعدة بيانات
- données = بيانات

Text:
Une base de données est un système utilisé pour stocker et organiser des données.""",
    },
    {
        "id": "6",
        "topic": "Databases",
        "reference_simplified": "استعلامات SQL تسمح بقراءة البيانات وتعديلها داخل قاعدة البيانات.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- requête = استعلام
- base de données = قاعدة بيانات

Text:
Les requêtes SQL permettent d’interroger et de modifier les données dans une base de données.""",
    },
    {
        "id": "7",
        "topic": "Networking",
        "reference_simplified": "عنوان IP يميز الجهاز المتصل بالشبكة.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- réseau = شبكة
- adresse IP = عنوان IP

Text:
Une adresse IP permet d’identifier un appareil connecté à un réseau.""",
    },
    {
        "id": "8",
        "topic": "Networking",
        "reference_simplified": "نظام أسماء النطاقات يحول أسماء المواقع إلى عناوين IP حتى تتمكن الأجهزة من الوصول إلى الخوادم.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- DNS = نظام أسماء النطاقات
- serveur = خادم

Text:
Le DNS traduit les noms de domaine en adresses IP afin que les ordinateurs puissent localiser les serveurs.""",
    },
    {
        "id": "9",
        "topic": "Web Infrastructure",
        "reference_simplified": "شبكة توزيع المحتوى توزع محتوى الويب على عدة خوادم لتحسين سرعة التحميل.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- CDN = شبكة توزيع المحتوى
- serveur = خادم
- contenu = محتوى

Text:
Un CDN distribue le contenu web sur plusieurs serveurs afin d’améliorer la vitesse de chargement.""",
    },
    {
        "id": "10",
        "topic": "Web Development",
        "reference_simplified": "خادم الويب يرسل صفحات الويب إلى المتصفح عندما يزور المستخدم الموقع.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- serveur web = خادم ويب
- navigateur = متصفح

Text:
Un serveur web envoie des pages web aux navigateurs lorsqu’un utilisateur visite un site.""",
    },
    {
        "id": "11",
        "topic": "Web Development",
        "reference_simplified": "HTTP بروتوكول يُستخدم لنقل صفحات الويب بين الخادم والمتصفح.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- protocole = بروتوكول
- HTTP = بروتوكول HTTP

Text:
HTTP est un protocole utilisé pour transférer des pages web entre un serveur et un navigateur.""",
    },
    {
        "id": "12",
        "topic": "Cybersecurity",
        "reference_simplified": "الجدار الناري يحمي الشبكة عبر تصفية الاتصالات الواردة والصادرة.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- pare-feu = جدار ناري
- réseau = شبكة

Text:
Un pare-feu protège un réseau en filtrant les connexions entrantes et sortantes.""",
    },
    {
        "id": "13",
        "topic": "Cybersecurity",
        "reference_simplified": "البرمجيات الخبيثة برامج صُممت لإلحاق الضرر بالنظام المعلوماتي.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- malware = برمجيات خبيثة

Text:
Les logiciels malveillants sont des programmes conçus pour endommager un système informatique.""",
    },
    {
        "id": "14",
        "topic": "Cloud Computing",
        "reference_simplified": "الحوسبة السحابية تسمح باستخدام خوادم بعيدة لتخزين البيانات ومعالجتها.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- cloud = حوسبة سحابية
- serveur = خادم

Text:
Le cloud computing permet d’utiliser des serveurs distants pour stocker et traiter des données.""",
    },
    {
        "id": "15",
        "topic": "Web Development",
        "reference_simplified": "واجهة برمجة التطبيقات تسمح لتطبيقين بالتواصل وتبادل البيانات.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- API = واجهة برمجة التطبيقات

Text:
Une API permet à deux applications de communiquer et d’échanger des données.""",
    },
    {
        "id": "16",
        "topic": "Operating Systems",
        "reference_simplified": "نظام التشغيل يدير الموارد المادية والبرمجية في الحاسوب.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- système d'exploitation = نظام التشغيل

Text:
Le système d’exploitation gère les ressources matérielles et logicielles d’un ordinateur.""",
    },
    {
        "id": "17",
        "topic": "Programming Fundamentals",
        "reference_simplified": "المتغير مساحة في الذاكرة تُستخدم لتخزين قيمة داخل البرنامج.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- variable = متغير

Text:
Une variable est un espace mémoire utilisé pour stocker une valeur dans un programme.""",
    },
    {
        "id": "18",
        "topic": "Programming Fundamentals",
        "reference_simplified": "الحلقة تسمح بتكرار مجموعة من التعليمات عدة مرات داخل البرنامج.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- boucle = حلقة

Text:
Une boucle permet de répéter une série d’instructions plusieurs fois dans un programme.""",
    },
    {
        "id": "19",
        "topic": "Web Development",
        "reference_simplified": "صفحة الويب وثيقة تظهر داخل متصفح الإنترنت.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- page web = صفحة ويب
- navigateur = متصفح

Text:
Une page web est un document affiché dans un navigateur internet.""",
    },
    {
        "id": "20",
        "topic": "Software Engineering",
        "reference_simplified": "هندسة البرمجيات هي عملية تصميم البرامج وتطويرها وصيانتها.",
        "prompt": """Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- logiciel = برنامج
- ingénierie logicielle = هندسة البرمجيات

Text:
L’ingénierie logicielle consiste à concevoir, développer et maintenir des logiciels.""",
    },
]


def run_generate(prompt: str, adapter_path: str | None = None) -> str:
    cmd = [
        "python",
        "-m",
        "mlx_lm",
        "generate",
        "--model",
        MODEL_BASE,
        "--prompt",
        prompt,
    ]

    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    return result.stdout.strip()


def extract_main_output(raw_text: str) -> str:
    if "==========" in raw_text:
        parts = raw_text.split("==========")
        candidates = [p.strip() for p in parts if p.strip()]
        if candidates:
            return candidates[0]

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    cleaned_lines = []
    for line in lines:
        if line.startswith("Fetching "):
            continue
        if line.startswith("Prompt: "):
            continue
        if line.startswith("Generation: "):
            continue
        if line.startswith("Peak memory: "):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def main() -> None:
    rows_raw = []
    rows_clean = []

    total = len(PROMPTS)
    for idx, item in enumerate(PROMPTS, start=1):
        print(f"[{idx}/{total}] Running prompt {item['id']} - {item['topic']}")

        base_raw = run_generate(item["prompt"])
        ft_raw = run_generate(item["prompt"], adapter_path=MODEL_ADAPTER)

        base_clean = extract_main_output(base_raw)
        ft_clean = extract_main_output(ft_raw)

        rows_raw.append(
            {
                "id": item["id"],
                "topic": item["topic"],
                "prompt": item["prompt"],
                "reference_simplified": item["reference_simplified"],
                "base_raw_output": base_raw,
                "fine_tuned_raw_output": ft_raw,
            }
        )

        rows_clean.append(
            {
                "id": item["id"],
                "topic": item["topic"],
                "prompt": item["prompt"],
                "reference_simplified": item["reference_simplified"],
                "base_output": base_clean,
                "fine_tuned_output": ft_clean,
                "base_fluency": "",
                "base_meaning": "",
                "base_simplicity": "",
                "base_terminology": "",
                "ft_fluency": "",
                "ft_meaning": "",
                "ft_simplicity": "",
                "ft_terminology": "",
                "notes": "",
            }
        )

    with RAW_OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "topic",
                "prompt",
                "reference_simplified",
                "base_raw_output",
                "fine_tuned_raw_output",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_raw)

    with CLEAN_OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "topic",
                "prompt",
                "reference_simplified",
                "base_output",
                "fine_tuned_output",
                "base_fluency",
                "base_meaning",
                "base_simplicity",
                "base_terminology",
                "ft_fluency",
                "ft_meaning",
                "ft_simplicity",
                "ft_terminology",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_clean)

    print(f"\nSaved raw outputs to: {RAW_OUTPUT_CSV}")
    print(f"Saved clean outputs to: {CLEAN_OUTPUT_CSV}")


if __name__ == "__main__":
    main()