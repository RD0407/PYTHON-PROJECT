# compatibility_matcher.py
import json
import os
from typing import List, Dict, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich import box

console = Console()

DATA_FILE = "candidates.json"

# ========== Personality & Emojis ==========
PERSONALITY_TRAITS = [
    "Kindness","Humor","Intellect","Confidence","Caring","Adventurous",
    "Honesty","Loyalty","Empathy","Creativity","Patience","Ambition",
    "Optimism","Maturity","Playfulness","Calmness","Leadership",
    "Sociability","Discipline","Open-mindedness"
]

TRAIT_EMOJI = {
    "Kindness":"💖","Humor":"😂","Intellect":"🧠","Confidence":"💪","Caring":"🤗","Adventurous":"🌍",
    "Honesty":"🗝️","Loyalty":"🤝","Empathy":"❤️","Creativity":"🎨","Patience":"⏳","Ambition":"🚀",
    "Optimism":"☀️","Maturity":"🧘","Playfulness":"🎭","Calmness":"🌊","Leadership":"👑",
    "Sociability":"🗣️","Discipline":"📏","Open-mindedness":"🌈"
}

# ========== Physical Traits & Options ==========
PHYSICAL_TRAITS_BY_GENDER = {
    "male": [
        "Height","Weight","Hair Color","Hair Type","Eye Color","Skin Tone",
        "Body Type","Facial Hair","Penis Size"
    ],
    "female": [
        "Height","Weight","Hair Color","Hair Type","Eye Color","Skin Tone",
        "Body Type","Breast Size"
    ],
}

PHYSICAL_EMOJI = {
    "Height":"📏","Weight":"⚖️","Hair Color":"💇","Hair Type":"🌀","Eye Color":"👀","Skin Tone":"🎨",
    "Body Type":"🏋️","Facial Hair":"🧔","Penis Size":"🍆","Breast Size":"👙"
}

PHYSICAL_OPTIONS = {
    "Height": ["<155 cm","155–160","160–165","165–170","170–175","175–180","180–185",">185 cm"],
    "Weight": ["<45 kg","45–50","50–55","55–60","60–65","65–70","70–80",">80 kg"],
    "Hair Color": ["Black","Dark Brown","Brown","Light Brown","Blonde","Red","Grey/White","Other"],
    "Hair Type": ["Straight","Wavy","Curly","Coily","Bald/Shaved"],
    "Eye Color": ["Brown","Dark Brown","Hazel","Green","Blue","Grey"],
    "Skin Tone": ["Fair","Light","Medium","Olive","Tan","Brown","Dark"],
    "Body Type": ["Slim","Toned","Athletic","Muscular","Average","Curvy","Plus-size"],
    "Facial Hair": ["Clean-shaven","Light stubble","Heavy stubble","Short beard","Full beard","Mustache"],
    "Penis Size": ["<12 cm","12–15 cm","15–18 cm",">18 cm"],
    "Breast Size": ["Flat","Small","Medium","Large","Extra Large"]
}

# Ranking weights for up to 3 preferences per physical trait
PREF_WEIGHTS = [1.0, 0.7, 0.5]

CONTINENTS = ["No Preference","Europe","Asia","Africa","North America","South America","Oceania"]
LANGUAGES = ["No Preference","English","Italian","German","French","Spanish"]
SMOKING_PREFS = ["Doesn't matter", "Yes", "No"]

# =================== Helpers ===================
def load_candidates(gender: str) -> List[Dict]:
    if not os.path.exists(DATA_FILE):
        console.print(f"[red]❌ {DATA_FILE} not found. Create it next to this script.[/red]")
        return []
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    return data.get(gender.lower(), [])

def generate_age_ranges(start=18, end=60, step=3) -> List[str]:
    return [f"{i}–{i+step}" for i in range(start, end, step)]

def safe_int_input(prompt, min_val, max_val, default=None):
    while True:
        try:
            val = input(prompt).strip()
            if not val and default is not None:
                return default
            num = int(val)
            if min_val <= num <= max_val:
                return num
        except ValueError:
            pass
        console.print(f"[red]Enter a number between {min_val} and {max_val}[/red]")

def safe_float_input(prompt, min_val, max_val, default=None):
    while True:
        try:
            val = input(prompt).strip()
            if not val and default is not None:
                return default
            num = float(val)
            if min_val <= num <= max_val:
                return num
        except ValueError:
            pass
        console.print(f"[red]Enter a number between {min_val} and {max_val}[/red]")

def pick_two_age_ranges() -> List[str]:
    console.print(Panel.fit("📅 Age Preference — pick TWO ranges (3-year buckets)", style="bold cyan"))
    ranges = generate_age_ranges()
    for i, ar in enumerate(ranges, 1):
        console.print(f"{i}. {ar}")
    total = len(ranges)
    a = safe_int_input(f"Select first range [1–{total}]: ", 1, total) - 1
    b = safe_int_input(f"Select second range [1–{total}] (different): ", 1, total) - 1
    if a == b:
        console.print("[yellow]Same range picked twice — using only one.[/yellow]")
        return [ranges[a]]
    return [ranges[a], ranges[b]]

def ask_continent() -> str:
    console.print(Panel.fit("🌍 Nationality (continent) — 0 = No Preference", style="bold cyan"))
    for i, c in enumerate(CONTINENTS, 0):
        console.print(f"{i}. {c}")
    c = safe_int_input("Pick continent number: ", 0, len(CONTINENTS)-1, default=0)
    return CONTINENTS[int(c)]

def ask_languages(max_choices=3) -> List[str]:
    console.print(Panel.fit(f"🗣️ Languages — pick up to {max_choices} (0 = No Preference)", style="bold cyan"))
    for i, l in enumerate(LANGUAGES, 0):
        console.print(f"{i}. {l}")
    chosen = set()
    while len(chosen) < max_choices:
        val = safe_int_input(f"Add language #{len(chosen)+1}: ", 0, len(LANGUAGES)-1, default=0)
        if val == 0:
            break
        chosen.add(LANGUAGES[val])
        if "No Preference" in chosen:
            return []
    return list(chosen)

def ask_smoking_pref() -> str:
    console.print(Panel.fit("🚬 Smoking Preference — 0 = Doesn't matter", style="bold cyan"))
    for i, s in enumerate(SMOKING_PREFS, 0):
        console.print(f"{i}. {s}")
    val = safe_int_input("Pick smoking preference: ", 0, len(SMOKING_PREFS)-1, default=0)
    return SMOKING_PREFS[val]

def ask_personality_preferences() -> Dict[str, float]:
    console.print(Panel.fit("🧠 Personality — pick TOP 6 traits, then desired rating (0–10)", style="bold green"))
    table = Table(title="Available Traits", box=box.SIMPLE_HEAVY)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Trait", style="bold")
    rows = [(i, f"{TRAIT_EMOJI[t]} {t}") for i, t in enumerate(PERSONALITY_TRAITS, 1)]
    half = (len(rows)+1)//2
    for ix in range(half):
        left = rows[ix]
        right = rows[ix+half] if ix+half < len(rows) else ("","")
        table.add_row(str(left[0]), left[1] + (" " * 5) + (f"{right[0]}  {right[1]}" if right[0] != "" else ""))
    console.print(table)

    picks = []
    while len(picks) < 6:
        idx = safe_int_input(f"Pick trait #{len(picks)+1} (1–{len(PERSONALITY_TRAITS)}): ", 1, len(PERSONALITY_TRAITS))
        if idx not in picks:
            picks.append(idx)

    prefs = {}
    for p in picks:
        trait = PERSONALITY_TRAITS[p-1]
        desired = safe_float_input(f"Desired {TRAIT_EMOJI[trait]} {trait} (0–10): ", 0.0, 10.0, default=8.0)
        prefs[trait] = desired
    return prefs

def ask_physical_preferences(gender: str) -> Dict[str, List[str]]:
    traits = PHYSICAL_TRAITS_BY_GENDER[gender]
    console.print(Panel.fit("💪 Physical — for EACH trait pick up to 3 preferred options (0 = No Preference)", style="bold yellow"))
    all_prefs = {}
    for trait in traits:
        options = PHYSICAL_OPTIONS[trait]
        table = Table(title=f"{PHYSICAL_EMOJI.get(trait,'')} {trait}", box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Option")
        for i, opt in enumerate(options, 1):
            table.add_row(str(i), opt)
        table.add_row("0", "No Preference")
        console.print(table)

        prefs = []
        for r in range(3):
            choice = safe_int_input(f"Pick preference #{r+1} for {trait}: ", 0, len(options), default=0)
            if choice == 0:
                prefs = []
                break
            pick = options[choice-1]
            if pick not in prefs:
                prefs.append(pick)
        all_prefs[trait] = prefs
    return all_prefs

# ========== Scoring ==========
def in_any_age_bucket(age: int, buckets: List[str]) -> bool:
    for b in buckets:
        lo, hi = b.split("–")
        if int(lo) <= age <= int(hi):
            return True
    return False

def personality_score(user_prefs: Dict[str, float], cand_personality: Dict[str, float]) -> Tuple[float,float]:
    score = 0.0
    maxs = 0.0
    for trait, desired in user_prefs.items():
        cand = cand_personality.get(trait, 0.0)
        diff = abs(desired - cand)
        score += (10.0 - diff)
        maxs += 10.0
    return score, maxs

def physical_score(user_phys: Dict[str, List[str]], cand_phys: Dict[str, str]) -> Tuple[float,float]:
    score = 0.0
    maxs = 0.0
    for trait, prefs in user_phys.items():
        if not prefs:
            continue
        cand_val = cand_phys.get(trait, None)
        for i, pref in enumerate(prefs[:3]):
            if cand_val == pref:
                score += 10.0 * PREF_WEIGHTS[i]
        maxs += 10.0
    return score, maxs

def misc_score(age_match: bool, nat_ok: bool, lang_ok: bool, smoke_ok: bool) -> Tuple[float,float]:
    score = 0.0
    maxs = 40.0
    if age_match:
        score += 10.0
    if nat_ok:
        score += 10.0
    if lang_ok:
        score += 10.0
    if smoke_ok:
        score += 10.0
    return score, maxs

def compute_compatibility(user_personality, user_physical, age_buckets, nat_pref, lang_prefs, smoking_pref, cand) -> float:
    p_s, p_m = personality_score(user_personality, cand["personality"])
    f_s, f_m = physical_score(user_physical, cand["physical"])
    age_ok = in_any_age_bucket(cand["age"], age_buckets)
    nat_ok = (nat_pref == "No Preference") or (cand.get("nationality") == nat_pref)
    lang_ok = True if not lang_prefs or "No Preference" in lang_prefs else any(l in cand.get("languages", []) for l in lang_prefs)
    smoke_ok = (smoking_pref == "Doesn't matter") or (cand.get("smoking") == smoking_pref)
    m_s, m_m = misc_score(age_ok, nat_ok, lang_ok, smoke_ok)

    p = (p_s / p_m) if p_m > 0 else 1.0
    f = (f_s / f_m) if f_m > 0 else 1.0
    m = (m_s / m_m) if m_m > 0 else 1.0

    return round((0.40 * p + 0.50 * f + 0.10 * m) * 100.0, 2)

# =================== Main Flow ===================
def main():
    console.print(Panel.fit("💘 Compatibility Matcher", style="bold magenta"))

    gender = Prompt.ask("Looking for", choices=["male","female"], default="female")
    candidates = load_candidates(gender)
    if not candidates:
        return

    age_buckets = pick_two_age_ranges()
    nat_pref = ask_continent()
    lang_prefs = ask_languages(max_choices=3)
    smoking_pref = ask_smoking_pref()
    user_personality = ask_personality_preferences()
    user_physical = ask_physical_preferences(gender)

    results = []
    for c in candidates:
        comp = compute_compatibility(user_personality, user_physical, age_buckets, nat_pref, lang_prefs, smoking_pref, c)
        results.append((c, comp))
    results.sort(key=lambda x: x[1], reverse=True)

    table = Table(title="💖 Your Matches", box=box.DOUBLE_EDGE)
    table.add_column("Name", style="bold cyan")
    table.add_column("Age", justify="center")
    table.add_column("Nationality", justify="center")
    table.add_column("Languages", justify="center")
    table.add_column("Compatibility %", style="bold magenta", justify="center")

    for cand, comp in results:
        table.add_row(cand["name"], str(cand["age"]), cand.get("nationality","?"), ", ".join(cand.get("languages",[])), f"{comp}")
    console.print(table)

    # Summary per candidate
    for cand, comp in results:
        high_matches = []
        lacking = []

        for trait, desired in user_personality.items():
            cand_val = cand["personality"].get(trait, 0)
            if abs(desired - cand_val) <= 2:
                high_matches.append(f"{TRAIT_EMOJI[trait]} {trait}")
            else:
                lacking.append(f"{TRAIT_EMOJI[trait]} {trait}")

        for trait, prefs in user_physical.items():
            if not prefs:
                continue
            cand_val = cand["physical"].get(trait, None)
            if cand_val in prefs:
                high_matches.append(f"{PHYSICAL_EMOJI.get(trait,'')} {trait} ({cand_val})")
            else:
                lacking.append(f"{PHYSICAL_EMOJI.get(trait,'')} {trait} ({cand_val})")

        if smoking_pref != "Doesn't matter":
            if cand.get("smoking") == smoking_pref:
                high_matches.append(f"🚭 Smoking preference matched ({cand.get('smoking')})")
            else:
                lacking.append(f"🚬 Smoking preference mismatch ({cand.get('smoking')})")

        console.print(Panel.fit(
            f"[bold]{cand['name']}[/bold] — {comp}% match\n\n"
            f"[green]✅ Why YES:[/green] " + (", ".join(high_matches) if high_matches else "No major matches") +
            "\n[red]❌ What’s lacking:[/red] " + (", ".join(lacking) if lacking else "Nothing major"),
            style="bold cyan"
        ))

if __name__ == "__main__":
    main()

