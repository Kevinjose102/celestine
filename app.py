"""
app.py
======
Ion Thruster AI — Complete Physics Dashboard
All modules:
  - sheath_erosion.py             (sheath + CEX erosion)
  - air_plasma.py                 (air-breathing physics)
  - air_thruster_deepdive.py      (EHD thrust, geometry optimizer, multi-stage)

Run:
  python app.py
Open: http://localhost:5000
"""

import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
from openai import OpenAI as OpenRouterClient  # OpenRouter uses OpenAI-compatible API

class SheathErosionAI:
    def __init__(self, *args, **kwargs): pass
class ElectrostaticsAI:
    pass

from air_plasma import AirPlasmaAI
from air_thruster_deepdive import AirThrusterAI
from propellants_db import PROPELLANTS as PROPELLANTS_DB, compare_propellants, list_vacuum_propellants, list_ehd_propellants, get_propellant
from emitter_array import EmitterArrayAI

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GROQ_API_KEY       = "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"
OPENROUTER_API_KEY = "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"

# ── DUAL-MODEL ARCHITECTURE ──────────────────────────────────────────────────
# BRAIN  : DeepSeek R1 via OpenRouter — real reasoning, picks the right physics fn
# VOICE  : Llama 3.3 70B via Groq    — high TPD, handles 8000-token explanations
#
# Flow:
#   message → brain_call() [OpenRouter, DeepSeek R1, ≤400 tokens]
#           → physics engine runs locally
#           → voice_call() [Groq, Llama 3.3 70B, ≤8000 tokens]
#
# If OpenRouter fails → brain falls back to Groq llama-3.3-70b-versatile
# If Groq voice fails → voice falls back to Kimi K2
# ─────────────────────────────────────────────────────────────────────────────

# ── MODEL CONFIG ─────────────────────────────────────────────────────────────
# PRIMARY: GPT-OSS 120B on Groq — best physics quality, free, fast
# FALLBACK: Llama 3.3 70B on Groq — when GPT-OSS 120B hits 200k TPD
# BRAIN (routing): DeepSeek R1 free on OpenRouter — JSON routing only
# ─────────────────────────────────────────────────────────────────────────────

PRIMARY_MODEL        = "openai/gpt-oss-120b"           # best physics, 200k TPD
FALLBACK_MODEL       = "llama-3.3-70b-versatile"       # when primary hits limit
BRAIN_MODEL          = "deepseek/deepseek-r1:free"     # OpenRouter free, routing only
BRAIN_FALLBACK       = "llama-3.3-70b-versatile"       # Groq fallback if OR fails

# Aliases for rest of codebase
VOICE_MODEL          = PRIMARY_MODEL
VOICE_FALLBACK       = FALLBACK_MODEL
REASONING_MODEL      = BRAIN_MODEL
INTERACTION_MODEL    = PRIMARY_MODEL
MODEL                = PRIMARY_MODEL

AVAILABLE_MODELS = [
    {"id": "openai/gpt-oss-120b",          "name": "GPT-OSS 120B",     "note": "Primary · Groq · best physics · 200k TPD"},
    {"id": "llama-3.3-70b-versatile",      "name": "Llama 3.3 70B",    "note": "Fallback · Groq · 500k TPD"},
    {"id": "openai/gpt-oss-20b",           "name": "GPT-OSS 20B",      "note": "Fast · Groq · lower quality"},
    {"id": "moonshotai/kimi-k2-instruct",  "name": "Kimi K2",          "note": "High limits · Groq"},
    {"id": "deepseek/deepseek-r1:free",    "name": "DeepSeek R1 Free", "note": "Brain/routing · OpenRouter free"},
]

SUPPORTS_REASONING_EFFORT = {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}
OPENROUTER_MODELS = {"deepseek/deepseek-r1", "deepseek/deepseek-r1:free",
                     "deepseek/deepseek-r1-zero", "anthropic/claude-3.5-sonnet"}


def make_groq_call(messages, effort="high", max_tokens=8000, model=None):
    """Call Groq API. effort only applied to SUPPORTS_REASONING_EFFORT models."""
    m = model or VOICE_MODEL
    kwargs = {"model": m, "messages": messages, "max_tokens": max_tokens}
    if m in SUPPORTS_REASONING_EFFORT:
        kwargs["reasoning_effort"] = effort
    return groq_client.chat.completions.create(**kwargs)


def make_or_call(messages, max_tokens=400, model=None):
    """Call OpenRouter API (OpenAI-compatible). Used for brain/reasoning."""
    m = model or BRAIN_MODEL
    return or_client.chat.completions.create(
        model=m,
        messages=messages,
        max_tokens=max_tokens,
        extra_headers={
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Celestine Ion Propulsion AI",
        }
    )


def reasoning_call(messages, max_tokens=400):
    """
    Brain call — DeepSeek R1 via OpenRouter picks the physics function.
    Falls back to Groq Llama if OpenRouter is unavailable.
    """
    try:
        return make_or_call(messages, max_tokens=max_tokens, model=BRAIN_MODEL)
    except Exception as e:
        err = str(e)
        # Any OR failure → fall back to Groq
        print(f"[brain] OpenRouter failed ({err[:80]}), falling back to Groq")
        return make_groq_call(messages, effort="low", max_tokens=max_tokens,
                              model=BRAIN_FALLBACK)


def interaction_call(messages, max_tokens=8000):
    """
    Voice call — GPT-OSS 120B (best physics quality).
    Auto-fallback to Llama 3.3 70B when 200k TPD is exhausted.
    """
    try:
        return make_groq_call(messages, effort="high", max_tokens=max_tokens,
                              model=PRIMARY_MODEL)
    except Exception as e:
        err = str(e)
        if "rate_limit" in err or "429" in err or "TPD" in err or "decommission" in err:
            print(f"[voice] {PRIMARY_MODEL} limit hit, falling back to {FALLBACK_MODEL}")
            return make_groq_call(messages, effort="low", max_tokens=max_tokens,
                                  model=FALLBACK_MODEL)
        raise


# Legacy alias so any remaining make_call() references still work
def make_call(messages, effort="high", max_tokens=8000, model=None):
    return make_groq_call(messages, effort=effort, max_tokens=max_tokens, model=model)
PROPELLANT    = "Xenon"
GRID_MATERIAL = "Molybdenum"

app          = Flask(__name__)
CORS(app)
groq_client  = Groq(api_key=GROQ_API_KEY)
or_client    = OpenRouterClient(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    timeout=600,  # 10 min — DeepSeek R1 reasoning can be slow
)
client       = groq_client   # legacy alias
ai_sheath    = SheathErosionAI(propellant=PROPELLANT, grid_material=GRID_MATERIAL)
ai_air       = AirPlasmaAI()
ai_ehd       = AirThrusterAI()
ai_estatic   = ElectrostaticsAI()
ai_array     = EmitterArrayAI()

conversation_history = []

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

# ── BRAIN PROMPT — sent to DeepSeek R1, JSON output only ──────────────────
# The brain must ONLY output the JSON function call, nothing else.
# No explanations, no prose, no "let me call...", no reasoning text.
# ── CLASSIFIER PROMPT — fast Groq Llama call, routes every message ─────────
# Replaces both keyword matching AND DeepSeek routing.
# Uses llama-3.3-70b-versatile (same as voice, but tiny output ~30 tokens).
# Outputs ONE JSON line or CONVERSATIONAL. No prose, no reasoning text.
CLASSIFIER_PROMPT = """You are a JSON router for an ion thruster physics engine.
Read the user message. Output ONE JSON line matching the best function. Nothing else.

FUNCTION MAP:
needle tip field / electric field at tip / tip concentration / enhancement factor
  → {"function":"tip_field","voltage_V":<V>,"tip_radius_mm":<r>,"emitter_length_mm":<L>}

needle array / multiple needles / arrange needles / maximize emitters / needle spacing / shielding / array design / how many emitters / emitter layout / packing
  → {"function":"array_design","diameter_mm":<D>,"voltage_V":<V>,"emitter_dia_mm":4.0,"tip_radius_mm":0.1,"material":"copper"}

voltage sweep / safe voltage range / at what voltage / onset sweep
  → {"function":"field_voltage_sweep","gap_mm":<g>,"emitter_radius_mm":<r>,"collector_inner_mm":<c>,"V_max":40000}

insulation / creepage / PETG safe / housing material / HV material
  → {"function":"insulation_check","voltage_V":<V>,"creepage_mm":<c>,"material":"PETG","humidity_pct":80}

2D field map / solve field / Poisson / full field distribution
  → {"function":"solve_field","voltage_V":<V>,"gap_mm":<g>,"emitter_radius_mm":<r>,"collector_inner_mm":<c>,"emitter_length_mm":<L>}

wire inside ring / coaxial / wire-to-cylinder
  → {"function":"coaxial_field","voltage_V":<V>,"wire_radius_mm":<r>,"cylinder_radius_mm":<R>}

new EHD thruster design / build prototype / design air thruster
  → {"function":"ehd_optimize_gap","voltage_V":<V>,"emitter_radius_mm":0.1,"target":"balanced"}

EHD performance / thrust for given setup / analyse existing design
  → {"function":"ehd_analysis","voltage_V":<V>,"current_mA":<I>,"gap_mm":<g>,"emitter_radius_mm":<r>,"emitter_length_mm":<L>}

air discharge / breakdown / corona analysis
  → {"function":"air_discharge","voltage_V":<V>,"pressure_Pa":101325,"gap_mm":<g>,"current_A":0.001}

parts / buy / sourcing / shopping
  → {"function":"parts_list","voltage_kV":<V>,"gap_mm":<g>,"emitter_dia_mm":1,"collector_inner_mm":62}

troubleshoot / not working / sparking / arc / no thrust
  → {"function":"troubleshoot","symptom":"<problem>","voltage_V":<V>,"gap_mm":<g>}

anything else (conceptual, follow-up, material choice, how X works)
  → CONVERSATIONAL

PARAMETER DEFAULTS if not mentioned: V=20000, r=0.1, L=30, g=23, c=46, N=8, s=10, I=1, R=50
For diameter mentions like "10cm diameter": N = round(pi * diameter_mm / spacing_mm)
OUTPUT: one line only. Replace all <placeholders> with numbers."""

# ── REQUIRED PARAMETERS — only what physics cannot derive ────────────────────
#
# Philosophy: the AI is the expert. It should COMPUTE geometry, tip radius,
# spacing, creepage, gap — these are OUTPUTS not inputs.
#
# The ONLY things that cannot be derived from physics alone:
#   1. Voltage — set by your power supply hardware (BUT can be optional if
#      the user asks "what voltage should I use" — then AI recommends)
#   2. Physical size constraint — if user has a fixed housing diameter
#   3. Target performance — thrust, lifetime, safety margin
#
# Everything else the AI calculates:
#   tip_radius    → AI recommends based on desired regime (stable corona = 0.1-0.3mm)
#   gap           → AI calculates from Peek onset + safety margin for given voltage
#   creepage      → AI calculates minimum from IEC 60664 + voltage + humidity
#   needle spacing→ AI calculates from shielding factor (η > 0.8 target)
#   n_needles     → AI calculates from diameter + optimal spacing
#   collector_dia → AI calculates as 2× gap (field uniformity rule)
#
# Only ask user when the question is genuinely ambiguous — not enough context
# to determine even a reasonable default.

REQUIRED_PARAMS = {
    # tip_field: voltage optional — if missing, AI runs sweep and recommends
    "tip_field": {
        "voltage_V": (False, None),   # AI can recommend if missing
    },
    # needle_array: only need voltage if user wants specific field values
    # diameter gives us n_needles; spacing is computed from shielding physics
    "needle_array_field": {
        "voltage_V": (False, None),
    },
    # voltage sweep: needs nothing — AI sweeps 1kV–40kV by default
    "field_voltage_sweep": {},
    # insulation: voltage is the one thing AI cannot guess — it's your hardware
    # creepage is computed by AI from voltage + IEC standard
    "insulation_check": {
        "voltage_V": (True, "What is the maximum voltage your housing will see? (e.g. 20kV, 30kV)"),
    },
    # solve_field: voltage is needed; everything else AI derives
    "solve_field": {
        "voltage_V": (True, "What voltage are you applying?"),
    },
    # coaxial: voltage + at least one geometry dimension
    "coaxial_field": {
        "voltage_V": (True, "What voltage are you applying?"),
    },
    # EHD design: voltage is the starting point — everything else is optimised
    "ehd_optimize_gap": {
        "voltage_V": (True, "What supply voltage are you using? (e.g. 20kV, 30kV)"),
    },
    # EHD analysis: voltage + gap are needed to analyse existing setup
    "ehd_analysis": {
        "voltage_V": (True, "What voltage are you applying?"),
        "gap_mm":    (True, "What gap distance are you using? (mm)"),
    },
    # Array design: need housing diameter + voltage; rest derived from physics
    "array_design": {
        "diameter_mm": (True, "What is your housing inner diameter? (e.g. 100mm, 10cm)"),
        "voltage_V":   (False, None),  # physics default applies
    },
}

# ── PHYSICS-DERIVED DEFAULTS ──────────────────────────────────────────────────
# When parameters are not given, compute them from physics rather than
# using arbitrary numbers. These are used in the classifier to fill gaps.
def physics_defaults(fn: str, params: dict, user_msg: str = "") -> dict:
    """
    Fill missing parameters using physics derivation, not arbitrary defaults.
    Voltage default depends on device type inferred from context.
    """
    import math

    p = dict(params)

    # ── Infer device type from function + user message context ───────────────
    _msg = user_msg.lower()
    _is_vacuum = any(w in _msg for w in ["xenon","krypton","argon","xe","kr","vacuum","gridded","ion thruster","screen","accel grid"])
    _is_ehd    = any(w in _msg for w in ["ehd","air","corona","emitter","needle","atmospheric","ionic wind","integza"])
    # Function-based inference when message has no clue
    _ehd_fns    = {"tip_field","needle_array_field","field_voltage_sweep","insulation_check","solve_field","coaxial_field","ehd_optimize_gap","ehd_analysis","air_discharge","ehd_analysis"}
    _vacuum_fns = {"sheath", "ion_optics", "propellants", "propellant_info"}
    if not _is_vacuum and not _is_ehd:
        _is_ehd    = fn in _ehd_fns
        _is_vacuum = fn in _vacuum_fns

    # ── Voltage default: physics-justified per device ────────────────────────
    if "voltage_V" not in p:
        if _is_vacuum:
            # Xe gridded ion thruster: typical screen voltage 800V
            # (Child-Langmuir limit for 0.5mm grid gap, 1e17 m⁻³ plasma density)
            p["voltage_V"] = 800
        elif _is_ehd:
            # Air EHD: Peek onset for 0.1mm tip ≈ 8kV; practical operation 15–25kV
            # Default to onset × 2 for stable corona
            p["voltage_V"] = 16000
        else:
            # Unknown — use EHD default and flag it
            p["voltage_V"] = 16000
        p["_voltage_assumed"] = True   # flag so voice model can mention assumption

    V = p["voltage_V"]

    if fn in ("tip_field", "needle_array_field", "solve_field",
              "coaxial_field", "ehd_optimize_gap", "ehd_analysis",
              "array_design"):

        # Tip radius: for stable corona (ratio 3–10), target E_tip = 5 × E_break
        # E_tip = V/(r·ln(2h/r)) ≈ V/(r·9) for h=30mm, r=0.1mm → r = V/(5·E_break·9)
        if "tip_radius_mm" not in p:
            r_opt = V / (5 * 3e6 * 9)           # target ratio=5 stable corona
            r_opt = max(0.05, min(0.5, r_opt * 1000))  # clamp 0.05–0.5 mm
            p["tip_radius_mm"] = round(r_opt, 3)

        # Emitter length: 2–3× tip radius × 100 (practical: 20–50mm)
        if "emitter_length_mm" not in p:
            p["emitter_length_mm"] = 30

        # Gap: from Peek onset + 2× safety margin
        # V_onset = E_onset · r · ln(2h/r); gap = V / (E_break × 0.3)
        if "gap_mm" not in p:
            gap = V / (3e6 * 0.25) * 1000       # gap where E_avg = 25% of breakdown
            p["gap_mm"] = round(max(5, min(100, gap)), 1)

        # Collector inner = 2× gap (field uniformity)
        if "collector_inner_mm" not in p and "gap_mm" in p:
            p["collector_inner_mm"] = round(p["gap_mm"] * 2, 1)

        # Wire/cylinder for coaxial
        if fn == "coaxial_field":
            if "wire_radius_mm" not in p:
                p["wire_radius_mm"] = p.get("tip_radius_mm", 0.1)
            if "cylinder_radius_mm" not in p:
                p["cylinder_radius_mm"] = p.get("collector_inner_mm", p["gap_mm"] * 2 if "gap_mm" in p else 46)

    if fn == "needle_array_field":
        # Optimal spacing: shielding η > 0.8 → s > -h/2.3 × ln(0.2)
        h = p.get("emitter_length_mm", 30) / 1000
        s_opt = -h / 2.3 * math.log(0.2) * 1000   # mm
        if "spacing_mm" not in p:
            p["spacing_mm"] = round(max(5, s_opt), 1)
        # n_needles from diameter if given
        if "n_needles" not in p:
            if "diameter_mm" in p:
                p["n_needles"] = max(3, round(math.pi * p["diameter_mm"] / p["spacing_mm"]))
            else:
                p["n_needles"] = 8

    if fn == "insulation_check":
        # Creepage: IEC 60664-1 pollution degree 3 at 80% RH = 3.2mm/kV × humidity_factor
        if "creepage_mm" not in p:
            hf = 1.0 + (p.get("humidity_pct", 80) - 50) / 100 * 1.5
            p["creepage_mm"] = round(V / 1000 * 3.2 * hf, 1)
        if "humidity_pct" not in p:
            p["humidity_pct"] = 80
        if "material" not in p:
            p["material"] = "PETG"

    if fn == "field_voltage_sweep":
        if "gap_mm" not in p:
            p["gap_mm"] = round(V / (3e6 * 0.25) * 1000, 1) if V else 23
        if "emitter_radius_mm" not in p:
            p["emitter_radius_mm"] = p.get("tip_radius_mm", 0.1)
        if "collector_inner_mm" not in p:
            p["collector_inner_mm"] = round(p["gap_mm"] * 2, 1)
        if "V_max" not in p:
            p["V_max"] = 40000

    return p


# Pending call state — stores intent + params while waiting for user input
# Format: {"function": "tip_field", "params": {...}, "missing": ["voltage_V"]}
_pending_call = {}

BRAIN_PROMPT = """You are a function router. Output ONE JSON line. Nothing else. No prose.

ROUTING MAP — match the user question to the best function:

electric field / tip field / field at needle / field concentration / enhancement factor
  → {"function":"tip_field","voltage_V":<V>,"tip_radius_mm":<r>,"emitter_length_mm":<L>}
  EXAMPLES:
  "what is the electric field at my needle tip? 20kV, 0.1mm tip radius, 30mm long emitter"
    → {"function":"tip_field","voltage_V":20000,"tip_radius_mm":0.1,"emitter_length_mm":30}
  "field at 0.05mm tip, 30kV, 25mm emitter"
    → {"function":"tip_field","voltage_V":30000,"tip_radius_mm":0.05,"emitter_length_mm":25}

wire-to-cylinder / coaxial field / wire inside collector ring
  → {"function":"coaxial_field","voltage_V":<V>,"wire_radius_mm":<r>,"cylinder_radius_mm":<R>}

needle array / multiple needles / N needles / array spacing / shielding between needles
  → {"function":"needle_array_field","voltage_V":<V>,"tip_radius_mm":<r>,"emitter_length_mm":<L>,"n_needles":<N>,"spacing_mm":<s>}

2D field map / solve Poisson / full field solution / field distribution map
  → {"function":"solve_field","voltage_V":<V>,"gap_mm":<g>,"emitter_radius_mm":<r>,"collector_inner_mm":<c>,"emitter_length_mm":<L>}

voltage sweep / safe operating window / at what voltage does corona start / onset voltage sweep
  → {"function":"field_voltage_sweep","gap_mm":<g>,"emitter_radius_mm":<r>,"collector_inner_mm":<c>,"V_max":40000}

insulation / creepage distance / is PETG safe / HV material / housing at high voltage
  → {"function":"insulation_check","voltage_V":<V>,"creepage_mm":<c>,"material":"PETG","humidity_pct":80}

design new EHD / build a thruster / design prototype / new air thruster / ionic propulsion
  → {"function":"ehd_optimize_gap","voltage_V":<V>,"emitter_radius_mm":0.1,"target":"balanced"}

analyse existing EHD / thrust for given setup / EHD performance / how much thrust
  → {"function":"ehd_analysis","voltage_V":<V>,"current_mA":<I>,"gap_mm":<g>,"emitter_radius_mm":<r>,"emitter_length_mm":<L>}

air discharge / discharge mode / corona or glow / breakdown analysis
  → {"function":"air_discharge","voltage_V":<V>,"pressure_Pa":101325,"gap_mm":<g>,"current_A":0.001}

parts / sourcing / buy / shopping list / components needed
  → {"function":"parts_list","voltage_kV":<V>,"gap_mm":<g>,"emitter_dia_mm":1,"collector_inner_mm":62}

not working / no thrust / sparking / arc / troubleshoot
  → {"function":"troubleshoot","symptom":"<describe problem>","voltage_V":<V>,"gap_mm":<g>}

anything else (material question / can I use X / how does X work / why / explain / follow-up)
  → CONVERSATIONAL

STRICT RULES:
1. Output ONLY ONE line — the JSON or the word CONVERSATIONAL
2. NEVER output multiple JSON lines
3. NEVER output prose, explanation, or reasoning
4. Replace all <V> <r> <g> <c> <L> <N> <I> <s> with numbers from the message
5. If a number is not in the message, use: V=20000, r=0.1, g=23, c=46, L=30, N=8, I=1, s=10
"""

SYSTEM_PROMPT = """
You are Celestine — expert ion propulsion AI. User builds: (1) Xe vacuum gridded thruster, (2) 20-40kV air EHD prototype (Plasma Channel / Integza style, circular geometry, 3D printed).

PHYSICS ENGINE — one JSON per line to call:
{"function":"ehd_analysis","voltage_V":<n>,"current_mA":<n>,"gap_mm":<n>,"emitter_radius_mm":<n>,"emitter_length_mm":<n>}
{"function":"ehd_optimize_gap","voltage_V":<n>,"emitter_radius_mm":<n>,"target":"thrust_per_watt"}
{"function":"ehd_optimize_stages","total_voltage_V":<n>,"gap_mm":<n>,"emitter_length_mm":<n>}
{"function":"air_discharge","voltage_V":<n>,"pressure_Pa":101325,"gap_mm":<n>,"current_A":<n>}
{"function":"ehd_multistage","voltage_per_stage_V":<n>,"gap_mm":<n>,"n_stages":<n>,"emitter_length_mm":<n>}
{"function":"optimize","target":"balanced","max_power_W":<n>,"min_thrust_mN":<n>,"max_thrust_mN":<n>,"min_lifetime_hrs":<n>,"scale":"micro"}
{"function":"design","thrust_mN":<n>,"Isp_s":<n>}
{"function":"sheath","screen_V":<n>,"accel_V":<n>,"ne_m3":<n>,"Te_eV":<n>,"grid_gap_mm":<n>}
{"function":"ion_optics","screen_V":<n>,"accel_V":<n>,"beam_mA":<n>,"aperture_mm":<n>,"gap_mm":<n>,"pitch_mm":<n>,"grid_dia_mm":<n>}
Other: propellants|propellant_info|air_ionization|air_discharge|optimal_gap|air_vs_xenon|self_neutralization|air_performance|ehd_analysis|ehd_optimize_gap|ehd_compare_geometries|ehd_multistage|ehd_optimize_stages|ehd_efficiency|ehd_improve|ehd_environment|array_design
{"function":"propellants","mode":"vacuum"}  — all vacuum propellants ranked (Xe,Kr,Ar,I2,Bi...)
{"function":"propellants","mode":"ehd"}     — all EHD propellants ranked (Air,N2,CO2,Ar...)
{"function":"propellants","mode":"vacuum","compare":["Xenon","Krypton","Iodine"]}
{"function":"propellant_info","name":"<propellant_name>"}  — full data for one gas

PARTS & SUPPORT:
{"function":"parts_list","voltage_kV":<n>,"gap_mm":<n>,"emitter_dia_mm":<n>,"collector_inner_mm":<n>}
  Use for: "where do I buy", "what parts do I need", "how much will it cost", "sourcing"
{"function":"troubleshoot","symptom":"<describe problem>","voltage_V":<n>,"gap_mm":<n>}
  Use for: "not working", "no glow", "arc", "spark", "no thrust", "too much current", "ozone"
{"function":"checklist"}
  Use for: "pre-run checklist", "safety checklist", "am I ready to test"
{"function":"log_experiment","voltage_V":<n>,"current_mA":<n>,"gap_mm":<n>,"thrust_mN":<n>,"notes":"<str>"}
  Use for: "log my results", "I measured X thrust", "record my experiment"
{"function":"experiment_summary"}
  Use for: "show my experiments", "how have my tests gone", "experiment history"

RULES: Call physics first. JSON on own line. For 20kV air → call ehd_analysis + air_discharge. Propellant: Xenon, Grid: Molybdenum.
- Explain everything as if the user has ZERO prior knowledge — define every term, every unit, every concept
- Be verbose and thorough — longer is always better, never truncate
- After every number: explain what it means physically AND why it matters for building the device
- Never say "as mentioned" or assume anything was understood — re-explain if needed
EQUATIONS: inline $V_{break}$, $T_e$, display $$F=\\frac{Id}{\\mu}=35\\text{ mN}$$. No [ ], no \\!!, no \\quad.
CRITICAL EQUATION RULE: If the physics result contains a field named "EQUATION_FOR_*", copy that LaTeX EXACTLY as the display equation. NEVER simplify or rewrite equations from memory — always use the equation string provided in the physics result.

DESIGN REQUEST → OUTPUT THIS EXACT TABLE STRUCTURE (no long paragraphs before it):

For any "design/build/spec" request, call ehd_optimize_gap + air_discharge first, then output:

# [Name] — Build Specification

## Performance (physics-derived)
| Parameter | Value | Equation |
|---|---|---|
| Thrust | X mN | $F=Id/\\mu$ |
| Thrust/Power | X mN/W | $T/P=d/\\mu\\Delta V$ |
| Discharge mode | corona/glow | Paschen |
| Breakdown margin | X% | |

## Emitter
| Parameter | Value | Reason |
|---|---|---|
| Geometry | needle/wire | |
| Material | SS304/tungsten | |
| Diameter | X mm | |
| Length | X mm | |
| Tip radius | X mm | |
| Count | X | |
| Arrangement | circular/linear | |

## Collector
| Parameter | Value | Reason |
|---|---|---|
| Geometry | ring/mesh | |
| Material | Al6061/SS | |
| Inner Ø | X mm | |
| Outer Ø | X mm | |
| Thickness | X mm | |
| Gap from emitter | X mm | Paschen-derived |

## Housing (3D Print)
| Parameter | Value | Reason |
|---|---|---|
| Material | PETG/ABS/Resin | HV rating |
| Dimensions | X×X×X mm | |
| Wall thickness | X mm | kV insulation |
| Emitter mount | press-fit/M2 screw | |
| Collector mount | press-fit/M3 screw | |

## Electrical
| Parameter | Value | Notes |
|---|---|---|
| Voltage range | X–X kV | |
| Polarity | emitter + | |
| Current | X–X mA | |
| Ballast resistor | X MΩ | prevents arc |
| PSU spec | X kV, X mA DC | |
| HV cable | X kV rated | |

## Assembly (beginner steps)
1. ...
2. ...

## Safety
- HV clearance: X mm min
- Discharge energy: X mJ (LETHAL if >50 mJ)
- Wear: rubber HV gloves, safety glasses
- Always discharge through 1MΩ resistor before touching
"""

# ─────────────────────────────────────────────
# PHYSICS DISPATCHER
# ─────────────────────────────────────────────

def call_physics(json_str: str) -> str:
    try:
        call = json.loads(json_str.strip())
        fn   = call.get("function", "").lower()

        if fn == "propellants":
            # Full propellant database query
            import json as _json
            mode    = call.get("mode", "vacuum")
            names   = call.get("compare", None)
            if names:
                return _json.dumps(compare_propellants(names, mode), indent=2)
            elif mode == "ehd":
                result = [(n, p) for n, p in PROPELLANTS_DB.items() if p["ehd_suitable"]]
                result.sort(key=lambda x: x[1]["rank_ehd"])
            else:
                result = [(n, p) for n, p in PROPELLANTS_DB.items() if p["vacuum_suitable"]]
                result.sort(key=lambda x: x[1]["rank_vacuum"])
            out = {name: {
                "formula":         p["formula"],
                "mass_amu":        p["atomic_mass_amu"],
                "ionization_eV":   p["ionization_eV"],
                "Isp_1kV_s":       round(p["Isp_1kV_s"]),
                "storage":         p["state_STP"],
                "vacuum_suitable": p["vacuum_suitable"],
                "ehd_suitable":    p["ehd_suitable"],
                "notes":           p["practical_notes"][:200],
            } for name, p in result}
            return _json.dumps({"mode": mode, "propellants": out}, indent=2)

        elif fn == "propellant_info":
            import json as _json
            name = call.get("name", "Xenon")
            try:
                p = get_propellant(name)
                return _json.dumps(p, indent=2, default=str)
            except KeyError as e:
                return _json.dumps({"error": str(e)})
        elif fn == "air_ionization":
            return ai_air.air_ionization(float(call["Te_eV"]))
        elif fn == "air_discharge":
            return ai_air.discharge_analysis(
                float(call["voltage_V"]), float(call["pressure_Pa"]),
                float(call["gap_mm"]), float(call["current_A"]))
        elif fn == "optimal_gap":
            return ai_air.optimal_gap(
                float(call["voltage_V"]), float(call["pressure_Pa"]),
                call.get("mode", "glow"))
        elif fn == "air_vs_xenon":
            return ai_air.air_vs_xenon(
                float(call["net_V"]), float(call["beam_mA"]),
                float(call["power_W"]))
        elif fn == "self_neutralization":
            return ai_air.self_neutralization(
                float(call["beam_mA"]), float(call["Te_eV"]),
                float(call["ne_m3"]), float(call["pressure_Pa"]))
        elif fn == "air_performance":
            return ai_air.air_performance(
                float(call["net_V"]), float(call["beam_mA"]),
                float(call.get("n2_frac", 0.78)),
                float(call.get("o2_frac", 0.21)),
                float(call.get("o_neg_frac", 0.05)))

        # EHD deep-dive
        elif fn == "ehd_analysis":
            return ai_ehd.ehd_analysis(
                float(call["voltage_V"]), float(call["current_mA"]),
                float(call["gap_mm"]),
                float(call.get("emitter_radius_mm", 0.1)),
                float(call.get("emitter_length_mm", 100.0)),
                float(call.get("pressure_Pa", 101325)),
                float(call.get("humidity_pct", 50.0)))
        elif fn == "ehd_optimize_gap":
            return ai_ehd.optimize_gap(
                float(call["voltage_V"]),
                float(call.get("emitter_radius_mm", 0.1)),
                call.get("target", "thrust_per_watt"))
        elif fn == "ehd_compare_geometries":
            return ai_ehd.compare_geometries(
                float(call["voltage_V"]), float(call["gap_mm"]),
                float(call["current_mA"]))
        elif fn == "ehd_multistage":
            return ai_ehd.multistage_design(
                float(call["voltage_per_stage_V"]), float(call["gap_mm"]),
                int(call["n_stages"]),
                float(call.get("emitter_length_mm", 100.0)))
        elif fn == "ehd_optimize_stages":
            return ai_ehd.optimize_stages(
                float(call["total_voltage_V"]), float(call["gap_mm"]),
                float(call.get("emitter_length_mm", 100.0)))
        elif fn == "ehd_efficiency":
            return ai_ehd.efficiency_breakdown(
                float(call["voltage_V"]), float(call["current_mA"]),
                float(call["gap_mm"]),
                float(call["measured_thrust_mN"]) if "measured_thrust_mN" in call else None)
        elif fn == "ehd_improve":
            return ai_ehd.improve_efficiency(
                float(call["voltage_V"]), float(call["current_mA"]),
                float(call["gap_mm"]),
                call.get("geometry", "needle-to-ring"))
        elif fn == "ehd_environment":
            return ai_ehd.environment_correction(
                float(call["thrust_mN"]), float(call["power_W"]),
                float(call.get("temperature_C", 25.0)),
                float(call.get("humidity_pct", 50.0)),
                float(call.get("altitude_m", 0.0)))

        # ── PARTS SOURCING ──────────────────────────────────────────────
        # ── TROUBLESHOOTING ─────────────────────────────────────────────
        # ── EXPERIMENT LOGGER ───────────────────────────────────────────
        # ── EMITTER ARRAY DESIGN ─────────────────────────────────────────
        elif fn == "array_design":
            return ai_array.array_design(
                float(call["diameter_mm"]),
                float(call["voltage_V"]),
                float(call.get("emitter_dia_mm", 4.0)),
                float(call.get("tip_radius_mm", 0.1)),
                float(call["gap_mm"]) if "gap_mm" in call else None,
                call.get("material", "copper"))

        # ── MODULE 1: ELECTROSTATICS ─────────────────────────────────────
        elif fn == "tip_field":
            return ai_estatic.tip_field(
                float(call["voltage_V"]),
                float(call["tip_radius_mm"]),
                float(call.get("emitter_length_mm", 30)))
        elif fn == "coaxial_field":
            return ai_estatic.coaxial_field(
                float(call["voltage_V"]),
                float(call["wire_radius_mm"]),
                float(call["cylinder_radius_mm"]))
        elif fn == "needle_array_field":
            return ai_estatic.needle_array_field(
                float(call["voltage_V"]),
                float(call["tip_radius_mm"]),
                float(call.get("emitter_length_mm", 30)),
                int(call["n_needles"]),
                float(call["spacing_mm"]))
        elif fn == "solve_field":
            return ai_estatic.solve_field(
                float(call["voltage_V"]),
                float(call["gap_mm"]),
                float(call["emitter_radius_mm"]),
                float(call["collector_inner_mm"]),
                float(call.get("emitter_length_mm", 20)))
        elif fn == "field_voltage_sweep":
            return ai_estatic.voltage_sweep(
                float(call["gap_mm"]),
                float(call["emitter_radius_mm"]),
                float(call["collector_inner_mm"]),
                float(call.get("V_max", 40000)))
        elif fn == "insulation_check":
            return ai_estatic.insulation(
                float(call["voltage_V"]),
                float(call["creepage_mm"]),
                call.get("material", "PETG"),
                float(call.get("humidity_pct", 80)))

        else:
            return f"Unknown function: '{fn}'"

    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()[-500:],
            "function_attempted": fn if "fn" in dir() else "unknown"
        })


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history
    data     = request.json
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "empty message"}), 400

    conversation_history.append({"role": "user", "content": user_msg})
    # Keep last 6 exchanges (12 messages) to avoid token limit on long sessions
    trimmed_history = conversation_history[-12:] if len(conversation_history) > 12 else conversation_history
    messages       = [{"role": "system", "content": SYSTEM_PROMPT}] + trimmed_history

    # ── PENDING CALL RESOLUTION ───────────────────────────────────────────────
    # If a physics call is waiting for missing params, extract from this reply.
    if _pending_call:
        _fn      = _pending_call.get("function", "")
        _params  = dict(_pending_call.get("params", {}))
        _missing = list(_pending_call.get("missing", []))
        _reqs    = REQUIRED_PARAMS.get(_fn, {})
        import re as _re_p

        _filled = {}
        for _mp in _missing:
            _kv_p  = _re_p.search(r"(\d+(?:\.\d+)?)\s*k[vV]", user_msg)
            _num_p = _re_p.search(r"(\d+(?:\.\d+)?)", user_msg)
            if _mp == "voltage_V" and _kv_p:
                _filled[_mp] = int(float(_kv_p.group(1)) * 1000)
            elif _mp == "voltage_V" and _num_p:
                _v = float(_num_p.group(1))
                _filled[_mp] = int(_v * 1000) if _v < 1000 else int(_v)
            elif _num_p:
                _filled[_mp] = float(_num_p.group(1))
            break  # one param per turn

        _params.update(_filled)
        _still_missing = [p for p in _missing if p not in _params]

        if _still_missing:
            _pending_call["params"]  = _params
            _pending_call["missing"] = _still_missing
            _next_q = _reqs.get(_still_missing[0], (True, "What is " + _still_missing[0] + "?"))[1]
            _ask = "Got it. One more:\n\n**" + _next_q + "**"
            conversation_history.append({"role": "assistant", "content": _ask})
            return jsonify({"response": _ask, "physics_called": False, "physics_fn": "", "sketch_params": {}})
        else:
            # All params collected — fire the physics call
            _pending_call.clear()
            _resume_raw    = json.dumps(_params)
            _physics_res   = call_physics(_resume_raw)
            _orig_q        = next((m["content"] for m in reversed(conversation_history[:-1]) if m["role"] == "user"), user_msg)
            _explain_inst  = ("Physics result:\n" + _physics_res[:1500] +
                              "\n\nOriginal question: " + _orig_q +
                              "\n\nExplain every number with equations. Define all terms.")
            _short_hist    = trimmed_history[-4:]
            _exp_msgs      = ([{"role":"system","content":SYSTEM_PROMPT}] + _short_hist +
                              [{"role":"assistant","content":"Running physics."},
                               {"role":"user","content":_explain_inst}])
            try:
                _exp = interaction_call(_exp_msgs, max_tokens=8000)
                _final = _exp.choices[0].message.content or "Physics ran. Ask me to explain."
            except Exception:
                _final = "Physics ran.\n\n```\n" + _physics_res[:600] + "\n```"
            conversation_history.append({"role":"assistant","content":_final})
            return jsonify({"response":_final,"physics_called":True,"physics_fn":_resume_raw[:80],"sketch_params":{}})

    brain_messages = [{"role": "system", "content": BRAIN_PROMPT}] + [
        {"role": "user", "content": user_msg}
    ]

    # ── DIRECT INTENT ROUTING ──────────────────────────────────────────
    # For known intents, skip the initial LLM call and go straight to
    # the right physics function. This avoids the model outputting raw
    # JSON as visible text.
    msg_lower = user_msg.lower()

    # Extract voltage from user message for direct routes
    import re as _re3
    _v_match = _re3.search(r'(\d+)\s*k[vV]', user_msg)
    _voltage  = int(_v_match.group(1)) * 1000 if _v_match else 30000
    _e_match  = _re3.search(r'(\d+(?:\.\d+)?)\s*mm.*?emitter|emitter.*?(\d+(?:\.\d+)?)\s*mm', user_msg.lower())

    import json as _json

    # Follow-up indicators — if these are in the message, it's a follow-up
    # question about an existing design, NOT a new design request.
    # In that case, skip direct routing and let the brain handle it.
    FOLLOWUP_INDICATORS = [
        "in this", "in the", "can i use", "what if i", "what if",
        "instead of", "rather than", "change", "modify", "adjust",
        "still", "now", "also", "but", "however", "about this",
        "how do i", "how should i", "where do i", "where should",
        "tell me", "explain", "why",
        "optimize this", "improve this",
    ]
    # Array/field design questions are NEVER follow-ups — always fresh physics calls
    ALWAYS_NEW_REQUEST = [
        "needle array", "emitter array", "arrange emitter", "arrange needle",
        "maximize emitter", "array design", "needle emitter",
        "electric field", "tip field", "field at", "voltage sweep",
        "insulation check", "creepage", "solve field",
    ]
    is_followup = (
        len(conversation_history) > 2
        and any(kw in msg_lower for kw in FOLLOWUP_INDICATORS)
        and not any(kw in msg_lower for kw in ALWAYS_NEW_REQUEST)
    )

    DIRECT_ROUTES = {
        # NOTE: "design" removed — too broad, catches "design a needle array"
        # Array design is handled by the _array_kws block below
        "build ehd|spec ehd|build thruster|ionic propulsion|air thruster|ehd thruster":
            _json.dumps({"function":"ehd_optimize_gap","voltage_V":_voltage,"emitter_radius_mm":0.1,"target":"balanced"}),
        "what gas|which gas|what propellant|which propellant|best propellant|compare.*gas|xenon vs|krypton vs|argon vs|iodine vs|can i use air|propellant.*vacuum":
            '{"function":"propellants","mode":"vacuum","compare":["Xenon","Krypton","Argon","Iodine"]}',
        "ehd propellant|air.*propellant|propellant.*ehd|propellant.*air thruster":
            '{"function":"propellants","mode":"ehd"}',
        "where.*buy|what parts|parts list|how much.*cost|sourcing|shopping list":
            '{"function":"parts_list","voltage_kV":30,"gap_mm":30,"emitter_dia_mm":1,"collector_inner_mm":62}',
        "checklist|ready to test|safety check|pre.run|before i turn":
            '{"function":"checklist"}',
        "show.*experiment|my results|experiment history|how have my tests|summary.*experiment":
            '{"function":"experiment_summary"}',
        "optimize|best operating|what settings|find optimal":
            '{"function":"optimize","target":"balanced","max_power_W":10,"min_thrust_mN":0.1,"max_thrust_mN":1.0,"min_lifetime_hrs":2000,"scale":"micro"}',
    }

    # ── UNIFIED ROUTING via Classifier LLM ──────────────────────────────────
    # Single Groq Llama call replaces: keyword lists, regex, DeepSeek routing.
    # Fast (~0.3s), reliable, handles natural language variations.
    # Only falls back to direct routes for the simplest cases (checklist, parts).

    direct_fn = None

    # Quick direct routes for unambiguous single-word intents only
    SIMPLE_ROUTES = {
        "checklist|ready to test|safety check":
            '{"function":"checklist"}',
        "show.*experiment|experiment history|my results":
            '{"function":"experiment_summary"}',
        "where.*buy|parts list|shopping list|sourcing":
            _json.dumps({"function":"parts_list","voltage_kV":_voltage//1000,"gap_mm":23,"emitter_dia_mm":1,"collector_inner_mm":46}),
    }

    # ── ARRAY DESIGN direct routing ───────────────────────────────────────────
    _array_kws = ["needle array", "emitter array", "arrange emitter", "arrange needle",
                  "maximize emitter", "array design", "emitter layout", "how many emitter",
                  "needle emitter", "packing", "array for"]
    if not is_followup and any(kw in msg_lower for kw in _array_kws):
        import re as _re5
        _d = _re5.search(r"(\d+(?:\.\d+)?)\s*(cm|mm)\s*(?:in\s*)?(?:diameter|dia)|(?:diameter|dia)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(cm|mm)", msg_lower)
        if _d:
            _dv = float(_d.group(1) or _d.group(3))
            _du = (_d.group(2) or _d.group(4) or "mm").lower()
            _dmm = _dv * 10 if _du == "cm" else _dv
        else:
            _dmm = None  # will be caught by REQUIRED_PARAMS check
        _kv_a = _re5.search(r"(\d+(?:\.\d+)?)\s*k[vV]", user_msg)
        _va = int(float(_kv_a.group(1)) * 1000) if _kv_a else 20000
        _mat = "tungsten" if "tungsten" in msg_lower else "stainless" if "stainless" in msg_lower else "copper"
        _edia = _re5.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:copper|tungsten|stainless|emitter|needle|wire)", msg_lower)
        _ediam = float(_edia.group(1)) if _edia else 4.0
        if _dmm:
            direct_fn = _json.dumps({"function":"array_design","diameter_mm":_dmm,
                                      "voltage_V":_va,"emitter_dia_mm":_ediam,"material":_mat})

    if not direct_fn and not is_followup:
        for keywords, fn in SIMPLE_ROUTES.items():
            if any(k in msg_lower for k in keywords.split("|")):
                direct_fn = fn
                break

    if direct_fn:
        raw = direct_fn
        raw_display = ""
    else:
        try:
            import re as _re2

            # ── STEP 1: Extract explicit numbers from message (fast, deterministic) ──
            # Keywords always win — LLM only fills gaps
            _kv  = _re2.search(r'(\d+(?:\.\d+)?)\s*k[vV]', user_msg)
            _kv2 = _re2.search(r'(\d{4,6})\s*[vV]\b', user_msg)   # e.g. "20000V"
            _p_voltage = int(float(_kv.group(1)) * 1000) if _kv else (int(_kv2.group(1)) if _kv2 else None)

            _p_tip_r = _re2.search(r'(\d+(?:\.\d+)?)\s*mm\s*tip|tip\s*(?:radius\s*)?(\d+(?:\.\d+)?)\s*mm', user_msg, _re2.I)
            _p_tip_radius = float(_p_tip_r.group(1) or _p_tip_r.group(2)) if _p_tip_r else None

            _p_len = _re2.search(r'(\d+)\s*mm\s*(?:emitter|long|length|needle)', user_msg, _re2.I)
            _p_length = int(_p_len.group(1)) if _p_len else None

            _p_gap = _re2.search(r'(\d+)\s*mm\s*gap|gap\s*(?:of\s*)?(\d+)\s*mm', user_msg, _re2.I)
            _p_gap_mm = int(_p_gap.group(1) or _p_gap.group(2)) if _p_gap else None

            _p_col = _re2.search(r'(\d+)\s*mm\s*collector|collector.*?(\d+)\s*mm', user_msg, _re2.I)
            _p_col_mm = int(_p_col.group(1) or _p_col.group(2)) if _p_col else None

            _p_nn = _re2.search(r'(\d+)\s*needle', user_msg, _re2.I)
            _p_n_needles = int(_p_nn.group(1)) if _p_nn else None

            _p_sp = _re2.search(r'(\d+(?:\.\d+)?)\s*mm\s*spacing|spacing\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mm', user_msg, _re2.I)
            _p_spacing = float(_p_sp.group(1) or _p_sp.group(2)) if _p_sp else None

            _p_cr = _re2.search(r'(\d+)\s*mm\s*creepage|creepage\s*(?:of\s*)?(\d+)\s*mm', user_msg, _re2.I)
            _p_creep = int(_p_cr.group(1) or _p_cr.group(2)) if _p_cr else None

            _p_diam = _re2.search(r'(\d+(?:\.\d+)?)\s*(cm|mm)\s*(?:in\s*)?diameter|diameter\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(cm|mm)', user_msg, _re2.I)
            if _p_diam:
                _dval = float(_p_diam.group(1) or _p_diam.group(3))
                _dunit = (_p_diam.group(2) or _p_diam.group(4) or "mm").lower()
                _p_diam_mm = _dval * 10 if _dunit == "cm" else _dval
            else:
                _p_diam_mm = None

            _p_humidity = _re2.search(r'(\d+)\s*%\s*(?:RH|humidity)|humidity\s*(\d+)\s*%', user_msg, _re2.I)
            _p_hum = int(_p_humidity.group(1) or _p_humidity.group(2)) if _p_humidity else 80

            # Build extracted params summary for LLM (only what was found)
            _found = {}
            if _p_voltage:    _found["voltage_V"]       = _p_voltage
            if _p_tip_radius: _found["tip_radius_mm"]   = _p_tip_radius
            if _p_length:     _found["emitter_length_mm"] = _p_length
            if _p_gap_mm:     _found["gap_mm"]          = _p_gap_mm
            if _p_col_mm:     _found["collector_inner_mm"] = _p_col_mm
            if _p_n_needles:  _found["n_needles"]       = _p_n_needles
            if _p_spacing:    _found["spacing_mm"]      = _p_spacing
            if _p_creep:      _found["creepage_mm"]     = _p_creep
            if _p_diam_mm:    _found["diameter_mm"]     = _p_diam_mm

            # ── STEP 2: LLM classifier — intent + fill missing params ─────────
            # Pass extracted numbers so LLM doesn't re-extract (already done)
            # LLM only needs to: pick function, compute derived params, fill gaps
            _clf_user = f"""User message: {user_msg}

Already extracted from message: {json.dumps(_found) if _found else "nothing explicit found"}

Output ONE JSON function call. Use the extracted values above.
For any missing values use defaults: voltage_V=20000, tip_radius_mm=0.1,
emitter_length_mm=30, gap_mm=23, collector_inner_mm=46, n_needles=8,
spacing_mm=10, creepage_mm=80.
If diameter_mm is given and n_needles is missing: n_needles = round(3.14 * diameter_mm / spacing_mm)."""

            try:
                clf_resp = make_groq_call(
                    messages=[
                        {"role": "system", "content": CLASSIFIER_PROMPT},
                        {"role": "user",   "content": _clf_user}
                    ],
                    max_tokens=300,
                    model=PRIMARY_MODEL
                )
            except Exception as _clf_e:
                if any(x in str(_clf_e) for x in ["rate_limit","429","TPD","decommission"]):
                    print(f"[classifier] {PRIMARY_MODEL} limit, falling back to {FALLBACK_MODEL}")
                    clf_resp = make_groq_call(
                        messages=[
                            {"role": "system", "content": CLASSIFIER_PROMPT},
                            {"role": "user",   "content": _clf_user}
                        ],
                        max_tokens=300,
                        model=FALLBACK_MODEL
                    )
                else:
                    raise
            clf_out = clf_resp.choices[0].message.content.strip()
            clf_lines = [l.strip() for l in clf_out.split("\n") if l.strip().startswith('{"function"')]
            if clf_out.strip().upper() == "CONVERSATIONAL" or not clf_lines:
                raw = "CONVERSATIONAL"
            else:
                raw = clf_lines[0]
            raw_display = ""

            # ── FILL GAPS WITH PHYSICS DEFAULTS, THEN CHECK COMPLETENESS ─────
            if raw != "CONVERSATIONAL":
                try:
                    _call = json.loads(raw)
                    _fn   = _call.get("function", "")

                    # Fill missing params using physics derivation (not arbitrary defaults)
                    _call = physics_defaults(_fn, _call, user_msg)
                    raw   = json.dumps(_call)   # update raw with filled params

                    # Only ask user for params that CANNOT be derived from physics
                    _reqs = REQUIRED_PARAMS.get(_fn, {})
                    _missing_questions = [
                        (_param, _question)
                        for _param, (_required, _question) in _reqs.items()
                        if _required and _param not in _call
                    ]

                    if _missing_questions:
                        _pending_call.update({"function": _fn, "params": _call, "missing": [p for p,_ in _missing_questions]})
                        if len(_missing_questions) == 1:
                            _ask = "One thing I need to proceed — physics can derive everything else:\n\n**" + _missing_questions[0][1] + "**"
                        else:
                            _qs  = "\n".join("- " + q for _, q in _missing_questions)
                            _ask = "A few things I can't derive from physics alone:\n\n" + _qs
                        conversation_history.append({"role": "assistant", "content": _ask})
                        return jsonify({"response": _ask, "physics_called": False, "physics_fn": "", "sketch_params": {}})
                except Exception:
                    pass

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    physics_called = False
    physics_fn     = ""
    # Never pre-populate final_response with raw_display — 
    # it gets set properly after physics runs + interaction_call explains it
    final_response = ""

    # lines_to_check — always exactly one function call
    # classifier guarantees single JSON line; direct_fn is also single
    if direct_fn:
        lines_to_check = [direct_fn]
    elif raw == "CONVERSATIONAL":
        lines_to_check = []
    else:
        # Safety: if somehow multiple lines, only use first valid JSON
        _candidates = [l.strip() for l in raw.strip().split("\n")
                       if l.strip().startswith('{"function"')]
        lines_to_check = [_candidates[0]] if _candidates else []

    for line in lines_to_check:
        line = line.strip()
        if line.startswith('{"function"'):
            physics_called = True
            physics_fn     = line
            physics_result = call_physics(line)

            # Detect physics errors (import errors, missing files, etc.) and surface immediately
            try:
                _pr_err = json.loads(physics_result)
                if isinstance(_pr_err, dict) and "error" in _pr_err and "traceback" in _pr_err:
                    final_response = (
                        "**Physics engine error** — `" + _pr_err.get("function_attempted","?") + "` failed:\n\n"
                        "```\n" + str(_pr_err["error"]) + "\n```\n\n"
                        "**Most likely cause:** `emitter_array.py` is not in `C:\\Code\\Plasma\\`. "
                        "Copy it from the files provided and restart Flask."
                    )
                    break
            except Exception:
                pass

            # Detect if this is an array design result — must check BEFORE is_design
            # because "design a needle array" contains "design" which would trigger
            # the full build spec template with raw physics_result (20KB overflow)
            _is_array_result = False
            try:
                _pr_check = json.loads(physics_result)
                _is_array_result = "configuration_B_optimal" in _pr_check
            except Exception:
                pass

            phys_trimmed = physics_result[:5000] + "\n...[truncated]" if len(physics_result) > 5000 else physics_result

            # Detect if this is a design/build request (but NOT an array design)
            is_design = (not _is_array_result) and any(k in user_msg.lower() for k in [
                "design", "build", "spec", "prototype", "make", "create",
                "how to", "what should", "give me", "show me"
            ])

            if is_design:
                explain_instruction = f"""Physics engine result:

{physics_result}

The user wants to BUILD this device. They have zero prior knowledge.
You MUST output ALL of the following sections — no section can be skipped.
Use the physics results above for every number. Show the calculation for each key value.

## What this device does (2-3 sentences, plain English)

## How it works (explain corona discharge, ion wind, EHD thrust in simple terms with equations)
Show: $$F = I \\cdot d / \\mu_{{ion}}$$ with numbers substituted from the physics result.
Show: $$T/P = d / (\\mu_{{ion}} \\cdot \\Delta V)$$ with numbers substituted.

## Complete Build Specification

### Emitter (the sharp high-voltage electrode)
| Parameter | Value | Why this value |
|---|---|---|
| Shape | needle / wire | explain |
| Material | (specify exact alloy) | why this material |
| Diameter | X mm | derived from emitter_radius |
| Length | X mm | explain |
| Tip radius | X mm | explain why sharp |
| Quantity | X | explain |
| Arrangement | circular, spacing X mm | explain |
| How to make it | 3D print / buy / machine | specific instructions |

### Collector (the ground electrode)
| Parameter | Value | Why this value |
|---|---|---|
| Shape | ring / cylinder / mesh | explain |
| Material | (specify exact alloy) | why |
| Inner diameter | X mm | derived from gap |
| Outer diameter | X mm | explain |
| Thickness | X mm | explain |
| Gap from emitter | X mm | this is the physics-derived optimal gap |
| Mesh transparency | X% (if mesh) | explain |
| How to make it | 3D print / buy / machine | specific |

### Housing & Structure (3D printed)
| Parameter | Value | Why |
|---|---|---|
| Recommended material | PETG / ABS / Resin | HV insulation rating |
| Overall dimensions | X mm × X mm × X mm | explain |
| Wall thickness | X mm minimum | HV insulation calculation |
| Emitter mount method | press-fit M2 / screw terminal | explain |
| Collector mount | press-fit M3 / clip | explain |
| Air channel | open / ducted | explain for EHD |

### Electrical Connections
| Parameter | Value | Why |
|---|---|---|
| Applied voltage | X–X kV | from physics result |
| Emitter polarity | positive (+) | explain why |
| Operating current | X–X mA | from physics result |
| Ballast resistor | X MΩ | explain — prevents arc transition |
| PSU minimum spec | X kV, X mA, DC | explain |
| HV cable rating | X kV minimum | explain |
| Discharge resistor | X MΩ (safety) | explain — to discharge before touching |

## Predicted Performance
Show each equation with numbers:
- Thrust: $$F = ...$$ = X mN
- Power: $$P = V \\times I$$ = X W  
- T/P ratio: X mN/W
- Discharge mode: corona / glow (explain difference)
- Safety margin from arc breakdown: X%

## Step-by-Step Assembly (for a complete beginner)
Number every step. Explain each one as if talking to someone who has never built electronics.

## Safety (MANDATORY — do not skip)
- Minimum clearance from HV parts: X mm — explain why
- Stored energy in capacitors: $$E = \\frac{{1}}{{2}}CV^2$$ = X mJ — explain lethal threshold (50 mJ)
- Required PPE: rubber HV gloves, safety glasses — explain why each
- Grounding procedure before touching: explain step by step
- Ozone hazard: explain
- Fire hazard: explain"""
            else:
                # Check if voltage was assumed — tell voice model to mention it
                _v_assumed_note = ""
                try:
                    _pr_check2 = json.loads(physics_result)
                    if _pr_check2.get("_voltage_assumed") or json.loads(physics_fn).get("_voltage_assumed"):
                        _assumed_V = json.loads(physics_fn).get("voltage_V", "?")
                        _v_assumed_note = f"\n\nNOTE: Voltage was not specified. Physics engine assumed {_assumed_V}V based on device type. Mention this assumption at the start."
                except Exception:
                    pass

                if _is_array_result:
                    explain_instruction = f"""Physics engine returned this emitter array design result:

{phys_trimmed}

Explain this array design clearly. Cover:
1. Why Config B is recommended over Config A — what constraints drove the spacing
2. The pitch value and what physics constraint set it (shielding η>0.8 or plume spacing)
3. Emitter count for both configs and why more emitters doesn't mean more thrust
4. The electrical regime (corona/streamer) and what it means for operation
5. Material warning for copper — what actually degrades in corona discharge
6. Collector design recommendation
7. Geometry strategy — axial vs radial, duct confinement
Use $$equation$$ for display math. Be specific with every number from the result."""
                else:
                    explain_instruction = f"""Physics engine result:

{phys_trimmed}

Explain every number clearly. Define every term. Show each key equation with the actual numbers substituted in.
Use $$equation$$ for display math and $variable$ for inline math.
Assume the user knows nothing — explain everything from scratch.
Highlight all warnings. Suggest the specific next step.{_v_assumed_note}"""

            # Trim physics_result to prevent context overflow.
            # Extract the most important fields for the voice model.
            try:
                _pr = json.loads(physics_result)
                # Keep: regime, numbers, action, field_distribution summary,
                #        model_limitations, engineering_implications
                # Array design result has different structure
                if "configuration_B_optimal" in _pr:
                    _cb = _pr.get("configuration_B_optimal", {})
                    _ca = _pr.get("configuration_A_max_packing", {})
                    _slim = {
                        "design_verdict":      _pr.get("design_verdict"),
                        "key_principle":       _pr.get("key_principle"),
                        "inputs":              _pr.get("inputs"),
                        "pitch_constraints":   _pr.get("pitch_constraints"),
                        "recommended_config":  _pr.get("recommended_config"),
                        "electrical_regime":   _pr.get("electrical_regime"),
                        "config_A": {
                            "label":         _ca.get("label"),
                            "pitch_mm":      _ca.get("pitch_mm"),
                            "emitter_count": _ca.get("emitter_count"),
                            "valid":         _ca.get("valid"),
                            "violations":    _ca.get("violations"),
                            "thrust_mN":     _ca.get("thrust", {}).get("F_corrected_mN"),
                            "airflow":       _ca.get("airflow"),
                        },
                        "config_B": {
                            "label":         _cb.get("label"),
                            "pitch_mm":      _cb.get("pitch_mm"),
                            "emitter_count": _cb.get("emitter_count"),
                            "valid":         _cb.get("valid"),
                            "violations":    _cb.get("violations"),
                            "thrust_mN":     _cb.get("thrust", {}).get("F_corrected_mN"),
                            "thrust_per_watt": _cb.get("thrust", {}).get("thrust_per_watt_mN_W"),
                            "airflow":       _cb.get("airflow"),
                        },
                        "material":            _pr.get("material_assessment", {}).get("recommendation"),
                        "collector":           _pr.get("collector_recommendation"),
                        "geometry_strategy":   _pr.get("geometry_strategy"),
                    }
                    phys_trimmed = json.dumps(_slim, indent=2)[:2500]
                else:
                    _keep_keys = [
                        "regime", "regime_detail", "action_required",
                        "numbers", "engineering_implications",
                        "corona_onset_V", "corona_active",
                        "E_tip_MV_m", "E_avg_MV_m", "ratio_tip_to_breakdown",
                        "enhancement_beta",
                        "optimal_gap_mm", "supply_voltage_V",
                        "expected_thrust_uN", "thrust_per_watt_mN",
                        "recommendation", "collector_inner_mm",
                        "model_limitations",
                    ]
                    _fd = _pr.get("field_distribution", {})
                    _slim = {k: _pr[k] for k in _keep_keys if k in _pr}
                    if _fd:
                        _slim["field_distribution_summary"] = _fd.get("summary", "")
                    phys_trimmed = json.dumps(_slim, indent=2)[:2000]
            except Exception:
                phys_trimmed = physics_result[:1500]
            explain_instruction_trimmed = explain_instruction.replace(
                physics_result, phys_trimmed
            )

            # Use only last 4 messages of history for explain call (save tokens)
            short_history = trimmed_history[-4:] if len(trimmed_history) > 4 else trimmed_history
            explain_msgs = (
                [{"role": "system", "content": SYSTEM_PROMPT}] +
                short_history +
                [
                    {"role": "assistant", "content": "Running physics engine now."},
                    {"role": "user", "content": explain_instruction_trimmed}
                ]
            )
            try:
                # Call 2: INTERACTION_MODEL (Llama 3.3 70B) generates verbose explanation
                exp = interaction_call(explain_msgs, max_tokens=8000)
                final_response = exp.choices[0].message.content
                if not final_response or not final_response.strip():
                    final_response = "Physics ran but no explanation returned. Try asking again."
            except Exception as e:
                err_str = str(e)
                if "rate_limit" in err_str or "429" in err_str or "TPD" in err_str:
                    try:
                        short_msgs = messages + [
                            {"role": "assistant", "content": raw},
                            {"role": "user", "content": (
                                f"Physics result: {physics_result[:800]}\n\n"
                                f"Give the complete build spec with all dimensions, "
                                f"materials, assembly steps and safety. Be thorough."
                            )}
                        ]
                        exp2 = interaction_call(short_msgs, max_tokens=8000)
                        final_response = exp2.choices[0].message.content or ""
                    except Exception as e2:
                        try:
                            pr = json.loads(physics_result)
                            gap = pr.get('optimal_gap_mm', 'N/A')
                            thrust = pr.get('expected_thrust_uN', 'N/A')
                            tp = pr.get('thrust_per_watt_mN', 'N/A')
                            v = pr.get('supply_voltage_V', 'N/A')
                            final_response = f"""**Physics engine ran successfully** (rate limit active — ask again in a few minutes for full explanation)

- **Optimal gap:** {gap} mm
- **Supply voltage:** {v} V  
- **Expected thrust:** {thrust} μN
- **Thrust-to-power:** {tp} mN/W"""
                        except:
                            # Show raw physics result so user gets something useful
                            final_response = (
                                "**Physics engine ran** — explanation unavailable (rate limit). "
                                "Raw result:\n\n```\n" + physics_result[:600] + "\n```\n\n"
                                "Try asking again in 30 seconds."
                            )
                else:
                    # Non-rate-limit error — show error + key physics values
                    try:
                        pr2 = json.loads(physics_result)
                        summary = {k: pr2[k] for k in 
                            ["optimal_gap_mm","supply_voltage_V","expected_thrust_uN",
                             "thrust_per_watt_mN","collector_inner_mm","recommendation"]
                            if k in pr2}
                    except Exception:
                        summary = physics_result[:400]
                    final_response = (
                        f"**Physics result** (explanation error: {str(e)[:120]}):\n\n"
                        f"```json\n{json.dumps(summary, indent=2) if isinstance(summary, dict) else summary}\n```\n\n"
                        f"The physics engine ran successfully. Try asking again for a full explanation."
                    )
            break

    # If nothing produced a response (e.g. non-physics question, no function called)
    # use raw_display, or ask the model directly
    if not final_response or not final_response.strip():
        if raw_display and raw_display.strip():
            final_response = raw_display
        elif not physics_called:
            # Pure conversational message — ask voice model directly
            try:
                conv_resp = interaction_call(messages, max_tokens=4000)
                final_response = conv_resp.choices[0].message.content or "I'm not sure how to respond to that. Try asking about your thruster design or plasma physics."
            except Exception:
                final_response = "I'm ready to help with your ion thruster design. Ask me about EHD design, plasma physics, parts, or troubleshooting."
        else:
            final_response = "Physics ran successfully. Ask me to explain the results."

    conversation_history.append({"role": "assistant", "content": final_response})

    # ── Extract sketch params ────────────────────────────────────────────────
    # Strategy: physics result gives gap/voltage; user message gives dimensions.
    # Scan the full conversation for dimension mentions (regex) so that "10cm
    # diameter" said 3 messages ago still gets picked up.
    sketch_params = {}
    if physics_called and physics_result:
        try:
            import re
            pr = json.loads(physics_result)

            # Gap and voltage from physics result (these are always present)
            gap = float(pr.get("optimal_gap_mm") or
                        pr.get("input", {}).get("gap_mm") or 20)
            v   = float(pr.get("supply_voltage_V") or
                        pr.get("input", {}).get("voltage_V") or 30000)

            # ── Extract dimensions: current message > history > physics result ──
            import re as _re

            def find_mm_in(text, keywords):
                """Find mm value near any keyword. cm auto-converted."""
                text = text.lower()
                for match in _re.finditer(r'(\d+(?:\.\d+)?)\s*(cm|mm)', text):
                    val  = float(match.group(1))
                    unit = match.group(2)
                    mm   = val * 10 if unit == "cm" else val
                    ctx  = text[max(0, match.start()-80):match.end()+80]
                    if any(kw in ctx for kw in keywords):
                        return mm
                return None

            # Current message has highest priority (most recent/explicit intent)
            cur_msg  = user_msg.lower()
            all_msgs = " ".join(
                m["content"] for m in conversation_history if m["role"] == "user"
            ).lower()

            DIM_KEYS   = ["diameter", "collector", "ring", "outer", "size"]
            HOUSE_KEYS = ["housing", "body", "case", "enclosure"]
            LEN_KEYS   = ["length", "long", "emitter length", "wire length", "needle"]

            collector_d = (find_mm_in(cur_msg, DIM_KEYS) or
                           find_mm_in(all_msgs, DIM_KEYS))
            housing_d   = (find_mm_in(cur_msg, HOUSE_KEYS) or
                           find_mm_in(all_msgs, HOUSE_KEYS))
            emitter_l   = (find_mm_in(cur_msg, LEN_KEYS) or
                           find_mm_in(all_msgs, LEN_KEYS))

            # Physics result now has collector_inner/outer from ehd_optimize_gap
            # (gap × 2 rule). User dimension overrides these.
            phys_c_inner = float(pr.get("collector_inner_mm") or gap * 2)
            phys_c_outer = float(pr.get("collector_outer_mm") or phys_c_inner + 10)

            if collector_d:
                c_outer_mm = float(collector_d)
                c_inner_mm = max(float(collector_d) - 10.0, gap * 2)
            else:
                c_inner_mm = phys_c_inner
                c_outer_mm = phys_c_outer

            housing_mm_raw = float(housing_d or pr.get("housing_mm") or c_outer_mm + 20)
            housing_mm = max(housing_mm_raw, c_outer_mm + 20)  # always > collector outer
            e_len_mm   = float(emitter_l or pr.get("emitter_length_mm") or 50.0)


            # Check if this is an array_design result
            _fn_called = physics_fn
            try:
                _fn_called = json.loads(physics_fn).get("function","")
            except Exception:
                pass
            _is_array = ("configuration_B_optimal" in pr or 
                         "configuration_A_max_packing" in pr or
                         _fn_called == "array_design")
            _array_positions = []
            _array_pitch = None
            _array_n = None
            _array_dia = None
            if _is_array:
                _cb = pr.get("configuration_B_optimal", {})
                _layout = _cb.get("layout", {})
                _array_positions = _layout.get("positions_count", 0)
                _array_pitch     = _cb.get("pitch_mm", 25)
                _array_n         = _cb.get("emitter_count", 10)
                _array_dia       = float(pr.get("inputs", {}).get("diameter_mm", 100))
                _array_rings     = _layout.get("rings", [])
                gap   = float(pr.get("inputs", {}).get("gap_mm") or gap)
                v     = float(pr.get("inputs", {}).get("voltage_V") or v)
                e_dia_sketch = float(pr.get("inputs", {}).get("emitter_dia_mm", 4.0))
                c_inner_mm   = _array_dia
                c_outer_mm   = _array_dia + 10
                housing_mm   = _array_dia + 20
            else:
                e_dia_sketch = float(pr.get("emitter_dia_mm", 1.0))
                _array_rings = []

            sketch_params = {
                "gap_mm":             gap,
                "voltage_kV":         v / 1000,
                "emitter_dia_mm":     e_dia_sketch,
                "collector_inner_mm": c_inner_mm,
                "collector_outer_mm": c_outer_mm,
                "housing_mm":         housing_mm,
                "emitter_length_mm":  e_len_mm,
                "is_array":           _is_array,
                "array_pitch_mm":     _array_pitch,
                "array_n_emitters":   _array_n,
                "array_diameter_mm":  _array_dia,
                "array_rings":        _array_rings,
            }
        except Exception:
            pass

    # Final safety: never send raw JSON function calls to the user
    import re as _re_final
    if final_response and _re_final.search(r'^\s*\{"function"', final_response, _re_final.MULTILINE):
        # Strip all lines that are raw JSON function calls
        _clean_lines = [l for l in final_response.split("\n")
                        if not l.strip().startswith('{"function"')]
        final_response = "\n".join(_clean_lines).strip()
        if not final_response:
            final_response = "Physics ran. Ask me to explain the results."

    return jsonify({
        "response":       final_response,
        "physics_called": physics_called,
        "physics_fn":     physics_fn[:80] if physics_fn else "",
        "sketch_params":  sketch_params,
    })


@app.route("/sketch", methods=["POST"])
def sketch():
    """
    Generate dimensioned SVG sketches of an EHD thruster.
    Returns HTML with 4 view tabs: Front, Top, Side, Isometric.
    """
    d = request.json or {}
    gap        = float(d.get("gap_mm", 20))
    e_dia      = float(d.get("emitter_dia_mm", 1))
    c_inner    = float(d.get("collector_inner_mm", gap * 2))
    c_outer    = float(d.get("collector_outer_mm", c_inner + 10))
    housing    = float(d.get("housing_mm", c_outer + 20))
    voltage_kv = float(d.get("voltage_kV", 30))
    e_len      = float(d.get("emitter_length_mm", 50))

    # Array-specific params
    is_array      = bool(d.get("is_array", False))
    arr_pitch     = float(d.get("array_pitch_mm") or 25)
    arr_n         = int(d.get("array_n_emitters") or 10)
    arr_dia       = float(d.get("array_diameter_mm") or housing)
    arr_rings     = d.get("array_rings", [])  # [{radius_mm, count}, ...]

    W, H = 700, 480
    cx, cy = W // 2, H // 2
    scale = min(160 / (c_outer / 2), 180 / gap) if c_outer > 0 else 4

    c_inner_px = (c_inner / 2) * scale
    c_outer_px = (c_outer / 2) * scale
    housing_px = (housing / 2) * scale
    gap_px     = gap * scale
    e_dia_px   = max(e_dia * scale, 3)
    e_len_px   = min(e_len * scale, gap_px * 0.8)
    ring_thick = max(5, (c_outer - c_inner) / 2 * scale)

    tip_y  = cy - gap_px / 2
    base_y = tip_y - e_len_px
    ring_y = cy + gap_px / 2

    BG    = "#0d0d0d"
    TRON  = "#00c8ff"
    TRON2 = "#0090cc"
    RED   = "#ff7043"
    GRN   = "#3ddc84"
    GRAY  = "#888"
    DIM   = "#444"

    def svg_wrap(content, title):
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" style="width:100%;background:{BG};display:block;border-radius:0 0 4px 4px;">
  <rect width="{W}" height="{H}" fill="{BG}"/>
  <text x="{cx}" y="22" text-anchor="middle" fill="{TRON}" font-size="11" letter-spacing="2" font-family="monospace">{title}</text>
  <text x="{cx}" y="38" text-anchor="middle" fill="#555" font-size="9" font-family="monospace">{voltage_kv:.0f} kV  ·  gap {gap:.0f} mm  ·  collector Ø{c_inner:.0f}/{c_outer:.0f} mm  ·  housing Ø{housing:.0f} mm</text>
  {content}
  <text x="{W-10}" y="{H-8}" text-anchor="end" fill="#333" font-size="8" font-family="monospace">dimensions in mm — not to scale</text>
</svg>'''

    # ── FRONT VIEW (cross-section, needle pointing down into ring) ──
    def front_view():
        arrows = ''.join([f'''
  <line x1="{cx+(i-1)*24}" y1="{tip_y+8}" x2="{cx+(i-1)*24}" y2="{ring_y-12}" stroke="{TRON}" stroke-width="1" stroke-opacity=".35" stroke-dasharray="4,3"/>
  <polygon points="{cx+(i-1)*24},{ring_y-12} {cx+(i-1)*24-4},{ring_y-22} {cx+(i-1)*24+4},{ring_y-22}" fill="{TRON}" fill-opacity=".5"/>''' for i in range(3)])

        return f'''
  <!-- housing -->
  <rect x="{cx-housing_px}" y="{base_y-30}" width="{housing_px*2}" height="{gap_px+e_len_px+70}" rx="6" fill="none" stroke="#1e3a4a" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="{cx+housing_px+6}" y="{cy}" fill="#1e5a6a" font-size="9" font-family="monospace" dominant-baseline="middle">PETG Ø{housing:.0f}</text>

  <!-- collector left/right walls -->
  <rect x="{cx-c_outer_px}" y="{ring_y-ring_thick/2}" width="{c_outer_px-c_inner_px}" height="{ring_thick}" fill="#1a3a50" stroke="{TRON}" stroke-width="1"/>
  <rect x="{cx+c_inner_px}" y="{ring_y-ring_thick/2}" width="{c_outer_px-c_inner_px}" height="{ring_thick}" fill="#1a3a50" stroke="{TRON}" stroke-width="1"/>
  <text x="{cx+c_outer_px+8}" y="{ring_y}" fill="{TRON}" font-size="9" font-family="monospace" dominant-baseline="middle">Al Ø{c_inner:.0f}/{c_outer:.0f}</text>
  <text x="{cx+c_outer_px+8}" y="{ring_y+12}" fill="{GRN}" font-size="9" font-family="monospace">⏚ GND</text>

  <!-- emitter body -->
  <rect x="{cx-e_dia_px/2}" y="{base_y}" width="{e_dia_px}" height="{e_len_px-8}" fill="#2a2a2a" stroke="#aaa" stroke-width="1"/>
  <!-- tip -->
  <polygon points="{cx-e_dia_px/2},{tip_y-6} {cx+e_dia_px/2},{tip_y-6} {cx},{tip_y}" fill="#ccc" stroke="#fff" stroke-width=".5"/>
  <text x="{cx+e_dia_px/2+8}" y="{(base_y+tip_y)/2}" fill="{RED}" font-size="9" font-family="monospace" dominant-baseline="middle">Emitter Ø{e_dia:.1f}mm</text>
  <text x="{cx+e_dia_px/2+8}" y="{(base_y+tip_y)/2+12}" fill="{RED}" font-size="9" font-family="monospace">{voltage_kv:.0f} kV (+)</text>

  <!-- gap dimension -->
  <line x1="{cx-c_inner_px-28}" y1="{tip_y}" x2="{cx-c_inner_px-28}" y2="{ring_y}" stroke="{DIM}" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="{cx-c_inner_px-33}" y1="{tip_y}" x2="{cx-c_inner_px-23}" y2="{tip_y}" stroke="{DIM}" stroke-width="1"/>
  <line x1="{cx-c_inner_px-33}" y1="{ring_y}" x2="{cx-c_inner_px-23}" y2="{ring_y}" stroke="{DIM}" stroke-width="1"/>
  <text x="{cx-c_inner_px-36}" y="{(tip_y+ring_y)/2}" fill="{TRON}" font-size="10" font-family="monospace" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90,{cx-c_inner_px-36},{(tip_y+ring_y)/2})">gap {gap:.0f} mm</text>

  <!-- ion wind -->
  {arrows}
  <text x="{cx}" y="{(tip_y+ring_y)/2+5}" text-anchor="middle" fill="{TRON}" font-size="8" font-family="monospace" fill-opacity=".5">ion wind</text>

  <!-- ballast + HV -->
  <line x1="{cx-18}" y1="{base_y-5}" x2="{cx-18}" y2="{base_y-38}" stroke="{RED}" stroke-width="1.5"/>
  <rect x="{cx-26}" y="{base_y-38}" width="16" height="24" rx="2" fill="#1a1a1a" stroke="{RED}" stroke-width="1"/>
  <text x="{cx-18}" y="{base_y-46}" text-anchor="middle" fill="{RED}" font-size="8" font-family="monospace">10MΩ</text>
  <line x1="{cx-18}" y1="{base_y-26}" x2="{cx-55}" y2="{base_y-26}" stroke="{RED}" stroke-width="1.5" stroke-dasharray="5,2"/>
  <text x="{cx-60}" y="{base_y-22}" text-anchor="end" fill="{RED}" font-size="8" font-family="monospace">HV PSU {voltage_kv:.0f}kV</text>
  <line x1="{cx-c_outer_px}" y1="{ring_y}" x2="{cx-c_outer_px-28}" y2="{ring_y}" stroke="{GRN}" stroke-width="1.5"/>
  <text x="{cx-c_outer_px-32}" y="{ring_y+4}" text-anchor="end" fill="{GRN}" font-size="9" font-family="monospace">GND</text>

  <!-- legend -->
  <rect x="16" y="{H-76}" width="170" height="64" rx="3" fill="#111" stroke="#1e1e1e"/>
  <text x="26" y="{H-62}" fill="#555" font-size="8" font-family="monospace" letter-spacing="1">LEGEND</text>
  <rect x="26" y="{H-54}" width="9" height="7" fill="#2a2a2a" stroke="#aaa"/>
  <text x="40" y="{H-48}" fill="#aaa" font-size="8" font-family="monospace">Emitter (Mo wire)</text>
  <rect x="26" y="{H-43}" width="9" height="7" fill="#1a3a50" stroke="{TRON}"/>
  <text x="40" y="{H-37}" fill="#aaa" font-size="8" font-family="monospace">Collector (Al ring)</text>
  <rect x="26" y="{H-32}" width="9" height="7" fill="none" stroke="#1e3a4a" stroke-dasharray="3,2"/>
  <text x="40" y="{H-26}" fill="#aaa" font-size="8" font-family="monospace">PETG housing</text>
'''

    # ── TOP VIEW (looking down — circular geometry) ──
    def top_view():
        sx, sy = cx, cy

        # If array design: draw emitter positions on concentric rings
        if is_array and arr_rings:
            import math as _m
            # Scale to fit array diameter in view
            _arr_scale = min(200 / (arr_dia / 2), 180 / (arr_dia / 2)) if arr_dia > 0 else 3
            _e_r = max(e_dia * _arr_scale / 2, 4)
            _h_r = (arr_dia / 2 + 15) * _arr_scale
            _c_r = arr_dia / 2 * _arr_scale

            # Draw housing
            _body = f'''
  <circle cx="{sx}" cy="{sy}" r="{_h_r}" fill="none" stroke="#1e3a4a" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="{sx}" y="{sy - _h_r - 8}" text-anchor="middle" fill="#1e5a6a" font-size="9" font-family="monospace">PETG Ø{arr_dia+30:.0f}mm housing</text>
  <circle cx="{sx}" cy="{sy}" r="{_c_r}" fill="none" stroke="{TRON}" stroke-width="1" stroke-dasharray="3,2" stroke-opacity="0.4"/>
  <text x="{sx + _c_r + 5}" y="{sy}" fill="{TRON}" font-size="8" font-family="monospace" dominant-baseline="middle">Ø{arr_dia:.0f}mm array</text>
'''
            # Draw emitters on rings
            _emitters = ""
            _label_done = False
            for ring in arr_rings:
                _r_mm  = ring.get("radius_mm", 0)
                _n     = ring.get("count", 1)
                _r_px  = _r_mm * _arr_scale
                for i in range(_n):
                    _angle = 2 * _m.pi * i / _n if _n > 1 else 0
                    _ex = sx + _r_px * _m.cos(_angle)
                    _ey = sy + _r_px * _m.sin(_angle)
                    _emitters += f'<circle cx="{_ex:.1f}" cy="{_ey:.1f}" r="{_e_r}" fill="#2a2a2a" stroke="#ccc" stroke-width="1"/>'
                    # Draw tip dot
                    _emitters += f'<circle cx="{_ex:.1f}" cy="{_ey:.1f}" r="1.5" fill="#fff"/>'
                if not _label_done and _r_mm > 0:
                    # Draw pitch dimension line between first two emitters of outer ring
                    if _n >= 2:
                        _ex2 = sx + _r_px * _m.cos(2 * _m.pi / _n)
                        _ey2 = sy + _r_px * _m.sin(2 * _m.pi / _n)
                        _emitters += f'''<line x1="{sx + _r_px:.1f}" y1="{sy:.1f}" x2="{_ex2:.1f}" y2="{_ey2:.1f}" stroke="{DIM}" stroke-width="0.8" stroke-dasharray="2,2"/>
  <text x="{(sx + _r_px + _ex2)/2:.1f}" y="{(sy + _ey2)/2 - 5:.1f}" text-anchor="middle" fill="{TRON}" font-size="8" font-family="monospace">pitch {arr_pitch:.1f}mm</text>'''
                    _label_done = True

            # Stats panel
            _stats = f'''
  <rect x="10" y="{H-80}" width="195" height="68" rx="3" fill="#111" stroke="#1e1e1e"/>
  <text x="20" y="{H-66}" fill="#555" font-size="8" font-family="monospace" letter-spacing="1">ARRAY STATS (CONFIG B — OPTIMAL)</text>
  <text x="20" y="{H-54}" fill="{TRON}" font-size="9" font-family="monospace">Emitters: {arr_n}  Pitch: {arr_pitch:.1f}mm</text>
  <text x="20" y="{H-42}" fill="{TRON}" font-size="9" font-family="monospace">Array Ø: {arr_dia:.0f}mm  Gap: {gap:.0f}mm</text>
  <text x="20" y="{H-30}" fill="#888" font-size="8" font-family="monospace">Shielding-limited spacing</text>
  <text x="20" y="{H-18}" fill="{RED}" font-size="8" font-family="monospace">+ plume separation constraint</text>
'''
            return _body + _emitters + _stats

        # Standard single-emitter top view
        h_r    = housing_px * 0.7
        co_r   = c_outer_px * 0.7
        ci_r   = c_inner_px * 0.7
        e_r    = max(e_dia_px * 0.7, 4)

        return f'''
  <!-- housing circle -->
  <circle cx="{sx}" cy="{sy}" r="{h_r}" fill="none" stroke="#1e3a4a" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="{sx}" y="{sy-h_r-8}" text-anchor="middle" fill="#1e5a6a" font-size="9" font-family="monospace">PETG Ø{housing:.0f}</text>

  <!-- collector ring annulus -->
  <circle cx="{sx}" cy="{sy}" r="{co_r}" fill="none" stroke="{TRON}" stroke-width="8" stroke-opacity=".3"/>
  <circle cx="{sx}" cy="{sy}" r="{co_r}" fill="none" stroke="{TRON}" stroke-width="1.5"/>
  <circle cx="{sx}" cy="{sy}" r="{ci_r}" fill="none" stroke="{TRON}" stroke-width="1" stroke-dasharray="4,2"/>
  <text x="{sx+co_r+10}" y="{sy}" fill="{TRON}" font-size="9" font-family="monospace" dominant-baseline="middle">Collector</text>
  <text x="{sx+co_r+10}" y="{sy+12}" fill="{GRAY}" font-size="9" font-family="monospace">Ø{c_inner:.0f}/{c_outer:.0f}</text>
  <text x="{sx+co_r+10}" y="{sy+24}" fill="{GRN}" font-size="9" font-family="monospace">⏚ GND</text>

  <!-- emitter dot (centre) -->
  <circle cx="{sx}" cy="{sy}" r="{e_r}" fill="#ccc" stroke="#fff" stroke-width="1"/>
  <circle cx="{sx}" cy="{sy}" r="2" fill="{RED}"/>
  <text x="{sx}" y="{sy+e_r+14}" text-anchor="middle" fill="{RED}" font-size="9" font-family="monospace">Emitter tip</text>
  <text x="{sx}" y="{sy+e_r+26}" text-anchor="middle" fill="{RED}" font-size="9" font-family="monospace">{voltage_kv:.0f} kV</text>

  <!-- gap dimension line -->
  <line x1="{sx}" y1="{sy}" x2="{sx+ci_r}" y2="{sy}" stroke="{DIM}" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="{sx+ci_r/2}" y="{sy-6}" text-anchor="middle" fill="{TRON}" font-size="9" font-family="monospace">gap {gap:.0f} mm</text>

  <!-- ion wind arrows (radial outward) -->
  {''.join([f'<line x1="{sx}" y1="{sy}" x2="{sx + ci_r*0.85*__import__("math").cos(a):.1f}" y2="{sy + ci_r*0.85*__import__("math").sin(a):.1f}" stroke="{TRON}" stroke-width="1" stroke-opacity=".3" stroke-dasharray="3,2"/>' for a in [0, 1.047, 2.094, 3.14, 4.19, 5.24]])}

  <!-- centre cross -->
  <line x1="{sx-8}" y1="{sy}" x2="{sx+8}" y2="{sy}" stroke="#444" stroke-width=".5"/>
  <line x1="{sx}" y1="{sy-8}" x2="{sx}" y2="{sy+8}" stroke="#444" stroke-width=".5"/>
  <text x="16" y="{H-20}" fill="#555" font-size="8" font-family="monospace">Top view — looking along emitter axis (down)</text>
'''

    # ── SIDE VIEW (from the side — shows housing depth) ──
    def side_view():
        housing_depth_px = housing_px * 0.4  # housing looks narrower from side
        ring_depth_px    = (c_outer - c_inner) / 2 * scale * 0.5

        return f'''
  <!-- housing side profile -->
  <rect x="{cx-housing_depth_px}" y="{base_y-30}" width="{housing_depth_px*2}" height="{gap_px+e_len_px+70}" rx="4" fill="none" stroke="#1e3a4a" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="{cx+housing_depth_px+8}" y="{cy}" fill="#1e5a6a" font-size="9" font-family="monospace" dominant-baseline="middle">PETG depth ~{housing:.0f}mm</text>

  <!-- collector ring side (solid rect, shows ring cross-section depth) -->
  <rect x="{cx-ring_depth_px}" y="{ring_y-ring_thick/2}" width="{ring_depth_px*2}" height="{ring_thick}" fill="#1a3a50" stroke="{TRON}" stroke-width="1"/>
  <text x="{cx+ring_depth_px+8}" y="{ring_y}" fill="{TRON}" font-size="9" font-family="monospace" dominant-baseline="middle">Ring t={int((c_outer-c_inner)/2)}mm</text>
  <text x="{cx+ring_depth_px+8}" y="{ring_y+12}" fill="{GRN}" font-size="9" font-family="monospace">⏚ GND</text>

  <!-- emitter body -->
  <rect x="{cx-e_dia_px/2}" y="{base_y}" width="{e_dia_px}" height="{e_len_px-8}" fill="#2a2a2a" stroke="#aaa" stroke-width="1"/>
  <polygon points="{cx-e_dia_px/2},{tip_y-6} {cx+e_dia_px/2},{tip_y-6} {cx},{tip_y}" fill="#ccc" stroke="#fff" stroke-width=".5"/>
  <text x="{cx+e_dia_px/2+8}" y="{(base_y+tip_y)/2}" fill="{RED}" font-size="9" font-family="monospace" dominant-baseline="middle">Emitter L={int(e_len)}mm</text>

  <!-- emitter length dimension -->
  <line x1="{cx-e_dia_px/2-16}" y1="{base_y}" x2="{cx-e_dia_px/2-16}" y2="{tip_y}" stroke="{DIM}" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="{cx-e_dia_px/2-20}" y1="{base_y}" x2="{cx-e_dia_px/2-12}" y2="{base_y}" stroke="{DIM}" stroke-width="1"/>
  <line x1="{cx-e_dia_px/2-20}" y1="{tip_y}" x2="{cx-e_dia_px/2-12}" y2="{tip_y}" stroke="{DIM}" stroke-width="1"/>
  <text x="{cx-e_dia_px/2-22}" y="{(base_y+tip_y)/2}" fill="{TRON}" font-size="9" font-family="monospace" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90,{cx-e_dia_px/2-22},{(base_y+tip_y)/2})">{e_len:.0f} mm</text>

  <!-- gap dimension -->
  <line x1="{cx-housing_depth_px-28}" y1="{tip_y}" x2="{cx-housing_depth_px-28}" y2="{ring_y}" stroke="{DIM}" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="{cx-housing_depth_px-33}" y1="{tip_y}" x2="{cx-housing_depth_px-23}" y2="{tip_y}" stroke="{DIM}" stroke-width="1"/>
  <line x1="{cx-housing_depth_px-33}" y1="{ring_y}" x2="{cx-housing_depth_px-23}" y2="{ring_y}" stroke="{DIM}" stroke-width="1"/>
  <text x="{cx-housing_depth_px-36}" y="{(tip_y+ring_y)/2}" fill="{TRON}" font-size="10" font-family="monospace" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90,{cx-housing_depth_px-36},{(tip_y+ring_y)/2})">gap {gap:.0f} mm</text>

  <!-- ion wind -->
  {''.join([f'<line x1="{cx+(i-1)*20}" y1="{tip_y+6}" x2="{cx+(i-1)*20}" y2="{ring_y-10}" stroke="{TRON}" stroke-width="1" stroke-opacity=".3" stroke-dasharray="4,3"/>' for i in range(3)])}

  <text x="16" y="{H-20}" fill="#555" font-size="8" font-family="monospace">Side view — 90° rotation from front</text>
'''

    # ── ISOMETRIC VIEW — clean cabinet oblique projection ──
    def iso_view():
        import math
        ox, oy = cx - 20, cy + 60

        s     = scale * 0.55
        ci_r  = c_inner / 2 * s
        co_r  = c_outer / 2 * s
        h_r   = housing / 2 * s
        gp    = gap * s
        el    = min(e_len * s, gp * 0.7)
        edia  = max(e_dia * s, 3)
        rthk  = max((c_outer - c_inner) / 2 * s, 5)
        depth = co_r * 0.28
        h_dep = h_r * 0.28

        ring_cy  = oy
        tip_y2   = ring_cy - gp
        base_y2  = tip_y2 - el
        ring_bot = ring_cy + rthk
        hv_x     = ox + co_r + 14

        n_arrows = 5
        arrow_xs = [ox + (i - n_arrows//2) * (ci_r * 1.6 / n_arrows) for i in range(n_arrows)]
        arrows = ''.join([
            f'<line x1="{x:.1f}" y1="{tip_y2+6:.1f}" x2="{x:.1f}" y2="{ring_cy-14:.1f}" '
            f'stroke="{TRON}" stroke-width="1" stroke-opacity=".5" stroke-dasharray="5,3"/>'
            f'<polygon points="{x:.1f},{ring_cy-14:.1f} {x-4:.1f},{ring_cy-24:.1f} {x+4:.1f},{ring_cy-24:.1f}" '
            f'fill="{TRON}" fill-opacity=".6"/>'
            for x in arrow_xs
        ])

        inner_ry = ci_r / co_r * depth if co_r > 0 else depth * 0.5

        return f"""
  <!-- HOUSING cylinder (dashed) -->
  <ellipse cx="{ox:.1f}" cy="{ring_bot+12:.1f}" rx="{h_r:.1f}" ry="{h_dep:.1f}" fill="none" stroke="#1e3a4a" stroke-width="1" stroke-dasharray="5,3"/>
  <line x1="{ox-h_r:.1f}" y1="{ring_bot+12:.1f}" x2="{ox-h_r:.1f}" y2="{base_y2-10:.1f}" stroke="#1e3a4a" stroke-width="1" stroke-dasharray="5,3"/>
  <line x1="{ox+h_r:.1f}" y1="{ring_bot+12:.1f}" x2="{ox+h_r:.1f}" y2="{base_y2-10:.1f}" stroke="#1e3a4a" stroke-width="1" stroke-dasharray="5,3"/>
  <ellipse cx="{ox:.1f}" cy="{base_y2-10:.1f}" rx="{h_r:.1f}" ry="{h_dep:.1f}" fill="none" stroke="#1e3a4a" stroke-width="1.5" stroke-dasharray="5,3"/>
  <text x="{ox+h_r+10}" y="{base_y2-10:.1f}" fill="#1e6a7a" font-size="10" font-family="monospace" dominant-baseline="middle">PETG  Ø{housing:.0f} mm</text>

  <!-- COLLECTOR RING top face -->
  <ellipse cx="{ox:.1f}" cy="{ring_cy:.1f}" rx="{co_r:.1f}" ry="{depth:.1f}" fill="#112830" stroke="{TRON}" stroke-width="2"/>
  <ellipse cx="{ox:.1f}" cy="{ring_cy:.1f}" rx="{ci_r:.1f}" ry="{inner_ry:.1f}" fill="{BG}" stroke="{TRON}" stroke-width="1.5"/>
  <!-- ring side walls -->
  <line x1="{ox-co_r:.1f}" y1="{ring_cy:.1f}" x2="{ox-co_r:.1f}" y2="{ring_bot:.1f}" stroke="{TRON}" stroke-width="1.5"/>
  <line x1="{ox+co_r:.1f}" y1="{ring_cy:.1f}" x2="{ox+co_r:.1f}" y2="{ring_bot:.1f}" stroke="{TRON}" stroke-width="1.5"/>
  <line x1="{ox-ci_r:.1f}" y1="{ring_cy:.1f}" x2="{ox-ci_r:.1f}" y2="{ring_bot:.1f}" stroke="{TRON2}" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="{ox+ci_r:.1f}" y1="{ring_cy:.1f}" x2="{ox+ci_r:.1f}" y2="{ring_bot:.1f}" stroke="{TRON2}" stroke-width="1" stroke-dasharray="3,2"/>
  <!-- ring bottom face -->
  <ellipse cx="{ox:.1f}" cy="{ring_bot:.1f}" rx="{co_r:.1f}" ry="{depth:.1f}" fill="#0a1e24" stroke="{TRON2}" stroke-width="1"/>
  <ellipse cx="{ox:.1f}" cy="{ring_bot:.1f}" rx="{ci_r:.1f}" ry="{inner_ry:.1f}" fill="{BG}" stroke="{TRON2}" stroke-width="1"/>
  <!-- collector labels -->
  <text x="{hv_x:.1f}" y="{ring_cy:.1f}" fill="{TRON}" font-size="11" font-family="monospace" dominant-baseline="middle" font-weight="bold">Collector ring</text>
  <text x="{hv_x:.1f}" y="{ring_cy+15:.1f}" fill="{GRAY}" font-size="10" font-family="monospace">Al  Ø{c_inner:.0f} / {c_outer:.0f} mm</text>
  <text x="{hv_x:.1f}" y="{ring_cy+29:.1f}" fill="{GRN}" font-size="10" font-family="monospace">⏚  GND</text>

  <!-- EMITTER NEEDLE body -->
  <rect x="{ox-edia:.1f}" y="{base_y2:.1f}" width="{edia*2:.1f}" height="{el-6:.1f}" fill="#222" stroke="#999" stroke-width="1" rx="1"/>
  <ellipse cx="{ox:.1f}" cy="{base_y2:.1f}" rx="{edia:.1f}" ry="{edia*0.35:.1f}" fill="#2a2a2a" stroke="#aaa" stroke-width="1"/>
  <!-- tip cone -->
  <line x1="{ox-edia:.1f}" y1="{tip_y2-4:.1f}" x2="{ox:.1f}" y2="{tip_y2:.1f}" stroke="#eee" stroke-width="1.5"/>
  <line x1="{ox+edia:.1f}" y1="{tip_y2-4:.1f}" x2="{ox:.1f}" y2="{tip_y2:.1f}" stroke="#eee" stroke-width="1.5"/>
  <ellipse cx="{ox:.1f}" cy="{tip_y2-4:.1f}" rx="{edia:.1f}" ry="{edia*0.35:.1f}" fill="#ddd" stroke="#fff" stroke-width=".5"/>
  <!-- emitter labels -->
  <text x="{hv_x:.1f}" y="{(base_y2+tip_y2)/2:.1f}" fill="{RED}" font-size="11" font-family="monospace" dominant-baseline="middle" font-weight="bold">Emitter needle</text>
  <text x="{hv_x:.1f}" y="{(base_y2+tip_y2)/2+15:.1f}" fill="{RED}" font-size="10" font-family="monospace">Ø{e_dia:.1f} mm  ·  {voltage_kv:.0f} kV (+)</text>

  <!-- ION WIND -->
  {arrows}
  <text x="{ox:.1f}" y="{(tip_y2+ring_cy)/2+4:.1f}" text-anchor="middle" fill="{TRON}" font-size="9" font-family="monospace" fill-opacity=".7">↓  ion wind  ↓</text>

  <!-- GAP dimension (left) -->
  <line x1="{ox-h_r-22:.1f}" y1="{tip_y2:.1f}" x2="{ox-h_r-22:.1f}" y2="{ring_cy:.1f}" stroke="{DIM}" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="{ox-h_r-27:.1f}" y1="{tip_y2:.1f}" x2="{ox-h_r-17:.1f}" y2="{tip_y2:.1f}" stroke="{DIM}" stroke-width="1"/>
  <line x1="{ox-h_r-27:.1f}" y1="{ring_cy:.1f}" x2="{ox-h_r-17:.1f}" y2="{ring_cy:.1f}" stroke="{DIM}" stroke-width="1"/>
  <text x="{ox-h_r-30:.1f}" y="{(tip_y2+ring_cy)/2:.1f}" fill="{TRON}" font-size="10" font-family="monospace" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90,{ox-h_r-30:.1f},{(tip_y2+ring_cy)/2:.1f})">gap  {gap:.0f} mm</text>

  <!-- COLLECTOR diameter dimension (bottom) -->
  <line x1="{ox-co_r:.1f}" y1="{ring_bot+depth+16:.1f}" x2="{ox+co_r:.1f}" y2="{ring_bot+depth+16:.1f}" stroke="{DIM}" stroke-width="1"/>
  <line x1="{ox-co_r:.1f}" y1="{ring_bot+depth+11:.1f}" x2="{ox-co_r:.1f}" y2="{ring_bot+depth+21:.1f}" stroke="{DIM}" stroke-width="1"/>
  <line x1="{ox+co_r:.1f}" y1="{ring_bot+depth+11:.1f}" x2="{ox+co_r:.1f}" y2="{ring_bot+depth+21:.1f}" stroke="{DIM}" stroke-width="1"/>
  <text x="{ox:.1f}" y="{ring_bot+depth+32:.1f}" text-anchor="middle" fill="{TRON}" font-size="10" font-family="monospace">Ø{c_outer:.0f} mm outer</text>

  <!-- HV WIRE -->
  <line x1="{ox:.1f}" y1="{base_y2-6:.1f}" x2="{ox:.1f}" y2="{base_y2-32:.1f}" stroke="{RED}" stroke-width="1.5" stroke-dasharray="4,2"/>
  <rect x="{ox-10:.1f}" y="{base_y2-50:.1f}" width="20" height="16" rx="2" fill="#1a0a0a" stroke="{RED}" stroke-width="1"/>
  <text x="{ox:.1f}" y="{base_y2-39:.1f}" text-anchor="middle" fill="{RED}" font-size="8" font-family="monospace">10MΩ</text>
  <line x1="{ox:.1f}" y1="{base_y2-50:.1f}" x2="{ox-38:.1f}" y2="{base_y2-62:.1f}" stroke="{RED}" stroke-width="1.5" stroke-dasharray="4,2"/>
  <text x="{ox-42:.1f}" y="{base_y2-62:.1f}" text-anchor="end" fill="{RED}" font-size="10" font-family="monospace">HV PSU  {voltage_kv:.0f} kV</text>

  <!-- GND WIRE -->
  <line x1="{ox-co_r:.1f}" y1="{ring_cy:.1f}" x2="{ox-co_r-28:.1f}" y2="{ring_cy:.1f}" stroke="{GRN}" stroke-width="1.5"/>
  <text x="{ox-co_r-32:.1f}" y="{ring_cy+4:.1f}" text-anchor="end" fill="{GRN}" font-size="10" font-family="monospace">GND</text>

  <!-- LEGEND -->
  <rect x="14" y="{H-92}" width="200" height="80" rx="3" fill="#111" stroke="#222"/>
  <text x="24" y="{H-78}" fill="#444" font-size="9" font-family="monospace" letter-spacing="1">LEGEND</text>
  <rect x="24" y="{H-69}" width="10" height="8" fill="#222" stroke="#999"/><text x="40" y="{H-62}" fill="#aaa" font-size="9" font-family="monospace">Emitter (Mo needle, HV+)</text>
  <rect x="24" y="{H-56}" width="10" height="8" fill="#112830" stroke="{TRON}"/><text x="40" y="{H-49}" fill="#aaa" font-size="9" font-family="monospace">Collector ring (Al, GND)</text>
  <rect x="24" y="{H-43}" width="10" height="8" fill="none" stroke="#1e3a4a" stroke-dasharray="3,2"/><text x="40" y="{H-36}" fill="#aaa" font-size="9" font-family="monospace">PETG housing</text>
  <line x1="24" y1="{H-24}" x2="34" y2="{H-24}" stroke="{TRON}" stroke-width="1" stroke-dasharray="4,2"/>
  <polygon points="34,{H-24} 30,{H-28} 30,{H-20}" fill="{TRON}" fill-opacity=".6"/>
  <text x="40" y="{H-20}" fill="#aaa" font-size="9" font-family="monospace">Ion wind (thrust direction ↓)</text>
"""

    views = {
        "front": svg_wrap(front_view(), "FRONT VIEW — CROSS SECTION"),
        "top":   svg_wrap(top_view(),   "TOP VIEW — LOOKING DOWN AXIS"),
        "side":  svg_wrap(side_view(),  "SIDE VIEW — 90° ROTATION"),
        "iso":   svg_wrap(iso_view(),   "ISOMETRIC VIEW — 3D PROJECTION"),
    }

    import uuid
    sketch_id = "sk_" + uuid.uuid4().hex[:8]

    # Return pure JSON — index.html reads data.id and data.views directly.
    # No comment parsing, no injection, no encoding issues.
    return jsonify({
        "id":    sketch_id,
        "views": {
            "front": views["front"],
            "top":   views["top"],
            "side":  views["side"],
            "iso":   views["iso"],
        }
    })

    # SVG canvas: 700 wide × 520 tall
    W, H   = 700, 520
    cx     = W // 2   # centre x
    # Scale: fit the collector ring within canvas
    scale  = min(180 / (c_outer / 2), 200 / gap) if c_outer > 0 else 4

    # Derived pixel dimensions
    c_inner_px = (c_inner / 2) * scale
    c_outer_px = (c_outer / 2) * scale
    housing_px = (housing / 2) * scale
    gap_px     = gap * scale
    e_dia_px   = max(e_dia * scale, 3)
    e_len_px   = min(e_len * scale, gap_px * 0.8)

    # Vertical positions (centre of diagram = H/2)
    cy      = H // 2
    tip_y   = cy - gap_px / 2      # needle tip (top)
    base_y  = tip_y - e_len_px     # needle base
    ring_y  = cy + gap_px / 2      # collector ring centre

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;border:1px solid #1e2a35;border-radius:4px;font-family:monospace;display:block;">
  <rect width="{W}" height="{H}" fill="#0d0d0d"/>

  <!-- ── TITLE ── -->
  <text x="{cx}" y="28" text-anchor="middle" fill="#00c8ff" font-size="13" letter-spacing="2">EHD THRUSTER — CROSS SECTION</text>
  <text x="{cx}" y="46" text-anchor="middle" fill="#555" font-size="10">{voltage_kv:.0f} kV  |  gap = {gap:.0f} mm  |  collector Ø{c_inner:.0f}/{c_outer:.0f} mm</text>

  <!-- ── HOUSING (outer boundary) ── -->
  <rect x="{cx - housing_px}" y="{tip_y - e_len_px - 30}"
        width="{housing_px * 2}" height="{gap_px + e_len_px + 80}"
        rx="6" fill="none" stroke="#1e3a4a" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="{cx + housing_px + 6}" y="{cy}" fill="#1e6a7a" font-size="9" dominant-baseline="middle">PETG housing</text>
  <text x="{cx + housing_px + 6}" y="{cy + 12}" fill="#1e6a7a" font-size="9">Ø{housing:.0f} mm</text>

  <!-- ── COLLECTOR RING (cross-section shown as two rectangles) ── -->
  <!-- left wall -->
  <rect x="{cx - c_outer_px}" y="{ring_y - 8 * scale / 4}"
        width="{c_outer_px - c_inner_px}" height="{8 * scale / 4}"
        fill="#1a3a50" stroke="#00c8ff" stroke-width="1"/>
  <!-- right wall -->
  <rect x="{cx + c_inner_px}" y="{ring_y - 8 * scale / 4}"
        width="{c_outer_px - c_inner_px}" height="{8 * scale / 4}"
        fill="#1a3a50" stroke="#00c8ff" stroke-width="1"/>
  <!-- collector label -->
  <text x="{cx + c_outer_px + 10}" y="{ring_y - 4}" fill="#00c8ff" font-size="10">Collector ring</text>
  <text x="{cx + c_outer_px + 10}" y="{ring_y + 8}" fill="#888" font-size="9">Al6061  Ø{c_inner:.0f}/{c_outer:.0f} mm</text>
  <!-- ground symbol -->
  <text x="{cx + c_outer_px + 10}" y="{ring_y + 22}" fill="#3ddc84" font-size="9">⏚ GND</text>

  <!-- ── EMITTER NEEDLE ── -->
  <!-- needle body -->
  <rect x="{cx - e_dia_px / 2}" y="{base_y}"
        width="{e_dia_px}" height="{e_len_px - 8}"
        fill="#2a2a2a" stroke="#aaa" stroke-width="1"/>
  <!-- needle tip (triangle) -->
  <polygon points="{cx - e_dia_px / 2},{tip_y - 8} {cx + e_dia_px / 2},{tip_y - 8} {cx},{tip_y}"
           fill="#cccccc" stroke="#ffffff" stroke-width="0.5"/>
  <!-- HV label -->
  <text x="{cx + e_dia_px / 2 + 8}" y="{(base_y + tip_y) / 2}" fill="#ff7043" font-size="10" dominant-baseline="middle">Emitter (HV+)</text>
  <text x="{cx + e_dia_px / 2 + 8}" y="{(base_y + tip_y) / 2 + 13}" fill="#888" font-size="9">Mo wire  Ø{e_dia:.1f} mm</text>
  <text x="{cx + e_dia_px / 2 + 8}" y="{(base_y + tip_y) / 2 + 26}" fill="#ff7043" font-size="9">{voltage_kv:.0f} kV (+)</text>

  <!-- ── GAP DIMENSION LINE ── -->
  <line x1="{cx - c_inner_px - 30}" y1="{tip_y}" x2="{cx - c_inner_px - 30}" y2="{ring_y}"
        stroke="#444" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="{cx - c_inner_px - 35}" y1="{tip_y}" x2="{cx - c_inner_px - 25}" y2="{tip_y}" stroke="#444" stroke-width="1"/>
  <line x1="{cx - c_inner_px - 35}" y1="{ring_y}" x2="{cx - c_inner_px - 25}" y2="{ring_y}" stroke="#444" stroke-width="1"/>
  <text x="{cx - c_inner_px - 38}" y="{(tip_y + ring_y) / 2}" fill="#00c8ff" font-size="11"
        text-anchor="middle" dominant-baseline="middle" transform="rotate(-90,{cx - c_inner_px - 38},{(tip_y + ring_y) / 2})">gap = {gap:.0f} mm</text>

  <!-- ── ION WIND ARROWS ── -->
  {''.join([f'''
  <line x1="{cx + (i - 1) * 28}" y1="{tip_y + 10}" x2="{cx + (i - 1) * 28}" y2="{ring_y - 14}"
        stroke="#00c8ff" stroke-width="1" stroke-opacity="0.4" stroke-dasharray="4,3"/>
  <polygon points="{cx + (i-1)*28},{ring_y-14} {cx+(i-1)*28-4},{ring_y-24} {cx+(i-1)*28+4},{ring_y-24}"
           fill="#00c8ff" fill-opacity="0.5"/>''' for i in range(3)])}
  <text x="{cx}" y="{(tip_y + ring_y) / 2 + 6}" text-anchor="middle" fill="#00c8ff" font-size="9" fill-opacity="0.6">ion wind →</text>

  <!-- ── BALLAST RESISTOR ── -->
  <line x1="{cx - 20}" y1="{base_y - 5}" x2="{cx - 20}" y2="{base_y - 40}" stroke="#ff7043" stroke-width="1.5"/>
  <rect x="{cx - 28}" y="{base_y - 40}" width="16" height="28" rx="2" fill="#1a1a1a" stroke="#ff7043" stroke-width="1"/>
  <text x="{cx - 20}" y="{base_y - 48}" text-anchor="middle" fill="#ff7043" font-size="9">10 MΩ ballast</text>

  <!-- ── HV SUPPLY LINE ── -->
  <line x1="{cx - 20}" y1="{base_y - 12}" x2="{cx - 60}" y2="{base_y - 12}" stroke="#ff7043" stroke-width="1.5" stroke-dasharray="5,2"/>
  <text x="{cx - 65}" y="{base_y - 8}" text-anchor="end" fill="#ff7043" font-size="9">HV PSU</text>
  <text x="{cx - 65}" y="{base_y + 4}" text-anchor="end" fill="#888" font-size="9">{voltage_kv:.0f} kV DC</text>

  <!-- ── GROUND LINE ── -->
  <line x1="{cx - c_outer_px}" y1="{ring_y}" x2="{cx - c_outer_px - 30}" y2="{ring_y}" stroke="#3ddc84" stroke-width="1.5"/>
  <text x="{cx - c_outer_px - 35}" y="{ring_y + 4}" text-anchor="end" fill="#3ddc84" font-size="9">GND</text>

  <!-- ── DIMENSION: emitter dia ── -->
  <line x1="{cx - e_dia_px / 2}" y1="{base_y + e_len_px / 2}" x2="{cx - e_dia_px / 2 - 12}" y2="{base_y + e_len_px / 2}" stroke="#666" stroke-width="0.8"/>
  <line x1="{cx + e_dia_px / 2}" y1="{base_y + e_len_px / 2}" x2="{cx + e_dia_px / 2 + 4}" y2="{base_y + e_len_px / 2}" stroke="#666" stroke-width="0.8"/>

  <!-- ── LEGEND ── -->
  <rect x="20" y="{H - 80}" width="200" height="68" rx="3" fill="#111" stroke="#1e1e1e"/>
  <text x="30" y="{H - 64}" fill="#555" font-size="9" letter-spacing="1">LEGEND</text>
  <rect x="30" y="{H - 56}" width="10" height="8" fill="#2a2a2a" stroke="#aaa"/>
  <text x="46" y="{H - 49}" fill="#aaa" font-size="9">Emitter (Mo wire, HV+)</text>
  <rect x="30" y="{H - 44}" width="10" height="8" fill="#1a3a50" stroke="#00c8ff"/>
  <text x="46" y="{H - 37}" fill="#aaa" font-size="9">Collector ring (Al, GND)</text>
  <rect x="30" y="{H - 32}" width="10" height="8" fill="none" stroke="#1e3a4a" stroke-dasharray="3,2"/>
  <text x="46" y="{H - 25}" fill="#aaa" font-size="9">PETG housing (3D printed)</text>

  <!-- ── SCALE NOTE ── -->
  <text x="{W - 20}" y="{H - 10}" text-anchor="end" fill="#333" font-size="8">Not to scale — dimensions in mm</text>
</svg>'''

    return svg, 200, {'Content-Type': 'text/plain; charset=utf-8'}


@app.route("/clear", methods=["POST"])
def clear():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "cleared"})


@app.route("/model", methods=["GET", "POST"])
def model_endpoint():
    global MODEL, REASONING_MODEL, INTERACTION_MODEL
    if request.method == "POST":
        data = request.json or {}
        valid_ids = [m["id"] for m in AVAILABLE_MODELS]

        # Support setting either slot independently, or legacy single-model
        changed = {}
        if "reasoning_model" in data:
            if data["reasoning_model"] in valid_ids:
                REASONING_MODEL = data["reasoning_model"]
                changed["reasoning_model"] = REASONING_MODEL
        if "interaction_model" in data:
            if data["interaction_model"] in valid_ids:
                INTERACTION_MODEL = data["interaction_model"]
                MODEL = INTERACTION_MODEL  # keep legacy MODEL in sync
                changed["interaction_model"] = INTERACTION_MODEL
        # Legacy single-model switch (sets both to same model)
        if "model" in data and "reasoning_model" not in data and "interaction_model" not in data:
            new_model = data["model"].strip()
            if new_model in valid_ids:
                MODEL = new_model
                INTERACTION_MODEL = new_model
                changed = {"model": MODEL}
            else:
                return jsonify({"error": f"Unknown model: {new_model}"}), 400

        if changed:
            return jsonify({"status": "ok", **changed})
        return jsonify({"error": "No valid model field provided"}), 400

    return jsonify({
        "reasoning_model":  REASONING_MODEL,
        "interaction_model": INTERACTION_MODEL,
        "current":          MODEL,  # legacy compat
        "available":        AVAILABLE_MODELS,
    })


@app.route("/experiments", methods=["GET"])
def experiments():
    exps = []
    avg  = {}
    return jsonify({"experiments": exps, "summary": avg})


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "reasoning_model":   REASONING_MODEL,
        "interaction_model": INTERACTION_MODEL,
        "model":             MODEL,
        "modules": [
            "core performance",
            "plasma physics",
            "magnetic confinement",
            "ionization (EEDF)",
            "neutralizer",
            "discharge stability",
            "sheath thickness",
            "CEX erosion",
            "ion optics",
            "thermal model",
            "lifetime predictor",
            "multi-mode optimizer",
            "self-consistent solver",
            "air plasma chemistry",
            "atmospheric discharge",
            "self-neutralization",
            "EHD thrust model",        # NEW
            "electrode geometry",      # NEW
            "multi-stage design",      # NEW
            "power efficiency",        # NEW
            "environment correction",  # NEW
        ],
        "total_functions": 42,
        "vacuum_functions": 26,
        "air_breathing_functions": 6,
        "ehd_functions": 8,
        "solver_functions": 2,
        "experiments": 0,
    })


# Set longer socket timeout for slow model responses (DeepSeek R1 can take 60-90s)
import socket
socket.setdefaulttimeout(600)  # 10 minutes

if __name__ == "__main__":
    print("=" * 55)
    print("  Ion Thruster AI — Dual Mode Physics Engine")
    print("=" * 55)
    print("  Vacuum:        26 functions (Xe gridded ion)")
    print("  Air/EHD:       14 functions (atmospheric + EHD)")
    print("  Solver:         2 functions (self-consistent)")
    print("  Total:         42 physics functions")
    print("  Open: http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, port=5000, threaded=True)
