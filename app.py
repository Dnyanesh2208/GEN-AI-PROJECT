import re
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError, ServerError

st.set_page_config(page_title="Concept Explainer (ELI5)", page_icon="🧠", layout="centered")

MODEL_NAME = "gemini-3.5-flash-lite"

PRESET_EXPLANATIONS = {
    "Recursion": {
        "kid": "Imagine opening a nesting doll, and inside it is a smaller nesting doll, and inside that one is an even smaller one—until you reach the tiny solid one at the end and close them all back up. Recursion is when a helper solves a big puzzle by asking a smaller version of the same puzzle until it reaches an easy stop.",
        "student": "Recursion happens when a function calls itself with a smaller input until it hits a **base case** that stops the loop. For example, to find the factorial of 4 (`4!`), it computes `4 * 3!`, then `3 * 2!`, then `2 * 1!`, and stops at `1! = 1`. Each call waits on the **call stack** until the base case returns.",
        "expert": "Recursion defines a function or structure in terms of itself over a well-founded domain, requiring a terminating base case to guarantee convergence. In imperative languages, each invocation allocates a stack frame (return address, arguments, locals), risking stack overflow on deep recursion unless optimized via tail-call elimination (TCE) or rewritten iteratively with an explicit heap/stack or memoization to avoid exponential overlapping subproblems.",
    },
    "Compound Interest": {
        "kid": "Suppose you put 10 stickers in a box and get 1 extra sticker every week. Next week, you get a tiny bonus sticker for that extra one too! Compound interest is earning 'bonus points' on your points so your pile grows faster and faster.",
        "student": "Compound interest means you earn interest not only on your initial money (**principal**), but also on the interest that has already accumulated. Using `A = P(1 + r/n)^(nt)`, `$1,000` at `10%` compounded yearly becomes `$1,100` after year 1 and `$1,210` after year 2—earning `$110` in year 2 instead of `$100`.",
        "expert": "Compound interest models discrete or continuous exponential capital growth via reinvestment of accrued interest into the principal base: `A(t) = P(1 + r/n)^(nt)`, converging to `Pe^(rt)` as compounding frequency `n -> ∞`. Key nuances include effective annual rate (EAR) vs. nominal APR, day-count conventions, inflation/tax drag on real returns, and sequence-of-returns risk when cash flows are irregular.",
    },
}


@st.cache_resource
def get_client() -> genai.Client:
    return genai.Client()


def parse_sections(raw_text: str) -> dict[str, str]:
    sections = {"kid": "", "student": "", "expert": ""}
    kid_match = re.search(r"===KID===\s*(.*?)(?====STUDENT===|===EXPERT===|$)", raw_text, re.S)
    student_match = re.search(r"===STUDENT===\s*(.*?)(?====EXPERT===|$)", raw_text, re.S)
    expert_match = re.search(r"===EXPERT===\s*(.*?)$", raw_text, re.S)

    if kid_match:
        sections["kid"] = kid_match.group(1).strip()
    if student_match:
        sections["student"] = student_match.group(1).strip()
    if expert_match:
        sections["expert"] = expert_match.group(1).strip()

    if not any(sections.values()):
        sections["kid"] = raw_text.strip()
    return sections


@st.cache_data(show_spinner=False)
def fetch_full_explanation(concept: str) -> dict[str, str]:
    key = concept.strip().lower()
    if key == "recursion":
        return PRESET_EXPLANATIONS["Recursion"]
    if key == "compound interest":
        return PRESET_EXPLANATIONS["Compound Interest"]

    client = get_client()
    prompt = f"""
Explain "{concept}" concisely at 3 levels (~60-80 words each).
Use these exact headers and nothing else:

===KID===
Simple analogy for an 8-year-old, zero jargon.

===STUDENT===
High school/early college explanation with key terms and 1 short example.

===EXPERT===
Technical mechanisms, edge cases, and trade-offs.
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=420,
            temperature=0.4,
        ),
    )
    return parse_sections(response.text or "")


def stream_explanation(concept: str):
    client = get_client()
    prompt = f"""
Explain "{concept}" concisely at 3 levels (~60-80 words each).
Use these exact headers:

===KID===
Simple analogy for an 8-year-old, zero jargon.

===STUDENT===
High school/early college explanation with key terms and 1 short example.

===EXPERT===
Technical mechanisms, edge cases, and trade-offs.
"""
    stream = client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=420,
            temperature=0.4,
        ),
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def render_levels(data: dict[str, str], display_mode: str) -> None:
    levels = [
        ("🧒 Kid", data.get("kid", "")),
        ("🎓 Student", data.get("student", "")),
        ("🔬 Expert", data.get("expert", "")),
    ]
    if display_mode == "Tabs":
        tabs = st.tabs([label for label, _ in levels])
        for tab, (_, text) in zip(tabs, levels):
            with tab:
                st.markdown(text or "_No content generated for this level._")
    else:
        for label, text in levels:
            with st.expander(label, expanded=(label.startswith("🧒"))):
                st.markdown(text or "_No content generated for this level._")


st.title("Concept Explainer (ELI5)")
st.write("Enter any complex topic to get Kid, Student, and Expert explanations from a single prompt.")

display_mode = st.radio("Display layout", ["Tabs", "Expanders"], horizontal=True)

col1, col2 = st.columns(2)
preset_clicked = None
if col1.button("⚡ Instant: Recursion"):
    preset_clicked = "Recursion"
if col2.button("⚡ Instant: Compound Interest"):
    preset_clicked = "Compound Interest"

concept = st.text_input(
    "Concept to explain",
    value=preset_clicked or "",
    placeholder="e.g., DNS, Backpropagation, Inflation",
)

should_run = st.button("Explain Concept", type="primary") or (preset_clicked is not None)

if should_run:
    cleaned = concept.strip()
    if not cleaned:
        st.warning("Please enter a concept first.")
    else:
        key = cleaned.lower()
        try:
            if key in ("recursion", "compound interest"):
                render_levels(fetch_full_explanation(cleaned), display_mode)
            else:
                cached = fetch_full_explanation.clear  # keep cache working cleanly
                live_box = st.empty()
                collected = ""
                for piece in stream_explanation(cleaned):
                    collected += piece
                    live_box.markdown(collected)
                live_box.empty()
                parsed = parse_sections(collected)
                render_levels(parsed, display_mode)
        except ServerError:
            st.error("The API is temporarily busy. Please click **Explain Concept** again.")
        except APIError as exc:
            st.error(f"API error ({exc.code}): {exc.message}")
        except Exception as exc:
            st.error(f"Error: {exc}")