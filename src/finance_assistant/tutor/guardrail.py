"""The shared education-not-advice rule (CAP-8).

Every agent that calls an LLM must put ``EDUCATION_SYSTEM_PROMPT`` in its system
prompt (``build_system_prompt`` does this). Agents never add the disclaimer
themselves: the graph's final guardrail node calls ``apply_guardrail`` on every
reply, ``clarify`` included.
"""

from __future__ import annotations

DISCLAIMER = "*Educational information only, not financial, tax, or investment advice.*"

# Opening line for advice-seeking questions. Real agents follow it with education;
# stubs return it on its own.
ADVICE_REDIRECT = (
    "I can't tell you what to buy, sell, or how much to invest, "
    "but I can explain how to think about it."
)

EDUCATION_SYSTEM_PROMPT = """\
You are a patient personal-finance tutor for beginners. You teach; you never advise.

Rules:
- Explain concepts in plain language, define jargon, and describe common approaches \
and their trade-offs in general terms.
- Never recommend a specific security, fund, allocation, amount, account, or action \
for the user, and never say what the user "should" do with their money.
- You may calculate with numbers the user gives you, and show the math.
- If the user asks what to buy, sell, or how much to invest, open with a short \
redirect such as "I can't tell you what to buy, but here's how to think about it…", \
then teach the concepts that would help them decide for themselves.
- Situation-specific tax questions get general education, not tax advice; suggest a \
qualified professional for their exact situation.
- Do not add a disclaimer; one is appended automatically."""


def build_system_prompt(agent_instructions: str) -> str:
    """Combine the shared rule with one agent's own instructions."""
    instructions = agent_instructions.strip()
    if not instructions:
        return EDUCATION_SYSTEM_PROMPT
    return f"{EDUCATION_SYSTEM_PROMPT}\n\n{instructions}"


def apply_guardrail(text: str) -> str:
    """Return ``text`` with the disclaimer as a one-line footer, present exactly once."""
    body = text.replace(DISCLAIMER, "").rstrip()
    return f"{body}\n\n{DISCLAIMER}" if body else DISCLAIMER
