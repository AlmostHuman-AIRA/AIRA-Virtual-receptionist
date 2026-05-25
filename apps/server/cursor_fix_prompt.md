# Cursor Prompt: Fix Time Awareness & Slack Reply Parsing

## Context
This is an AI receptionist system (Jarvis) that schedules meetings.
Relevant files:
- `services/query_router.py` — handles scheduling logic, slot checking, Slack reply routing
- `receptionist/database.py` — `get_available_slots` lives here
- `models/groq_processor.py` — `_normalize_time` and `_build_system_message` live here

## Critical time format rule (read before touching any time code)
- Slots are stored in the DB and compared internally as **24-hour `"HH:MM"` strings** (e.g. `"13:30"`). Keep all internal comparisons in this format — string comparison works correctly for `"09:00" < "13:30"`.
- Everything **shown to the user or passed to the LLM** must be **12-hour format with AM/PM** (e.g. `"1:30 PM"`). The LLM inherits this and will say "1:30" not "13:30".
- Add a single helper `_fmt_display(hhmm: str) -> str` in `query_router.py` that converts from internal 24hr to display 12hr. Use it everywhere a time string is passed to `_llm_reply`.

```python
# Add near the top of query_router.py, after imports
def _fmt_display(hhmm: str) -> str:
    """Convert internal 24hr 'HH:MM' to display '12:30 PM' for the LLM and user."""
    if not hhmm:
        return hhmm
    try:
        from datetime import datetime
        return datetime.strptime(hhmm, "%H:%M").strftime("%-I:%M %p")  # e.g. "1:30 PM"
        # On Windows use %#I instead of %-I:
        # return datetime.strptime(hhmm, "%H:%M").strftime("%#I:%M %p")
    except ValueError:
        return hhmm  # fallback: return as-is if already formatted
```

---

## Bug 1: `get_available_slots` returns past time slots

### Observed behaviour
At 12:14, the system suggests `9:00 AM`, `9:30 AM`, `10:00 AM` — all already in the past.

### Root cause
`get_available_slots(employee_name, date)` in `receptionist/database.py` returns ALL
unbooked slots for the day with no concept of the current wall-clock time.

### Fix required in `receptionist/database.py`

Modify `get_available_slots` to accept two optional parameters. Both bounds use the
**internal 24-hour `"HH:MM"` format** for comparison, matching how slots are stored.

```python
def get_available_slots(
    employee_name: str,
    date: str,
    after_time: str = None,   # internal "HH:MM" 24hr — exclude slots at or before this
    before_time: str = None,  # internal "HH:MM" 24hr — exclude slots at or after this
) -> list[str]:
    # ... existing DB query to get `slots` as a list of "HH:MM" strings ...

    from datetime import datetime

    today_str = datetime.now().strftime("%Y-%m-%d")

    # Always filter out past slots when querying for today
    if date == today_str:
        current_hhmm = datetime.now().strftime("%H:%M")  # 24hr for comparison only
        slots = [s for s in slots if s > current_hhmm]

    # Apply caller-supplied bounds (also 24hr strings, for string comparison)
    if after_time:
        slots = [s for s in slots if s > after_time]
    if before_time:
        slots = [s for s in slots if s < before_time]

    return slots
```

---

## Bug 2: `_handle_scheduling` does not pass current time to slot query

### Fix required in `services/query_router.py`, function `_handle_scheduling`

Replace the raw `get_available_slots` call. Pass `after_time` in internal 24hr format
for comparison. When formatting slots for the LLM, convert to 12hr with `_fmt_display`.

```python
# BEFORE:
slots = get_available_slots(state["sched_employee_name"], state["sched_date"])
if state["sched_time"] not in slots:
    return await _llm_reply(
        f"Suggest alternate slots: {', '.join(slots[:3])}.", state, query, client_id
    )

# AFTER:
from datetime import datetime
today_str = datetime.now().strftime("%Y-%m-%d")
# after_time stays in 24hr — it's an internal bound passed to DB layer
after_time = datetime.now().strftime("%H:%M") if state["sched_date"] == today_str else None

slots = get_available_slots(
    state["sched_employee_name"],
    state["sched_date"],
    after_time=after_time,
)

if not slots:
    return await _llm_reply(
        "Tell the visitor there are no more available slots today. "
        "Ask if they'd like to schedule for tomorrow instead.",
        state, query, client_id
    )

if state["sched_time"] not in slots:
    # Convert slots to 12hr display format before passing to LLM
    display_slots = [_fmt_display(s) for s in slots[:3]]
    return await _llm_reply(
        f"The requested time is unavailable. Suggest these alternate slots: "
        f"{', '.join(display_slots)}.",
        state, query, client_id
    )
```

Apply the same `after_time` fix and `_fmt_display` wrapping in `_handle_availability_check`
wherever it calls `get_available_slots` and passes slot strings to `_llm_reply`.

---

## Bug 3: Slack reply time constraint is ignored

### Observed behaviour
Lucy replies: `"i wont be in office that time, can u do it before 12:30"`
System finds `12:30` in the reply, stores it as `sched_time = "12:30"`, then queries
slots with no upper bound — returning 1:00 PM, 1:30 PM, 2:00 PM (all after 12:30).

### Root cause
The `SLACK_REPLY` handler parses a time but ignores the word "before", so no upper
bound is ever stored. The slot query runs unbounded.

### Fix required in `services/query_router.py`, inside the `SLACK_REPLY` handling block

Find the existing `time_match` block and replace the entire section:

```python
# EXISTING code to keep — finds a time in the host's Slack reply
time_match = re.search(
    r'\b((?:1[0-2]|0?[1-9])(?::[0-5][0-9])?\s*(?:am|pm)?|(?:[01]?\d|2[0-3]):[0-5][0-9])\b',
    slack_reply, re.IGNORECASE
)

# ── REPLACE the existing `if time_match: state["sched_time"] = new_time` with this ──
if time_match:
    # _normalize_time returns an internal 24hr "HH:MM" string
    constraint_time_24 = _normalize_time(time_match.group(1))

    if constraint_time_24:
        reply_lower = slack_reply.lower()
        before_match = re.search(r'\bbefore\b', reply_lower)
        after_match  = re.search(r'\b(after|from)\b', reply_lower)

        if before_match:
            # "before 12:30" → upper bound, visitor must pick a new slot
            state["sched_time_before"] = constraint_time_24  # internal 24hr
            state["sched_time"] = None   # clear so visitor re-confirms
        elif after_match:
            # "after 2pm" → lower bound
            state["sched_time_after"] = constraint_time_24   # internal 24hr
            state["sched_time"] = None
        else:
            # Host directly proposed a specific time (no before/after)
            state["sched_time"] = constraint_time_24
# ─────────────────────────────────────────────────────────────────────────────

# Then update the situation string sent to the LLM to use 12hr display format:
if not host_confirmed:
    constraint_desc = ""
    if state.get("sched_time_before"):
        constraint_desc = f"before {_fmt_display(state['sched_time_before'])}"
    elif state.get("sched_time_after"):
        constraint_desc = f"after {_fmt_display(state['sched_time_after'])}"
    elif state.get("sched_time"):
        constraint_desc = f"at {_fmt_display(state['sched_time'])}"

    situation = (
        f'{host_name} replied via Slack: "{slack_reply}". '
        f"{'The host proposes a time ' + constraint_desc + '. ' if constraint_desc else ''}"
        f"Ask the visitor if this works or relay the message naturally."
    )
```

Then in `_handle_scheduling`, pass both bounds (internal 24hr) to the DB query and
display them in 12hr to the LLM:

```python
from datetime import datetime
today_str = datetime.now().strftime("%Y-%m-%d")
after_time  = datetime.now().strftime("%H:%M") if state["sched_date"] == today_str else None
before_time = state.get("sched_time_before")  # internal 24hr, set by Slack parser

host_after = state.get("sched_time_after")    # internal 24hr
if host_after:
    after_time = max(after_time, host_after) if after_time else host_after

slots = get_available_slots(
    state["sched_employee_name"],
    state["sched_date"],
    after_time=after_time,
    before_time=before_time,
)

# Clear one-shot constraints after use
state.pop("sched_time_before", None)
state.pop("sched_time_after", None)

if not slots:
    # Build human-readable constraint description in 12hr for the LLM
    if before_time and after_time:
        constraint_desc = f"between {_fmt_display(after_time)} and {_fmt_display(before_time)}"
    elif before_time:
        constraint_desc = f"before {_fmt_display(before_time)}"
    elif after_time:
        constraint_desc = f"after {_fmt_display(after_time)}"
    else:
        constraint_desc = "today"

    return await _llm_reply(
        f"Tell the visitor there are no available slots {constraint_desc}. "
        f"Ask if they'd like a different day.",
        state, query, client_id
    )

if state["sched_time"] not in slots:
    display_slots = [_fmt_display(s) for s in slots[:3]]  # 12hr for LLM
    return await _llm_reply(
        f"The requested time is unavailable. Suggest these alternate slots: "
        f"{', '.join(display_slots)}.",
        state, query, client_id
    )
```

---

## Bug 4: LLM system prompt does not include current time

### Root cause
`_llm_reply` builds its own prompt without current time, so the LLM can hallucinate
past slots or format times incorrectly.

### Fix required in `services/query_router.py`, function `_llm_reply`

```python
async def _llm_reply(situation: str, state, user_query: str, client_id: str) -> str:
    from datetime import datetime
    llm = GroqProcessor.get_instance()
    visitor = state.get("visitor_name") or "Visitor"
    host = (
        state.get("meeting_with_resolved")
        or state.get("sched_employee_name")
        or "Admin"
    )
    # Inject current time in 12hr format so LLM uses correct time language
    now_str = datetime.now().strftime("%-I:%M %p on %A, %d %B %Y")  # e.g. "12:14 PM on Monday, 19 May 2026"
    # Windows: use %#I instead of %-I
    time_context = f"[Current time: {now_str}] "

    info = f"TALKING TO: {visitor} | HOST: {host} | STATUS: {state['conv_state']}"
    is_first = not state.get("greeted", False)
    prompt = (
        f"{BASE_SYSTEM_PROMPT}\n"
        f"KB: {info}\n"
        f"{time_context}"
        f"TIME FORMAT RULE: Always say times in 12-hour format with AM/PM (e.g. '1:30 PM', not '13:30').\n"
        f"RULES: {'Greet warmly' if is_first else 'DO NOT greet again'}. "
        f"GOAL: {situation}\n"
        f"USER: {user_query}"
    )
    resp = await llm.get_raw_response(prompt, client_id=client_id)
    state["greeted"] = True
    return resp
```

---

## Summary of all changes

| File | Function | Change |
|---|---|---|
| `services/query_router.py` | top of file | Add `_fmt_display(hhmm)` helper — converts internal `"HH:MM"` to display `"1:30 PM"` |
| `receptionist/database.py` | `get_available_slots` | Add `after_time` / `before_time` params (24hr strings); auto-filter past slots for today |
| `services/query_router.py` | `_handle_scheduling` | Pass `after_time=now` (24hr) to DB; wrap slot strings in `_fmt_display()` before passing to LLM |
| `services/query_router.py` | `_handle_availability_check` | Same as above |
| `services/query_router.py` | `SLACK_REPLY` handler | Parse `before`/`after` keywords; store 24hr bounds in state; pass 12hr display to LLM situation string |
| `services/query_router.py` | `_llm_reply` | Inject `[Current time: 12hr]` + explicit `TIME FORMAT RULE: use 12-hour AM/PM` into every prompt |

## Verification checklist after fix
- [ ] User says "1:30 PM" → Jarvis repeats "1:30 PM", never "13:30"
- [ ] At 12:14, no slots before 12:14 are suggested
- [ ] After Lucy says "before 12:30", offered slots are between current time and 12:30 PM
- [ ] After Lucy says "after 2 PM", offered slots are after 2:00 PM
- [ ] If no slots exist in the window, Jarvis says so in 12hr language and offers tomorrow
- [ ] `_llm_reply` prompts always contain current time in 12hr format
