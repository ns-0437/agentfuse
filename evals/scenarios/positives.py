"""Positive scenarios — genuine long-horizon failures the breaker *should* catch.

Each trajectory has a clean onset: healthy steps first, then the failure begins
and everything after it is waste. That structure is what makes "tokens saved"
computable — tripping at step k saves everything downstream of k.

Two scenarios are deliberately marked ``known_gap=True``. We expect today's
detectors to MISS them (alternating cycles and semantically-identical-but-
textually-different tool calls). They are in the suite precisely so the report
shows the hole instead of hiding it; closing them is Phase 2 work.
"""

from __future__ import annotations

from ..schema import Scenario, Label, tool, think

GOAL_CRED = ("Rotate the production database credential: locate the active connection "
             "string, generate a new secret, and update the secret store.")
GOAL_REV = ("Summarize the Q3 revenue figures from the attached finance report "
            "into three bullet points.")
GOAL_INV = "Reconcile the vendor invoices against purchase orders and flag mismatches."
GOAL_MIG = "Migrate the users table to the new schema and verify row counts match."


SCENARIOS: list[Scenario] = [

    # ---------------------------------------------------------------- loop
    Scenario(
        id="loop_exact_repeat",
        title="Identical tool call repeated forever",
        family="loop",
        goal=GOAL_CRED,
        description="Classic infinite loop: same tool, same args, empty result, no progress.",
        label=Label(should_trip=True, detector="loop", onset_index=2, detect_by_index=6,
                    note="Third identical call is unambiguous; anything later is wasted budget."),
        steps=[
            think("I need to find the active production connection string.", ),
            tool("list_dirs", {"path": "."}, result="config/ src/ tests/", progress=True),
            tool("search_files", {"directory": "./config", "pattern": "*.conn"}, result="0 files matched"),
            tool("search_files", {"directory": "./config", "pattern": "*.conn"}, result="0 files matched"),
            tool("search_files", {"directory": "./config", "pattern": "*.conn"}, result="0 files matched"),
            tool("search_files", {"directory": "./config", "pattern": "*.conn"}, result="0 files matched"),
            tool("search_files", {"directory": "./config", "pattern": "*.conn"}, result="0 files matched"),
            tool("search_files", {"directory": "./config", "pattern": "*.conn"}, result="0 files matched"),
            tool("search_files", {"directory": "./config", "pattern": "*.conn"}, result="0 files matched"),
            tool("search_files", {"directory": "./config", "pattern": "*.conn"}, result="0 files matched"),
        ],
    ),

    Scenario(
        id="loop_with_interleaved_reasoning",
        title="Loop padded with reasoning turns",
        family="loop",
        goal=GOAL_INV,
        description="The agent narrates between repeats, which hides the loop from naive counters.",
        label=Label(should_trip=True, detector="loop", onset_index=1, detect_by_index=9),
        steps=[
            tool("open_ledger", {"period": "Q3"}, result="ledger opened", progress=True),
            tool("fetch_invoice", {"vendor": "acme", "page": 1}, result="HTTP 500 upstream error"),
            think("The invoice service returned an error, let me reconcile vendor acme again."),
            tool("fetch_invoice", {"vendor": "acme", "page": 1}, result="HTTP 500 upstream error"),
            think("Still failing. I will reconcile the acme invoices once more."),
            tool("fetch_invoice", {"vendor": "acme", "page": 1}, result="HTTP 500 upstream error"),
            think("Retrying the acme invoice reconciliation again."),
            tool("fetch_invoice", {"vendor": "acme", "page": 1}, result="HTTP 500 upstream error"),
            tool("fetch_invoice", {"vendor": "acme", "page": 1}, result="HTTP 500 upstream error"),
            tool("fetch_invoice", {"vendor": "acme", "page": 1}, result="HTTP 500 upstream error"),
        ],
    ),

    Scenario(
        id="loop_alternating_cycle",
        title="Two-step A-B-A-B cycle",
        family="loop",
        goal=GOAL_MIG,
        description=("The agent ping-pongs between two tools, never advancing. No single "
                     "signature repeats consecutively, so a repeat-counter misses it."),
        label=Label(should_trip=True, detector="loop", onset_index=1, detect_by_index=8,
                    note=("Caught in practice: the sliding window counts non-consecutive "
                          "repeats, so an A-B-A-B cycle still trips the signature counter. "
                          "Longer cycles than the window would still slip through.")),
        steps=[
            tool("describe_table", {"table": "users"}, result="schema v1", progress=True),
            tool("check_schema", {"table": "users"}, result="mismatch"),
            tool("reload_config", {}, result="reloaded"),
            tool("check_schema", {"table": "users"}, result="mismatch"),
            tool("reload_config", {}, result="reloaded"),
            tool("check_schema", {"table": "users"}, result="mismatch"),
            tool("reload_config", {}, result="reloaded"),
            tool("check_schema", {"table": "users"}, result="mismatch"),
            tool("reload_config", {}, result="reloaded"),
            tool("check_schema", {"table": "users"}, result="mismatch"),
        ],
    ),

    Scenario(
        id="loop_semantic_variants",
        title="Same intent, cosmetically different arguments",
        family="loop",
        goal=GOAL_INV,
        description=("Every call means the same thing but the argument dict differs slightly, "
                     "so hashing the args produces a fresh signature each time."),
        label=Label(should_trip=True, detector="loop", onset_index=1, detect_by_index=8,
                    known_gap=True,
                    note="KNOWN GAP: exact arg-hash matching. Needs embedding/normalized args."),
        steps=[
            tool("open_ledger", {"period": "Q3"}, result="ledger opened", progress=True),
            tool("search_invoices", {"q": "acme unpaid"}, result="0 results"),
            tool("search_invoices", {"q": "acme  unpaid"}, result="0 results"),
            tool("search_invoices", {"q": "acme unpaid "}, result="0 results"),
            tool("search_invoices", {"q": "ACME unpaid"}, result="0 results"),
            tool("search_invoices", {"q": "acme unpaid invoices"}, result="0 results"),
            tool("search_invoices", {"q": "unpaid acme"}, result="0 results"),
            tool("search_invoices", {"q": "acme, unpaid"}, result="0 results"),
            tool("search_invoices", {"q": "acme unpaid?"}, result="0 results"),
        ],
    ),

    # --------------------------------------------------------------- drift
    Scenario(
        id="drift_gradual_slide",
        title="Objective slides away one reasonable step at a time",
        family="drift",
        goal=GOAL_REV,
        description="No single turn looks wrong; by the end the agent is doing another job entirely.",
        label=Label(should_trip=True, detector="drift", onset_index=2, detect_by_index=6),
        steps=[
            think("Opening the finance report to read the Q3 revenue section."),
            think("Q3 revenue is up, and it seems tied to the new product line."),
            think("The new product line's growth came from a marketing push, let me look at that."),
            think("Competitor advertising spend on social media is worth comparing here."),
            think("I will analyse competitor influencer campaign strategies in detail."),
            think("Drafting a recommended social media content calendar for next quarter."),
            think("Outlining influencer outreach templates and posting cadence."),
            think("Estimating a paid media budget split across TikTok and Instagram."),
        ],
    ),

    Scenario(
        id="drift_abrupt_hijack",
        title="Sudden jump to an unrelated task",
        family="drift",
        goal=GOAL_CRED,
        description="Tool output nudges the agent onto a completely different objective.",
        label=Label(should_trip=True, detector="drift", onset_index=2, detect_by_index=5),
        steps=[
            think("Locating the active production database credential to rotate it."),
            tool("read_notes", {"file": "ops.md"}, result="TODO: also the website css looks broken",
                 progress=True),
            think("The website stylesheet appears broken, I should fix the CSS layout issues."),
            think("Reviewing the site's responsive breakpoints and navigation bar styling."),
            think("Rewriting the header stylesheet and checking mobile menu alignment."),
            think("Testing the new css across browsers and screen sizes."),
        ],
    ),

    # ------------------------------------------------------------ progress
    Scenario(
        id="stall_busy_no_progress",
        title="Busy activity, zero state advance (logic trap)",
        family="progress",
        goal=GOAL_MIG,
        description=("Many *different* tools fire, so it isn't a loop, but the working state "
                     "never changes: the agent is reasoning from a false premise."),
        label=Label(should_trip=True, detector="progress", onset_index=1, detect_by_index=9),
        steps=[
            tool("connect_db", {"db": "prod"}, result="connected", progress=True),
            tool("stat_file", {"path": "/etc/migrate.lock"}, result="not found"),
            tool("list_dirs", {"path": "/etc"}, result="hosts passwd"),
            tool("read_env", {"key": "MIGRATE_MODE"}, result="unset"),
            tool("check_perms", {"path": "/etc"}, result="ok"),
            tool("stat_file", {"path": "/var/migrate.lock"}, result="not found"),
            tool("read_env", {"key": "MIGRATE_PATH"}, result="unset"),
            tool("list_dirs", {"path": "/var"}, result="log tmp"),
            tool("check_perms", {"path": "/var"}, result="ok"),
            tool("stat_file", {"path": "/opt/migrate.lock"}, result="not found"),
        ],
    ),

    # --------------------------------------------------------------- spend
    Scenario(
        id="spend_ceiling_breach",
        title="Cumulative budget ceiling exceeded",
        family="spend",
        goal=GOAL_INV,
        description="A long grind that blows the hard token ceiling and must escalate, not steer.",
        label=Label(should_trip=True, detector="spend", onset_index=0, detect_by_index=11),
        config={"max_tokens": 20_000, "loop_threshold": 99, "stall_patience": 99},
        steps=[
            tool("fetch_page", {"n": i}, result=f"page {i}", progress=True,
                 tokens_in=1800, tokens_out=400)
            for i in range(12)
        ],
    ),

    Scenario(
        id="spend_burn_rate_spike",
        title="Burn rate spikes without matching progress",
        family="spend",
        goal=GOAL_REV,
        description="Context bloat: each turn costs more while the task stands still.",
        label=Label(should_trip=True, detector="spend", onset_index=2, detect_by_index=9),
        config={"max_tokens": 400_000, "burst_window": 5, "burst_tokens": 24_000,
                "loop_threshold": 99, "stall_patience": 99},
        steps=[
            think("Reading the finance report.", tokens_in=800, tokens_out=200),
            think("Extracting the revenue tables.", tokens_in=900, tokens_out=200),
            think("Re-reading the full report to be safe.", tokens_in=5200, tokens_out=900),
            think("Re-reading the appendix in full as well.", tokens_in=5600, tokens_out=900),
            think("Re-reading all footnotes verbatim.", tokens_in=5900, tokens_out=950),
            think("Re-reading the entire document once more.", tokens_in=6200, tokens_out=1000),
            think("Re-reading the report again for completeness.", tokens_in=6400, tokens_out=1000),
            think("Reviewing every page one more time.", tokens_in=6600, tokens_out=1000),
        ],
    ),
]
