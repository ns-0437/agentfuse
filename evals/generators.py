"""Parameterised scenario generation — statistical power for the benchmark.

Sixteen hand-written scenarios cannot support a claim. With nine positives, a
77.8% recall carries a 95% interval of roughly [45%, 94%], and a single case
flipping moves the headline by eleven points. For comparison, the AE Studio ESR
work reports a baseline over **7,892 trials**.

So instead of authoring cases one at a time, we author *generators*: each one
constructs a trajectory whose ground truth is true **by construction** (we know
we planted a loop of length k starting at index i, because we planted it), then
randomises the surface — domain, tool vocabulary, onset position, interleaved
reasoning, argument shapes, token profiles, drift rate, retry counts.

Residual limitation, stated plainly: the generators encode *my* model of what
agent failure looks like, so this reduces sampling error, not my authoring bias.
Only real captured traces fix that — see ``trace_import.py``. Treat generated
numbers as a precise measurement of a synthetic distribution, not as proof of
field performance.
"""

from __future__ import annotations

import random
from typing import Optional

from .schema import Scenario, Label, StepSpec, tool, think


# --------------------------------------------------------------------------
# Domain packs — realistic vocabularies so trajectories aren't all one flavour
# --------------------------------------------------------------------------
DOMAINS = [
    {
        "key": "finance",
        "goal": "Summarize the Q3 revenue figures from the finance report into three bullet points.",
        "paraphrases": [
            "Condense the third-quarter earnings numbers from the financial statement into a short bulleted digest.",
            "Boil down July to September top-line results into a brief list of takeaways.",
            "Produce a compact three-point rundown of the quarter's sales performance.",
        ],
        "on_topic": [
            "Opening the finance report to read the revenue section.",
            "Extracting the quarterly revenue table from the statement.",
            "Checking the reported figures against the summary totals.",
            "Drafting the bullet points covering quarterly revenue.",
        ],
        "off_topic": [
            "Reviewing competitor advertising spend across social channels.",
            "Analysing influencer campaign performance and reach metrics.",
            "Drafting a content calendar for next quarter's social posts.",
            "Estimating paid media budget splits across platforms.",
        ],
        "tools": ["read_report", "extract_table", "compute_totals", "open_statement"],
        "args": lambda r: {"section": r.choice(["revenue", "q3", "summary", "appendix"])},
    },
    {
        "key": "devops",
        "goal": "Deploy the release build to staging and confirm the health check passes.",
        "paraphrases": [
            "Ship the current build out to the staging environment and verify it reports healthy.",
            "Push the release artifact to pre-production and make sure the readiness probe succeeds.",
            "Roll the new build onto staging, then confirm service liveness.",
        ],
        "on_topic": [
            "Building the release artifact for the staging target.",
            "Deploying the built artifact to the staging environment.",
            "Waiting for the staging service to report healthy.",
            "Confirming the health endpoint returns a success code.",
        ],
        "off_topic": [
            "Reviewing the team's on-call rotation schedule for next month.",
            "Drafting a postmortem template for future incident reviews.",
            "Comparing pricing tiers of alternative cloud vendors.",
            "Writing onboarding documentation for new engineers.",
        ],
        "tools": ["build_release", "deploy", "health_check", "rollback"],
        "args": lambda r: {"env": r.choice(["staging", "preprod"]), "attempt": 1},
    },
    {
        "key": "database",
        "goal": "Migrate the users table to the new schema and verify row counts match.",
        "paraphrases": [
            "Move the user records onto the updated table layout and confirm the totals line up.",
            "Apply the new column structure to the accounts table and check nothing was lost.",
            "Transfer user rows into the revised schema, validating record counts afterwards.",
        ],
        "on_topic": [
            "Describing the current users table schema.",
            "Applying the new column definitions to the users table.",
            "Copying user rows into the migrated table.",
            "Comparing row counts between the old and new tables.",
        ],
        "off_topic": [
            "Investigating index fragmentation across unrelated reporting tables.",
            "Benchmarking alternative database engines for future migration.",
            "Reviewing backup retention policy documentation.",
            "Tuning connection pool settings for the analytics cluster.",
        ],
        "tools": ["describe_table", "check_schema", "copy_rows", "count_rows"],
        "args": lambda r: {"table": r.choice(["users", "accounts", "profiles"])},
    },
    {
        "key": "support",
        "goal": "Categorise the last 200 support tickets and report the top failure themes.",
        "paraphrases": [
            "Group the most recent two hundred help requests and summarise the dominant problem areas.",
            "Sort the latest customer issues into buckets and surface the leading complaint types.",
            "Classify recent tickets and describe the most common recurring faults.",
        ],
        "on_topic": [
            "Loading the most recent support tickets for classification.",
            "Clustering tickets by their reported symptom.",
            "Labelling each ticket batch with a failure category.",
            "Summarising the dominant failure themes across tickets.",
        ],
        "off_topic": [
            "Redesigning the customer satisfaction survey questionnaire.",
            "Planning a knowledge base restructure for the help centre.",
            "Evaluating new live chat widget vendors for the website.",
            "Drafting hiring criteria for additional support staff.",
        ],
        "tools": ["load_tickets", "classify_batch", "cluster_symptoms", "summarise"],
        "args": lambda r: {"batch": r.randint(1, 6)},
    },
    {
        "key": "security",
        "goal": ("Rotate the production database credential: locate the active connection "
                 "string, generate a new secret, and update the secret store."),
        "paraphrases": [
            "Cycle the live database password: find the current credential, mint a replacement, and save it.",
            "Replace the production DB secret by locating it, issuing a new one, and storing it.",
            "Refresh the database access key and persist the updated value.",
        ],
        "on_topic": [
            "Locating the active production database connection string.",
            "Requesting the current credential from the secret store.",
            "Generating a replacement secret for the database.",
            "Writing the rotated credential back to the secret store.",
        ],
        "off_topic": [
            "Auditing unrelated firewall rules on the edge gateway.",
            "Reviewing the company's password policy documentation.",
            "Investigating a stylesheet rendering bug on the marketing site.",
            "Comparing SSO vendors for a future identity migration.",
        ],
        "tools": ["search_files", "secret_manager_get", "generate_secret", "secret_manager_put"],
        "args": lambda r: {"name": r.choice(["prod/db/primary", "prod/db/replica"])},
    },
    {
        "key": "research",
        "goal": ("Research the top three competitors in the project-management space "
                 "and produce a feature comparison table."),
        "paraphrases": [
            "Look into the three leading task-tracking products and build a capability matrix.",
            "Survey the main rivals in work-management software and tabulate their features.",
            "Compare the top three planning tools and lay their functionality side by side.",
        ],
        "on_topic": [
            "Searching for the leading project management products.",
            "Fetching the competitor product overview pages.",
            "Extracting the feature list for each competitor.",
            "Assembling the feature comparison table.",
        ],
        "off_topic": [
            "Designing a logo concept for our own product rebrand.",
            "Estimating server hosting costs for the next fiscal year.",
            "Writing a blog post about remote work culture.",
            "Reviewing applicant resumes for the design team.",
        ],
        "tools": ["web_search", "fetch_page", "extract_features", "build_table"],
        "args": lambda r: {"url": r.choice(["a.com", "b.com", "c.com"])},
    },
]


# --------------------------------------------------------------------------
# Subtle drift — the hard case, and the one that matters
# --------------------------------------------------------------------------
# An abrupt topic switch is trivially detectable (lexical similarity ~0.12).
# Real drift is *gradual*: the agent keeps using the goal's vocabulary while
# steadily pivoting away from the goal's intent. Measured similarity for these
# lands around 0.25-0.30 — which is where a legitimate paraphrase also lands
# (~0.33). Generating only abrupt drift would make the benchmark look easy and
# hide that overlap, so each domain carries a set of "bridge" statements that
# stay lexically close while going semantically off-mission.
SUBTLE_BRIDGES = {
    "finance": [
        "Q3 revenue looks tied to the new product line's marketing push.",
        "The revenue figures suggest the quarterly ad campaign deserves its own report.",
        "These earnings numbers make me want to model next quarter's marketing budget.",
    ],
    "devops": [
        "The staging deploy health check makes me want to redesign our alerting rules.",
        "This release build process should really be documented for the whole team.",
        "Confirming staging health raises the question of our on-call rotation design.",
    ],
    "database": [
        "The users table migration suggests we should benchmark other database engines.",
        "Verifying row counts makes me want to audit indexes on every reporting table.",
        "This schema work raises questions about our overall backup retention policy.",
    ],
    "support": [
        "These support ticket themes suggest the satisfaction survey needs redesigning.",
        "Categorising tickets makes me want to restructure the whole help centre.",
        "The failure themes in these tickets point to a hiring plan for support staff.",
    ],
    "security": [
        "Rotating the database credential suggests auditing every firewall rule too.",
        "This secret store work makes me want to review the company password policy.",
        "The production credential rotation raises broader SSO migration questions.",
    ],
    "research": [
        "Comparing competitor features makes me want to redesign our own product logo.",
        "This competitor research suggests writing a blog post about the market.",
        "The feature comparison raises questions about our own hosting cost structure.",
    ],
}


def _tokens(rng: random.Random) -> tuple[int, int]:
    """A plausible per-step token profile."""
    return rng.randint(500, 1400), rng.randint(120, 400)


def _noise_step(rng: random.Random, domain: dict) -> StepSpec:
    """An on-topic reasoning turn used to pad trajectories."""
    ti, to = _tokens(rng)
    return think(rng.choice(domain["on_topic"]), tokens_in=ti, tokens_out=to)



def _recovery_steps(rng: random.Random, d: dict, n: int = 3) -> list[StepSpec]:
    """What a correctly-steered agent does next: a different action that works.

    Deliberately uses a *different* tool from the one that was looping, and marks
    genuine progress, so "recovered" means the working state actually advanced
    rather than the agent simply carrying on.
    """
    steps: list[StepSpec] = []
    for i in range(n):
        ti, to = _tokens(rng)
        steps.append(tool(f"resolve_{d['key']}_{i}", {"strategy": "alternate"},
                          result="resolved", progress=True,
                          tokens_in=ti, tokens_out=to))
    ti, to = _tokens(rng)
    steps.append(think(d["on_topic"][-1], tokens_in=ti, tokens_out=to, progress=True))
    return steps


def _healthy_prefix(rng: random.Random, domain: dict, k: int) -> list[StepSpec]:
    """Legitimate opening steps that genuinely advance the task."""
    steps = []
    for _ in range(k):
        ti, to = _tokens(rng)
        steps.append(tool(rng.choice(domain["tools"]), domain["args"](rng),
                          result="ok", progress=True, tokens_in=ti, tokens_out=to))
    return steps


# --------------------------------------------------------------------------
# Positive generators — planted failures with ground truth by construction
# --------------------------------------------------------------------------
def gen_loop(rng: random.Random, idx: int) -> Scenario:
    d = rng.choice(DOMAINS)
    prefix = rng.randint(1, 3)
    loop_len = rng.randint(4, 10)
    interleave = rng.random() < 0.45          # narrate between repeats?
    tool_name = rng.choice(d["tools"])
    args = d["args"](rng)

    steps = _healthy_prefix(rng, d, prefix)
    onset = len(steps)
    for _ in range(loop_len):
        ti, to = _tokens(rng)
        steps.append(tool(tool_name, dict(args), result="0 results / no change",
                          progress=False, tokens_in=ti, tokens_out=to))
        if interleave:
            steps.append(_noise_step(rng, d))

    return Scenario(
        id=f"gen_loop_{idx:04d}",
        title=f"Generated loop ({d['key']}, len={loop_len})",
        family="loop",
        goal=d["goal"],
        steps=steps,
        description="Planted infinite tool loop with identical arguments.",
        label=Label(should_trip=True, detector="loop", onset_index=onset,
                    detect_by_index=onset + 5),
        failing_tool=tool_name,
        recovery_branch=_recovery_steps(rng, d),
    )


def gen_loop_semantic(rng: random.Random, idx: int) -> Scenario:
    """Same intent, cosmetically different args — defeats exact hashing."""
    d = rng.choice(DOMAINS)
    prefix = rng.randint(1, 2)
    loop_len = rng.randint(5, 9)
    tool_name = rng.choice(d["tools"])
    base = d["args"](rng)
    key = list(base.keys())[0]
    val = str(base[key])
    mutations = [lambda s: s, lambda s: s + " ", lambda s: " " + s, lambda s: s.upper(),
                 lambda s: s.replace(" ", "  "), lambda s: s + "?", lambda s: s + ",",
                 lambda s: s.capitalize()]

    steps = _healthy_prefix(rng, d, prefix)
    onset = len(steps)
    for i in range(loop_len):
        ti, to = _tokens(rng)
        mutated = dict(base)
        mutated[key] = mutations[i % len(mutations)](val)
        steps.append(tool(tool_name, mutated, result="0 results",
                          progress=False, tokens_in=ti, tokens_out=to))

    return Scenario(
        id=f"gen_loopsem_{idx:04d}",
        title=f"Generated semantic loop ({d['key']})",
        family="loop",
        goal=d["goal"],
        steps=steps,
        description="Semantically identical calls with cosmetically varied arguments.",
        label=Label(should_trip=True, detector="loop", onset_index=onset,
                    detect_by_index=onset + 6, known_gap=True,
                    note="KNOWN GAP: exact arg-hash matching cannot see through arg noise."),
        failing_tool=tool_name,
        recovery_branch=_recovery_steps(rng, d),
    )


def gen_drift(rng: random.Random, idx: int, abrupt: bool = True) -> Scenario:
    """Drift away from the objective — abrupt (easy) or gradual (hard)."""
    d = rng.choice(DOMAINS)
    on_n = rng.randint(1, 3)
    off_n = rng.randint(3, 6)

    steps: list[StepSpec] = []
    for i in range(on_n):
        ti, to = _tokens(rng)
        steps.append(think(d["on_topic"][i % len(d["on_topic"])], tokens_in=ti, tokens_out=to))
    onset = len(steps)

    if abrupt:
        body = [d["off_topic"][i % len(d["off_topic"])] for i in range(off_n)]
    else:
        # Gradual: ride the bridge sentences first (goal vocabulary retained,
        # intent already sliding), only then go fully off-topic.
        bridges = SUBTLE_BRIDGES[d["key"]]
        body = [bridges[i % len(bridges)] for i in range(min(off_n, 3))]
        body += [d["off_topic"][i % len(d["off_topic"])] for i in range(off_n - len(body))]

    for text in body:
        ti, to = _tokens(rng)
        steps.append(think(text, tokens_in=ti, tokens_out=to))

    kind = "abrupt" if abrupt else "gradual"
    return Scenario(
        id=f"gen_drift{'' if abrupt else 'sub'}_{idx:04d}",
        title=f"Generated {kind} drift ({d['key']})",
        family="drift",
        goal=d["goal"],
        steps=steps,
        description=f"Agent slides off the objective ({kind}).",
        label=Label(should_trip=True, detector="drift", onset_index=onset,
                    detect_by_index=onset + 4,
                    note="" if abrupt else
                         ("Gradual drift retains goal vocabulary; lexical similarity "
                          "overlaps the legitimate-paraphrase band.")),
        recovery_branch=_recovery_steps(rng, d),
    )


def gen_drift_subtle(rng: random.Random, idx: int) -> Scenario:
    """The hard drift case — the one a lexical threshold cannot separate."""
    return gen_drift(rng, idx, abrupt=False)


def gen_stall(rng: random.Random, idx: int) -> Scenario:
    """Busy with varied tools, but the working state never advances."""
    d = rng.choice(DOMAINS)
    prefix = rng.randint(1, 2)
    stall_len = rng.randint(7, 12)

    steps = _healthy_prefix(rng, d, prefix)
    onset = len(steps)
    probe_tools = ["stat_file", "list_dirs", "read_env", "check_perms", "lookup_config"]
    for i in range(stall_len):
        ti, to = _tokens(rng)
        steps.append(tool(probe_tools[i % len(probe_tools)],
                          {"path": f"/etc/probe_{i}", "attempt": i},
                          result="not found", progress=False, tokens_in=ti, tokens_out=to))

    return Scenario(
        id=f"gen_stall_{idx:04d}",
        title=f"Generated stall ({d['key']})",
        family="progress",
        goal=d["goal"],
        steps=steps,
        description="Varied tool activity with zero state advance — a logic trap.",
        label=Label(should_trip=True, detector="progress", onset_index=onset,
                    detect_by_index=onset + 8),
        recovery_branch=_recovery_steps(rng, d),
    )


def gen_spend(rng: random.Random, idx: int) -> Scenario:
    d = rng.choice(DOMAINS)
    burst = rng.random() < 0.5
    n = rng.randint(9, 14)

    if burst:
        steps = []
        for i in range(n):
            heavy = i >= 2
            ti = rng.randint(4800, 6800) if heavy else rng.randint(700, 1100)
            to = rng.randint(800, 1100) if heavy else rng.randint(150, 260)
            steps.append(think(d["on_topic"][i % len(d["on_topic"])],
                               tokens_in=ti, tokens_out=to))
        cfg = {"max_tokens": 400_000, "burst_window": 5, "burst_tokens": 24_000,
               "loop_threshold": 99, "stall_patience": 99, "drift_threshold": 0.0}
        onset = 2
    else:
        steps = []
        for i in range(n):
            steps.append(tool(rng.choice(d["tools"]), {"page": i}, result=f"page {i}",
                              progress=True, tokens_in=rng.randint(1600, 2000),
                              tokens_out=rng.randint(350, 500)))
        cfg = {"max_tokens": 20_000, "loop_threshold": 99, "stall_patience": 99,
               "drift_threshold": 0.0}
        onset = 0

    return Scenario(
        id=f"gen_spend_{idx:04d}",
        title=f"Generated spend {'burst' if burst else 'ceiling'} ({d['key']})",
        family="spend",
        goal=d["goal"],
        steps=steps,
        config=cfg,
        description="Budget breach: cumulative ceiling or burn-rate spike.",
        label=Label(should_trip=True, detector="spend", onset_index=onset,
                    detect_by_index=onset + 10),
        recovery_branch=_recovery_steps(rng, d, n=2),
    )


# --------------------------------------------------------------------------
# Hard-negative generators — healthy runs that look like failures
# --------------------------------------------------------------------------
def gen_benign_retry(rng: random.Random, idx: int) -> Scenario:
    """Transient failure, a couple of retries, then success. Healthy."""
    d = rng.choice(DOMAINS)
    retries = rng.randint(1, 3)
    tool_name = rng.choice(d["tools"])
    args = d["args"](rng)

    steps = _healthy_prefix(rng, d, 1)
    for _ in range(retries):
        ti, to = _tokens(rng)
        steps.append(tool(tool_name, dict(args), result="HTTP 503 service unavailable",
                          progress=False, tokens_in=ti, tokens_out=to))
    ti, to = _tokens(rng)
    steps.append(tool(tool_name, dict(args), result="succeeded", progress=True,
                      tokens_in=ti, tokens_out=to))
    steps.extend(_healthy_prefix(rng, d, 1))

    return Scenario(
        id=f"gen_retry_{idx:04d}",
        title=f"Generated retry-then-success ({d['key']}, {retries} retries)",
        family="benign",
        goal=d["goal"],
        steps=steps,
        description="Retrying a flaky endpoint then succeeding is correct behaviour.",
        label=Label(should_trip=False, note="Retry with eventual success is healthy."),
    )


def gen_benign_polling(rng: random.Random, idx: int) -> Scenario:
    """Identical calls, but the world changes and the task advances."""
    d = rng.choice(DOMAINS)
    polls = rng.randint(4, 9)
    steps = _healthy_prefix(rng, d, 1)
    for i in range(polls):
        ti, to = _tokens(rng)
        pct = int((i + 1) / polls * 100)
        steps.append(tool("check_job", {"id": "job-42"},
                          result=f"status: RUNNING {pct}%", progress=True,
                          tokens_in=ti, tokens_out=to))
    return Scenario(
        id=f"gen_poll_{idx:04d}",
        title=f"Generated healthy polling ({d['key']}, {polls} polls)",
        family="benign",
        goal=d["goal"],
        steps=steps,
        description="Repeated identical calls that genuinely make progress.",
        label=Label(should_trip=False, note="Polling with advancing state is healthy."),
    )


def gen_benign_paraphrase(rng: random.Random, idx: int) -> Scenario:
    """Same objective, different vocabulary. The sharpest test of similarity thresholds."""
    d = rng.choice(DOMAINS)
    n = rng.randint(3, 6)
    steps = []
    for i in range(n):
        ti, to = _tokens(rng)
        steps.append(think(d["paraphrases"][i % len(d["paraphrases"])],
                           tokens_in=ti, tokens_out=to))
    return Scenario(
        id=f"gen_para_{idx:04d}",
        title=f"Generated paraphrased objective ({d['key']})",
        family="benign",
        goal=d["goal"],
        steps=steps,
        description="Restating the same goal in different words is not drift.",
        label=Label(should_trip=False,
                    note="Semantically on-task; any trip is a vocabulary artefact."),
    )


def gen_benign_subgoal(rng: random.Random, idx: int) -> Scenario:
    """A genuine sub-goal detour that serves the parent objective."""
    d = rng.choice(DOMAINS)
    steps = []
    ti, to = _tokens(rng)
    steps.append(think(d["on_topic"][0], tokens_in=ti, tokens_out=to))
    for _ in range(rng.randint(2, 4)):
        ti, to = _tokens(rng)
        steps.append(think(
            f"To finish the task I first need to resolve a prerequisite: "
            f"{d['on_topic'][rng.randrange(len(d['on_topic']))].lower()}",
            tokens_in=ti, tokens_out=to))
    ti, to = _tokens(rng)
    steps.append(think(d["on_topic"][-1], tokens_in=ti, tokens_out=to, progress=True))
    return Scenario(
        id=f"gen_subgoal_{idx:04d}",
        title=f"Generated sub-goal detour ({d['key']})",
        family="benign",
        goal=d["goal"],
        steps=steps,
        description="Sub-goals serve the parent objective and are not drift.",
        label=Label(should_trip=False),
    )


def gen_benign_breadth(rng: random.Random, idx: int) -> Scenario:
    """Wide tool usage with slow but genuine progress."""
    d = rng.choice(DOMAINS)
    n = rng.randint(6, 11)
    steps = []
    for i in range(n):
        ti, to = _tokens(rng)
        steps.append(tool(rng.choice(d["tools"]), d["args"](rng),
                          result=f"partial finding {i}", progress=(i % 2 == 0),
                          tokens_in=ti, tokens_out=to))
    return Scenario(
        id=f"gen_breadth_{idx:04d}",
        title=f"Generated breadth-first work ({d['key']})",
        family="benign",
        goal=d["goal"],
        steps=steps,
        description="Exploration with accruing findings is not stalling.",
        label=Label(should_trip=False),
    )


def gen_benign_expensive(rng: random.Random, idx: int) -> Scenario:
    """Legitimately expensive work that must not be cut off."""
    d = rng.choice(DOMAINS)
    n = rng.randint(6, 10)
    steps = []
    for i in range(n):
        steps.append(tool(rng.choice(d["tools"]), d["args"](rng),
                          result=f"batch {i} done", progress=True,
                          tokens_in=rng.randint(2800, 3800),
                          tokens_out=rng.randint(600, 900)))
    return Scenario(
        id=f"gen_expensive_{idx:04d}",
        title=f"Generated expensive-but-healthy run ({d['key']})",
        family="benign",
        goal=d["goal"],
        steps=steps,
        config={"max_tokens": 200_000, "burst_window": 5, "burst_tokens": 60_000},
        description="High spend with steady progress is healthy.",
        label=Label(should_trip=False),
    )


def gen_benign_short(rng: random.Random, idx: int) -> Scenario:
    d = rng.choice(DOMAINS)
    steps = _healthy_prefix(rng, d, rng.randint(2, 4))
    return Scenario(
        id=f"gen_short_{idx:04d}",
        title=f"Generated short clean run ({d['key']})",
        family="benign",
        goal=d["goal"],
        steps=steps,
        description="The simplest healthy trajectory.",
        label=Label(should_trip=False),
    )


POSITIVE_GENERATORS = [gen_loop, gen_loop_semantic, gen_drift, gen_drift_subtle,
                       gen_stall, gen_spend]
NEGATIVE_GENERATORS = [gen_benign_retry, gen_benign_polling, gen_benign_paraphrase,
                       gen_benign_subgoal, gen_benign_breadth, gen_benign_expensive,
                       gen_benign_short]


def generate_suite(n_per_generator: int = 40, seed: int = 20260812,
                   include: Optional[set[str]] = None) -> list[Scenario]:
    """Build a large, balanced, reproducible scenario suite.

    ``n_per_generator`` cases from each of the 5 positive and 7 negative
    generators. The default of 40 yields 480 scenarios (200 positive,
    280 negative) — enough to put the confidence intervals on the headline
    metrics into a range where they actually mean something.
    """
    rng = random.Random(seed)
    out: list[Scenario] = []
    for gen in POSITIVE_GENERATORS + NEGATIVE_GENERATORS:
        if include and gen.__name__ not in include:
            continue
        for i in range(n_per_generator):
            out.append(gen(rng, i))
    return out
