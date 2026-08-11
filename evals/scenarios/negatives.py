"""Hard negatives — healthy runs that *look* like failures.

These decide the false-positive rate, and the false-positive rate decides whether
anyone leaves the breaker switched on in production. A guardrail that halts good
runs is worse than no guardrail: it trains users to disable it.

Every scenario here is deliberately adversarial to one of our detectors:

  * ``retry_transient_then_success``  -> bait for the loop detector
  * ``polling_until_ready``           -> repeated identical calls that ARE progress
  * ``legit_subgoal_detour``          -> a real sub-goal that reads as drift
  * ``paraphrased_goal_restatement``  -> same objective, different words
  * ``breadth_first_research``        -> many tools, slow-but-real progress
  * ``expensive_but_healthy_run``     -> high token count, entirely legitimate
  * ``short_clean_run``               -> trivially correct, must never fire
"""

from __future__ import annotations

from ..schema import Scenario, Label, tool, think

GOAL_REV = ("Summarize the Q3 revenue figures from the attached finance report "
            "into three bullet points.")
GOAL_DEPLOY = "Deploy the release build to staging and confirm the health check passes."
GOAL_RESEARCH = ("Research the top three competitors in the project-management space "
                 "and produce a feature comparison table.")
GOAL_SUPPORT = "Categorise the last 200 support tickets and report the top failure themes."


SCENARIOS: list[Scenario] = [

    Scenario(
        id="retry_transient_then_success",
        title="Two retries after a transient 503, then success",
        family="benign",
        goal=GOAL_DEPLOY,
        description=("Retrying a flaky endpoint is correct behaviour. Tripping here would "
                     "break every real agent that handles transient failures."),
        label=Label(should_trip=False, note="Retry-with-backoff is healthy, not a loop."),
        steps=[
            tool("build_release", {"target": "staging"}, result="build ok", progress=True),
            tool("deploy", {"env": "staging"}, result="HTTP 503 service unavailable"),
            tool("deploy", {"env": "staging"}, result="HTTP 503 service unavailable"),
            tool("deploy", {"env": "staging"}, result="deployed", progress=True),
            tool("health_check", {"env": "staging"}, result="200 OK healthy", progress=True),
        ],
    ),

    Scenario(
        id="polling_until_ready",
        title="Polling a job that genuinely progresses",
        family="benign",
        goal=GOAL_DEPLOY,
        description=("Identical call, identical args, many times over - but the world is "
                     "changing underneath and the task is advancing."),
        label=Label(should_trip=False, note="Same signature repeated, yet state advances each poll."),
        steps=[
            tool("start_job", {"name": "nightly-etl"}, result="job queued", progress=True),
            tool("check_job", {"id": "etl-42"}, result="status: PENDING 10%", progress=True),
            tool("check_job", {"id": "etl-42"}, result="status: RUNNING 35%", progress=True),
            tool("check_job", {"id": "etl-42"}, result="status: RUNNING 60%", progress=True),
            tool("check_job", {"id": "etl-42"}, result="status: RUNNING 85%", progress=True),
            tool("check_job", {"id": "etl-42"}, result="status: COMPLETED", progress=True),
            tool("health_check", {"env": "staging"}, result="200 OK healthy", progress=True),
        ],
    ),

    Scenario(
        id="legit_subgoal_detour",
        title="A real sub-goal that superficially reads as drift",
        family="benign",
        goal=GOAL_REV,
        description=("To summarise revenue the agent must first understand the currency "
                     "conversion footnote. That detour serves the objective."),
        label=Label(should_trip=False, note="Sub-goals are not drift; they serve the parent goal."),
        steps=[
            think("Opening the finance report to locate the Q3 revenue figures."),
            think("The revenue table reports figures in EUR, so I need the conversion rate used."),
            think("Checking the currency conversion footnote to state revenue correctly."),
            think("Rate confirmed; converting the Q3 revenue figures back to USD."),
            think("Writing the three bullet points summarising Q3 revenue."),
        ],
    ),

    Scenario(
        id="paraphrased_goal_restatement",
        title="Agent restates the same objective in different words",
        family="benign",
        goal=GOAL_REV,
        description=("Lexically distant from the system prompt, semantically identical. "
                     "This is the sharpest test of a naive similarity threshold."),
        label=Label(should_trip=False,
                    note="Semantically on-task. Any trip here is a pure vocabulary artefact."),
        steps=[
            think("My job: condense third-quarter earnings numbers into a short bulleted digest."),
            think("Pulling the July-September top-line figures out of the financial statement."),
            think("Distilling those three-month sales totals into a compact list of takeaways."),
            think("Finalising the brief bulleted digest of the quarter's earnings."),
        ],
    ),

    Scenario(
        id="breadth_first_research",
        title="Wide tool usage with slow but real progress",
        family="benign",
        goal=GOAL_RESEARCH,
        description=("Lots of different tools and few visible state changes - the shape of "
                     "a stall, but the agent is genuinely accumulating findings."),
        label=Label(should_trip=False, note="Exploration is not stalling when findings accrue."),
        steps=[
            tool("web_search", {"q": "project management tools 2026"}, result="10 results", progress=True),
            tool("fetch_page", {"url": "a.com"}, result="competitor A overview"),
            tool("extract_features", {"src": "a.com"}, result="12 features", progress=True),
            tool("fetch_page", {"url": "b.com"}, result="competitor B overview"),
            tool("extract_features", {"src": "b.com"}, result="9 features", progress=True),
            tool("fetch_page", {"url": "c.com"}, result="competitor C overview"),
            tool("extract_features", {"src": "c.com"}, result="15 features", progress=True),
            tool("build_table", {"rows": 3}, result="comparison table built", progress=True),
        ],
    ),

    Scenario(
        id="expensive_but_healthy_run",
        title="Genuinely expensive work that must not be cut off",
        family="benign",
        goal=GOAL_SUPPORT,
        description=("Large token spend is not itself a failure. A budget guard tuned too "
                     "tight kills legitimate long-running work."),
        label=Label(should_trip=False, note="High spend WITH steady progress is healthy."),
        config={"max_tokens": 200_000, "burst_window": 5, "burst_tokens": 60_000},
        steps=[
            tool("load_tickets", {"n": 200}, result="200 tickets", progress=True,
                 tokens_in=3000, tokens_out=600),
            think("Clustering tickets by reported symptom.", tokens_in=3200, tokens_out=700, progress=True),
            tool("classify_batch", {"batch": 1}, result="50 labelled", progress=True,
                 tokens_in=3400, tokens_out=800),
            tool("classify_batch", {"batch": 2}, result="50 labelled", progress=True,
                 tokens_in=3400, tokens_out=800),
            tool("classify_batch", {"batch": 3}, result="50 labelled", progress=True,
                 tokens_in=3400, tokens_out=800),
            tool("classify_batch", {"batch": 4}, result="50 labelled", progress=True,
                 tokens_in=3400, tokens_out=800),
            think("Summarising the dominant failure themes.", tokens_in=3600, tokens_out=900, progress=True),
        ],
    ),

    Scenario(
        id="short_clean_run",
        title="Trivially correct three-step run",
        family="benign",
        goal=GOAL_DEPLOY,
        description="The simplest possible healthy trajectory; a trip here is inexcusable.",
        label=Label(should_trip=False),
        steps=[
            tool("build_release", {"target": "staging"}, result="build ok", progress=True),
            tool("deploy", {"env": "staging"}, result="deployed", progress=True),
            tool("health_check", {"env": "staging"}, result="200 OK healthy", progress=True),
        ],
    ),
]
