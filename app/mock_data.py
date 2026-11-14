"""Synthetic policies, claim history and demo scenarios.

Everything here is invented for the demo (the rulebook forbids real customer data).
The *hazard evidence* the agent gathers (earthquakes, rainfall, river flow, news) is real
public data; only the insurer-internal records below are mock.
"""

POLICIES = {
    "POL-TR-1001": {
        "holder": "Synthetic Holder A",
        "insured_location": "Kahramanmaras, Turkey",
        "policy_start": "2023-02-07",
        "perils_covered": ["Earthquake", "Fire"],
        "sum_insured_usd": 150000,
        "deductible_usd": 2000,
        "waiting_period_days": 15,
        "exclusions": [
            "Loss caused by an event that occurred before the policy start date",
            "Intentional damage",
        ],
    },
    "POL-TR-0457": {
        "holder": "Synthetic Holder B",
        "insured_location": "Gaziantep, Turkey",
        "policy_start": "2022-06-01",
        "perils_covered": ["Earthquake", "Fire"],
        "sum_insured_usd": 120000,
        "deductible_usd": 2000,
        "waiting_period_days": 15,
        "exclusions": [
            "Loss caused by an event that occurred before the policy start date",
            "Intentional damage",
        ],
    },
    "POL-IN-2210": {
        "holder": "Synthetic Holder C",
        "insured_location": "Indiranagar, Bengaluru, India",
        "policy_start": "2021-04-12",
        "perils_covered": ["Flood", "Fire", "Storm"],
        "sum_insured_usd": 40000,
        "deductible_usd": 500,
        "waiting_period_days": 15,
        "exclusions": [
            "Loss caused by an event that occurred before the policy start date",
            "Gradual seepage or poor maintenance",
        ],
    },
}

CLAIM_HISTORY = {
    "POL-TR-1001": [],
    "POL-TR-0457": [],
    "POL-IN-2210": [
        {"claim_id": "CLM-1388", "date": "2022-09-06", "peril": "Flood", "paid_usd": 9500},
        {"claim_id": "CLM-1710", "date": "2022-12-19", "peril": "Flood", "paid_usd": 11200},
    ],
}

# Three demo scenarios. Two share the SAME real disaster so the agent is shown to be
# discriminating, not just reacting to the word "earthquake".
SCENARIOS = [
    {
        "id": "late-buyer",
        "title": "Bought cover after the quake",
        "subtitle": "Turkey, Feb 2023. Policy starts the day after the M7.8 earthquake.",
        "claim": {
            "policy_id": "POL-TR-1001",
            "location": "Kahramanmaras, Turkey",
            "claim_type": "Earthquake",
            "loss_date": "2023-02-08",
            "claimed_amount_usd": 85000,
            "description": "Cracked load-bearing walls and partial collapse of the ground floor following the earthquake. Requesting full payout.",
        },
    },
    {
        "id": "genuine",
        "title": "Genuine long-time customer",
        "subtitle": "Same quake. Policy in force since June 2022.",
        "claim": {
            "policy_id": "POL-TR-0457",
            "location": "Gaziantep, Turkey",
            "claim_type": "Earthquake",
            "loss_date": "2023-02-06",
            "claimed_amount_usd": 42000,
            "description": "Partial wall collapse and a cracked foundation during the 6 February earthquake.",
        },
    },
    {
        "id": "no-disaster",
        "title": "Flood claim, no flood",
        "subtitle": "Bengaluru, Mar 2023. Customer reports a flood; two prior flood claims.",
        "claim": {
            "policy_id": "POL-IN-2210",
            "location": "Indiranagar, Bengaluru, India",
            "claim_type": "Flood",
            "loss_date": "2023-03-15",
            "claimed_amount_usd": 18000,
            "description": "Ground-floor shop flooded to knee height overnight; stock and fittings destroyed.",
        },
    },
]

SCENARIO_BY_ID = {s["id"]: s for s in SCENARIOS}
