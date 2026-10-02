"""The LLM front end, tested with scripted and recorded replies (no API calls)."""
import os

import pytest

import stochlift as sl
from conftest import farmer_scenarios
from stochlift.llm import build_prompt, parse_spec, propose_spec, validate, variable_groups

GOOD = """Here is my proposal.
```yaml
first_stage: ["acres_*"]
uncertain: [yield]
columns: {}
scenarios: {method: empirical}
holdout: 0
notes: Land is planted before the weather is known.
```"""


def test_prompt_contains_what_the_model_needs(farmer):
    prompt = build_prompt(farmer.build_model, farmer.DATA)
    assert "def build_model(data)" in prompt
    assert "acres_*: 3 continuous" in prompt and "sell_*" in prompt
    assert "- yield: 3 constraint coefficients" in prompt
    assert "- land: 1 right-hand sides" in prompt


def test_variable_groups(facility):
    groups = variable_groups(sl.to_linear_model(facility.build_model(facility.DATA)))
    assert groups["open[*]"]["count"] == 5 and groups["open[*]"]["integer"] == 5
    assert groups["ship[*]"]["count"] == 40 and groups["short[*]"]["count"] == 8


def test_parse_and_validate(farmer):
    spec = parse_spec(GOOD)
    assert spec.first_stage == ["acres_*"] and spec.uncertain == ["yield"]
    assert validate(farmer.build_model, farmer.DATA, spec) == []
    with pytest.raises(ValueError):
        parse_spec("I think acreage is first stage.")


def test_validation_catches_bad_proposals(farmer):
    bad = sl.Spec(first_stage=["plant_*"], uncertain=["rainfall"])
    problems = validate(farmer.build_model, farmer.DATA, bad)
    assert any("matches no variable" in p for p in problems)
    assert any("rainfall" in p for p in problems)
    everything = sl.Spec(first_stage=["*"], uncertain=["yield"])
    assert any("every variable is first-stage" in p
               for p in validate(farmer.build_model, farmer.DATA, everything))


def test_feedback_loop_repairs_a_bad_first_answer(farmer):
    replies = ["```yaml\nfirst_stage: [\"plant_*\"]\nuncertain: [rainfall]\n```", GOOD]
    prompts = []

    def scripted(prompt):
        prompts.append(prompt)
        return replies[len(prompts) - 1]

    spec = propose_spec(farmer.build_model, farmer.DATA, llm=scripted)
    assert spec.first_stage == ["acres_*"] and len(prompts) == 2
    assert "Problems with your previous answer" in prompts[1] and "plant_*" in prompts[1]

    with pytest.raises(ValueError, match="did not produce a usable spec"):
        propose_spec(farmer.build_model, farmer.DATA, llm=lambda p: "no idea", max_rounds=2)


def test_recorded_reply_farmer(farmer, fixtures):
    reply = open(os.path.join(fixtures, "farmer_reply.txt")).read()
    study = sl.lift(farmer.build_model, farmer.DATA, llm=lambda prompt: reply,
                    scenarios=farmer_scenarios(farmer.DATA))
    r = study.solve()
    assert -r.rp == pytest.approx(108390, abs=0.5) and r.vss == pytest.approx(1150, abs=0.5)


def test_recorded_reply_facility(facility, facility_history, fixtures):
    reply = open(os.path.join(fixtures, "facility_reply.txt")).read()
    proposed = propose_spec(facility.build_model, facility.DATA, history=facility_history,
                            llm=lambda prompt: reply)
    by_hand = sl.Spec.load(os.path.join(os.path.dirname(facility_history), "uncertainty.yaml"))
    a = sl.Study(facility.build_model, facility.DATA, proposed, history=facility_history)
    b = sl.Study(facility.build_model, facility.DATA, by_hand, history=facility_history)
    assert a.scenarios.names == b.scenarios.names and len(a.test) == len(b.test)
    assert a.solve().rp == pytest.approx(b.solve().rp)
