from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbanomy.methods.land_value_modeling.ga_mc_optimizer import Evaluation, StrategicAlignmentScorer
from urbanomy.methods.agent import SingleAgentBaseline


class _StructuredLLM:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, prompt):
        text = prompt[0].content if isinstance(prompt, list) else prompt
        assert "Сценарий:" in text
        return {"score": 0.7}


class _LLM:
    schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return _StructuredLLM(schema)


def test_structured_llm_score():
    llm = _LLM()
    scorer = StrategicAlignmentScorer(llm=llm, prompt="rate")

    result = scorer.score_candidate(
        params_repaired={"x": 1},
        land_value_gain=2,
        investor_npv=3,
    )

    assert llm.schema == Evaluation.model_json_schema()
    assert result["score"] == 0.7
    assert result["reasoning"] == ""


def test_structured_single_agent_baseline_score():
    llm = _LLM()
    baseline = SingleAgentBaseline(llm=llm, output_schema=Evaluation)
    scorer = StrategicAlignmentScorer(baseline=baseline, prompt="rate")

    result = scorer.score_candidate(
        params_repaired={"x": 1},
        land_value_gain=2,
        investor_npv=3,
    )

    assert llm.schema == Evaluation.model_json_schema()
    assert result["score"] == 0.7
    assert baseline.invoke_state("Сценарий: {}")["output"] == {"score": 0.7}


if __name__ == "__main__":
    test_structured_llm_score()
    test_structured_single_agent_baseline_score()
