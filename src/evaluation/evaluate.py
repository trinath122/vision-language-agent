"""
Evaluation using RAGAS and DeepEval across standard + adversarial inputs.
Checks robustness, faithfulness, and fairness of the VLM agent.
"""
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    BiasMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase
from datasets import Dataset
import mlflow
from typing import Any


def run_ragas_eval(eval_dataset: Dataset) -> dict:
    """
    Evaluate with RAGAS metrics: faithfulness, answer relevancy, context precision.
    eval_dataset must have: question, answer, contexts, ground_truth columns.
    """
    result = ragas_evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    scores = result.to_pandas().mean().to_dict()
    print("RAGAS scores:", scores)
    return scores


def run_deepeval(test_cases: list[dict]) -> dict:
    """
    Evaluate with DeepEval: relevancy, faithfulness, bias, toxicity.
    test_cases: list of {input, actual_output, expected_output, context}
    """
    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        BiasMetric(threshold=0.5),
        ToxicityMetric(threshold=0.5),
    ]

    cases = [
        LLMTestCase(
            input=tc["input"],
            actual_output=tc["actual_output"],
            expected_output=tc.get("expected_output", ""),
            context=tc.get("context", []),
        )
        for tc in test_cases
    ]

    results = deepeval_evaluate(cases, metrics)
    return results


def run_adversarial_eval(model, adversarial_inputs: list[dict]) -> dict:
    """
    Test model robustness against adversarial inputs:
    - misleading image-text pairs
    - out-of-distribution images
    - prompt injection attempts
    """
    results = {"passed": 0, "failed": 0, "details": []}

    for sample in adversarial_inputs:
        response = model.generate(
            images=sample.get("image"),
            prompt=sample["prompt"],
        )
        passed = sample["expected_behavior"](response)
        results["passed" if passed else "failed"] += 1
        results["details"].append({
            "prompt": sample["prompt"],
            "response": response,
            "passed": passed,
        })

    results["pass_rate"] = results["passed"] / len(adversarial_inputs)
    print(f"Adversarial pass rate: {results['pass_rate']:.2%}")
    return results


def full_evaluation(model, eval_dataset: Dataset, test_cases: list, adversarial_inputs: list):
    """Run all evaluations and log results to MLflow."""
    mlflow.set_experiment("vision-lang-evaluation")
    with mlflow.start_run():
        ragas_scores = run_ragas_eval(eval_dataset)
        mlflow.log_metrics({f"ragas_{k}": v for k, v in ragas_scores.items()})

        adversarial_results = run_adversarial_eval(model, adversarial_inputs)
        mlflow.log_metric("adversarial_pass_rate", adversarial_results["pass_rate"])

        run_deepeval(test_cases)
