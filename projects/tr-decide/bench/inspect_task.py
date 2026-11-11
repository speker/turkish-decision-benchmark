"""inspect-ai tasks with the same dataset fields, solver and scorer as hub/eval.yaml, read from the local hub folder.

    inspect eval bench/inspect_task.py@written --model hf/Qwen/Qwen2.5-0.5B-Instruct --limit 40 --seed 0

Once the dataset is on the Hub, `inspect eval hf/<namespace>/<name>` runs the same tasks straight from eval.yaml.
"""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

HUB = Path(__file__).resolve().parents[1] / "hub"


def bench_task(config: str) -> Task:
    fields = FieldSpec(input="prompt", target="answer", choices="options", id="id")
    return Task(dataset=hf_dataset(str(HUB), name=config, split="test", sample_fields=fields),
                solver=multiple_choice(), scorer=choice())


@task
def programmatic() -> Task:
    return bench_task("programmatic")


@task
def written() -> Task:
    return bench_task("written")
