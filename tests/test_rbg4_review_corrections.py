"""CPU-only regression checks; never imports Modal or loads model weights."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jspace_policy.budget import admit_run, estimate_cost

ROOT = Path(__file__).resolve().parents[1]


def test_multigpu_cost_scales_only_gpu_and_admission(tmp_path):
    one = estimate_cost('A100-80GB', 3600, cpu_cores=16, memory_gib=192)
    four = estimate_cost('A100-80GB', 3600, gpu_count=4, cpu_cores=16, memory_gib=192)
    assert four.gpu_usd == pytest.approx(4 * one.gpu_usd)
    assert four.cpu_usd == one.cpu_usd
    assert four.memory_usd == one.memory_usd
    assert four.buffered_usd == pytest.approx(14.7391488)
    admit_run(tmp_path / 'empty.jsonl', four, study_limit_usd=20)
    with pytest.raises(RuntimeError):
        admit_run(tmp_path / 'empty.jsonl', four, study_limit_usd=14)


@pytest.mark.parametrize('count', [0, -1, 1.5, True])
def test_invalid_gpu_count(count):
    with pytest.raises(ValueError, match='positive integer'):
        estimate_cost('A100-80GB', 1, gpu_count=count)


def test_glm_measured_cost_matches_archived_ledger():
    root = ROOT / 'results/rbg4_multimodel'
    payload = json.loads((root / 'raw/behavior_glm45-air.json').read_text())
    ledger = [json.loads(line) for line in (root / 'cost_ledger.jsonl').read_text().splitlines()]
    row = next(x for x in ledger if x['run_id'] == payload['metadata']['run_id'])
    actual = estimate_cost('A100-80GB', payload['metadata']['elapsed_seconds'], gpu_count=4,
                           cpu_cores=16, memory_gib=192)
    for field in ('gpu_usd', 'cpu_usd', 'memory_usd', 'subtotal_usd', 'buffered_usd'):
        assert getattr(actual, field) == pytest.approx(row[field])


@pytest.mark.parametrize('gpu_count', [1, 4])
@pytest.mark.parametrize('fallback', [False, True])
def test_loader_selects_placement_before_one_successful_load(monkeypatch, gpu_count, fallback):
    import sys
    tree = ast.parse((ROOT / 'modal_rbg4_multimodel.py').read_text())
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_load_model')
    model = Mock()
    model.cuda.return_value = model
    primary = Mock()
    secondary = Mock()
    primary.from_pretrained.side_effect = ValueError('fallback') if fallback else None
    primary.from_pretrained.return_value = model
    secondary.from_pretrained.return_value = model
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(bfloat16='bf16'))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=primary, AutoModelForImageTextToText=secondary))
    scope = {'Any': object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<loader>', 'exec'), scope)
    assert scope['_load_model']({'model_id':'test', 'revision':'pin', 'gpu_count':gpu_count}) is model
    primary.from_pretrained.assert_called_once()
    assert secondary.from_pretrained.call_count == int(fallback)
    for call in [primary.from_pretrained.call_args] + ([secondary.from_pretrained.call_args] if fallback else []):
        assert call.kwargs.get('device_map') == ('auto' if gpu_count > 1 else None)
    assert model.cuda.call_count == int(gpu_count == 1)
    model.eval.assert_called_once()


def test_jlens_inference_clusters_targets_by_source_prompt():
    from collections import defaultdict
    from fractions import Fraction
    import importlib.util
    spec = importlib.util.spec_from_file_location('analysis', ROOT / 'scripts/analyze_v5_inverse_evidence.py')
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    rows = json.loads((ROOT / 'results/rbg4_multimodel/jlens_local_test.json').read_text())['rows']
    clusters = defaultdict(int)
    for row in rows:
        if row['baseline_correct']:
            clusters[row['prompt']] += int(row['average_lens_top1_is_target']) - int(row['prompt_local_top1_is_target'])
    assert len(clusters) == 12
    assert sum(value != 0 for value in clusters.values()) == 9
    assert analysis.exact_sign_flip_p([Fraction(value) for value in clusters.values()]) == 0.00390625
