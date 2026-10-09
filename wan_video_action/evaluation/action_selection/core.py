from __future__ import annotations

from collections import defaultdict
from math import hypot, isclose, isfinite
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping, Sequence
import json


def load_rules(task: str) -> dict[str, Any]:
    return json.loads(Path(__file__).with_name(f'{task}.json').read_text())


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x, y = _number(value[0]), _number(value[1])
    return None if x is None or y is None else (x, y)


def _valid(value: Any, kind: str) -> bool:
    if kind == 'point':
        return _point(value) is not None
    if kind == 'scalar':
        return _number(value) is not None
    return isinstance(value, (bool, int)) and value in (0, 1)


def _aggregate(values: Sequence[Any], kind: str) -> Any:
    values = [value for value in values if _valid(value, kind)]
    if not values:
        return None
    if kind == 'point':
        return [median(value[axis] for value in values) for axis in (0, 1)]
    if kind == 'scalar':
        return float(median(values))
    return fmean(float(bool(value)) for value in values) >= 0.5


def _order(candidate: Mapping[str, Any]) -> float:
    value = _number(candidate.get('action_order', candidate['action_index']))
    if value is None:
        raise ValueError('Ordered action selection requires numeric action_order.')
    return value


def _distance(value: Any, target: Mapping[str, Any]) -> float:
    kind = target['kind']
    if kind == 'rectangle':
        point = _point(value)
        if point is None:
            return float('inf')
        x, y = point
        x0, x1, y0, y1 = target['region']
        return hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))
    if kind == 'interval':
        number = _number(value)
        if number is None:
            return float('inf')
        lower, upper = target['range']
        return max(lower - number, 0.0, number - upper)
    if kind == 'y_threshold':
        point = _point(value)
        if point is None:
            return float('inf')
        return max(float(target['threshold']) - point[1], 0.0)
    if not _valid(value, 'binary'):
        return float('inf')
    return 0.0 if bool(value) == bool(target['value']) else 1.0


def _reaches(value: Any, target: Mapping[str, Any]) -> bool:
    if target['kind'] == 'y_threshold':
        point = _point(value)
        return point is not None and point[1] > float(target['threshold'])
    return _distance(value, target) == 0.0


def _boundary_choice(candidates, target):
    ordered = sorted(candidates, key=_order)
    center = sum(target['range']) / 2.0
    distances = {}
    roots = []
    for left, right in zip(ordered, ordered[1:]):
        left_delta = float(left['selection_value']) - center
        right_delta = float(right['selection_value']) - center
        if not (isclose(left_delta, 0.0, abs_tol=1e-12)
                or isclose(right_delta, 0.0, abs_tol=1e-12)
                or left_delta * right_delta < 0.0):
            continue
        denominator = right_delta - left_delta
        root = ((_order(left) + _order(right)) / 2.0
                if isclose(denominator, 0.0, abs_tol=1e-12)
                else _order(left) - left_delta * (_order(right) - _order(left)) / denominator)
        roots.append(root)
        for candidate in (left, right):
            index = candidate['action_index']
            distances[index] = min(distances.get(index, float('inf')), abs(_order(candidate) - root))
    selected = min(ordered, key=lambda candidate: (
        _distance(candidate['selection_value'], target),
        0 if candidate['action_index'] in distances else 1,
        distances.get(candidate['action_index'], float('inf')),
        abs(float(candidate['selection_value']) - center),
        _order(candidate),
    ))
    return selected, {'predicted_boundary_orders': roots}


def _select(candidates, target, rules):
    available = [candidate for candidate in candidates
                 if _valid(candidate['selection_value'], rules['outcome_kind'])]
    if not available:
        return None, {'reason': 'no_prediction'}
    strategy = target['selection']
    if strategy == 'boundary_crossing':
        return _boundary_choice(available, target)
    if strategy in ('first_reaching', 'first_crossing'):
        reaching = [candidate for candidate in sorted(available, key=_order)
                    if _reaches(candidate['selection_value'], target)]
        if reaching:
            return reaching[0], {'used_fallback': False}
        if strategy == 'first_crossing':
            return None, {'reason': 'prediction_never_crosses', 'used_fallback': False}
        fallback = target.get('fallback', 'nearest')
        if fallback == 'max_y_then_highest_action':
            decimals = int(target['fallback_round_decimals'])
            return max(available, key=lambda candidate: (
                round(candidate['selection_value'][1], decimals), _order(candidate),
            )), {'used_fallback': True, 'fallback': fallback}
        return min(available, key=lambda candidate: (
            _distance(candidate['selection_value'], target), _order(candidate),
        )), {'used_fallback': True, 'fallback': fallback}
    if rules.get('nearest_tiebreak') == 'action_index_string':
        labels = rules.get('action_index_labels', {})
        tiebreak = lambda candidate: str(labels.get(str(candidate['action_index']), candidate['action_index']))
    else:
        tiebreak = _order
    return min(available, key=lambda candidate: (
        _distance(candidate['selection_value'], target), tiebreak(candidate),
    )), {}


def _group_candidates(rows, rules):
    by_action = defaultdict(list)
    for row in rows:
        by_action[int(row['action_index'])].append(row)
    candidates = []
    for index, members in sorted(by_action.items()):
        queries = [member for member in members if not member.get('is_support', False)]
        predicted = [member for member in queries
                     if _valid(member.get('predicted_value'), rules['outcome_kind'])]
        observed = [member for member in members if member.get('is_support', False)]
        source = predicted or observed
        field = 'predicted_value' if predicted else 'ground_truth_value'
        orders = {_order(member) for member in members}
        if len(orders) != 1:
            raise ValueError('Repeated action candidates have inconsistent action_order.')
        candidates.append({
            'action_index': index,
            'action_order': next(iter(orders)),
            'selection_value': _aggregate([member.get(field) for member in source], rules['outcome_kind']),
            'ground_truth_value': _aggregate([member.get('ground_truth_value') for member in (queries or members)], rules['outcome_kind']),
            'selection_source': 'model_prediction' if predicted else 'observed_support' if observed else 'missing_prediction',
        })
    if len({_order(candidate) for candidate in candidates}) != len(candidates):
        raise ValueError('Action candidates must have distinct action_order values.')
    return sorted(candidates, key=_order)


def evaluate_candidates(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        key = (str(row.get('method', 'model')), int(row['environment_index']), int(row.get('decision_index', 0)))
        groups[key].append(row)
    decisions = []
    for (method, environment, decision_index), members in sorted(groups.items()):
        candidates = _group_candidates(members, rules)
        ground_truth_complete = all(_valid(candidate['ground_truth_value'], rules['outcome_kind']) for candidate in candidates)
        prediction_complete = all(_valid(candidate['selection_value'], rules['outcome_kind']) for candidate in candidates)
        scope_eligible = True
        if rules.get('eligibility') == 'distinct_binary_outcomes':
            scope_eligible = len({bool(candidate['ground_truth_value']) for candidate in candidates
                                  if _valid(candidate['ground_truth_value'], 'binary')}) == 2
        for target in rules['targets']:
            gt_reaching = [candidate for candidate in candidates if _reaches(candidate['ground_truth_value'], target)]
            gt_first = gt_reaching[0] if gt_reaching else None
            selected, selection_details = _select(candidates, target, rules)
            selected_index = None if selected is None else selected['action_index']
            policy = target['score']
            if policy == 'exact_first_crossing':
                feasible = [gt_first] if gt_first is not None else []
            elif policy == 'minimum_reaching':
                if gt_first is None:
                    feasible = []
                else:
                    first_rank = candidates.index(gt_first)
                    feasible = [candidate for rank, candidate in enumerate(candidates)
                                if first_rank <= rank <= first_rank + int(target['max_steps_above_minimum'])
                                and _reaches(candidate['ground_truth_value'], target)]
            else:
                feasible = gt_reaching
            reached = selected is not None and _reaches(selected['ground_truth_value'], target)
            eligible = scope_eligible and ground_truth_complete and gt_first is not None
            if rules['missing_prediction_policy'] == 'exclude_incomplete':
                eligible = eligible and prediction_complete
            score = int(selected_index in {candidate['action_index'] for candidate in feasible}) if eligible else None
            decisions.append({
                'task': rules['task'], 'method': method,
                'environment_index': environment, 'decision_index': decision_index,
                'target': target['name'], 'eligible': eligible, 'score': score,
                'candidate_count': len(candidates),
                'prediction_complete': prediction_complete,
                'ground_truth_complete': ground_truth_complete,
                'scope_eligible': scope_eligible,
                'gt_reachable': gt_first is not None,
                'selected_action_index': selected_index,
                'gt_first_action_index': None if gt_first is None else gt_first['action_index'],
                'feasible_action_indices': [candidate['action_index'] for candidate in feasible],
                'selected_outcome_source': None if selected is None else selected['selection_source'],
                'selected_predicted_value': None if selected is None else selected['selection_value'],
                'selected_ground_truth_value': None if selected is None else selected['ground_truth_value'],
                'selected_action_gt_reaches': bool(reached),
                'selection_details': selection_details,
            })
    return decisions


def summarize(decisions: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]) -> dict[str, Any]:
    methods = sorted({row['method'] for row in decisions})
    results = {}
    for method in methods:
        rows = [row for row in decisions if row['method'] == method]
        usable = [row for row in rows if row['eligible']]
        per_target = {}
        for target in rules['targets']:
            subset = [row for row in usable if row['target'] == target['name']]
            per_target[target['name']] = {
                'eligible_count': len(subset),
                'score': fmean(row['score'] for row in subset) if subset else None,
                'execution_success_rate': fmean(float(row['selected_action_gt_reaches']) for row in subset) if subset else None,
            }
        if rules['aggregation'] == 'equal_targets_then_equal_environments':
            target_means = []
            for target in rules['targets']:
                environments = defaultdict(list)
                for row in usable:
                    if row['target'] == target['name']:
                        environments[row['environment_index']].append(row['score'])
                if environments:
                    target_means.append(fmean(fmean(scores) for scores in environments.values()))
            score = fmean(target_means) if target_means else None
        else:
            score = fmean(row['score'] for row in usable) if usable else None
        results[method] = {
            'score': score, 'decision_count': len(rows), 'eligible_count': len(usable),
            'gt_unreachable_count': sum(not row['gt_reachable'] for row in rows),
            'incomplete_prediction_count': sum(not row['prediction_complete'] for row in rows),
            'incomplete_ground_truth_count': sum(not row['ground_truth_complete'] for row in rows),
            'out_of_scope_count': sum(not row['scope_eligible'] for row in rows),
            'per_target': per_target,
        }
    return {'task': rules['task'], 'aggregation': rules['aggregation'], 'methods': results}
