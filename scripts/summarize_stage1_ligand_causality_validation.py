#!/usr/bin/env python3
"""Summarize Stage-1 ligand-causality audit and diagnostic JSON outputs.

The report is intentionally post-hoc: it does not train a model or change any
checkpoint.  It combines dataset/contact-label audit results with one or more
`diagnose_stage1_prior.py` outputs and writes a single
`causality_validation_report.json` for go/no-go review.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SUBSETS = ('all_chi1', 'pocket', 'ligand_facing_apo_ca', 'apo_wrong', 'switch', 'non_contact')
CANDIDATE_LIFT_SUBSETS = ('contact', 'contact_switch', 'pocket_switch', 'switch')
CONTROL_VARIANTS = ('no_ligand', 'translated_away', 'scrambled_types', 'batch_shuffled_ligand')


def load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def parse_labeled_path(value: str) -> Tuple[str, Path]:
    if '=' not in value:
        path = Path(value)
        return path.parent.name or path.stem, path
    label, raw_path = value.split('=', 1)
    label = label.strip()
    if not label:
        raise ValueError(f'empty diagnostic label in {value!r}')
    return label, Path(raw_path)


def finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def metric_value(block: Dict[str, Any], variant: str, subset: str, field: str) -> Optional[float]:
    return finite_or_none(block.get(variant, {}).get(subset, {}).get(field))


def delta_table(block: Dict[str, Any], field: str, subsets: Iterable[str]) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for control in CONTROL_VARIANTS:
        out[control] = {}
        for subset in subsets:
            correct = metric_value(block, 'correct_ligand', subset, field)
            decoy = metric_value(block, control, subset, field)
            out[control][subset] = None if correct is None or decoy is None else correct - decoy
    return out


def raw_metric_table(block: Dict[str, Any], field: str, subsets: Iterable[str]) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for variant, subset_stats in block.items():
        out[variant] = {}
        for subset in subsets:
            out[variant][subset] = finite_or_none(subset_stats.get(subset, {}).get(field))
    return out


def posterior_contact_summary(posterior_metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for variant, subsets in posterior_metrics.items():
        out[variant] = {}
        for subset, stat in subsets.items():
            contact = stat.get('contact')
            if not isinstance(contact, dict):
                continue
            out[variant][subset] = {
                'n': contact.get('n'),
                'positives': contact.get('positives'),
                'positive_rate': finite_or_none(contact.get('positive_rate')),
                'average_precision': finite_or_none(contact.get('average_precision')),
                'auroc': finite_or_none(contact.get('auroc')),
                'ece': finite_or_none(contact.get('ece')),
            }
    return out


def summarize_diagnostic(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    metrics = data.get('metrics', {})
    candidate_metrics = data.get('candidate_rotamer_metrics', {})
    posterior_metrics = data.get('posterior_metrics', {})
    return {
        'path': str(path),
        'checkpoint': data.get('checkpoint'),
        'n_batches': data.get('n_batches'),
        'n_samples': data.get('n_samples'),
        'contact_threshold': data.get('contact_threshold'),
        'chi1_accuracy': raw_metric_table(metrics, 'chi1_acc', SUBSETS),
        'chi1_accuracy_deltas': delta_table(metrics, 'chi1_acc', SUBSETS),
        'candidate_rotamer_accuracy': raw_metric_table(candidate_metrics, 'rotamer_acc', CANDIDATE_LIFT_SUBSETS),
        'candidate_rotamer_lifts': delta_table(candidate_metrics, 'rotamer_acc', CANDIDATE_LIFT_SUBSETS),
        'contact_posterior_calibration': posterior_contact_summary(posterior_metrics),
    }


def get_nested_float(data: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return finite_or_none(current)


def pass_if(value: bool, detail: str) -> Dict[str, Any]:
    return {'pass': bool(value), 'detail': detail}


def find_main_threshold(summary: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    thresholds = summary.get('thresholds', {})
    main_key = thresholds.get('main_contact_threshold_key', '4p5')
    audit = summary.get('contact_threshold_audit', {})
    if main_key in audit:
        return main_key, audit[main_key]
    for key, value in audit.items():
        if finite_or_none(value.get('threshold')) == finite_or_none(thresholds.get('atom_contact_threshold')):
            return key, value
    return main_key, {}


def audit_checks(audit_data: Dict[str, Any], translated_contact_max_fraction: float) -> Dict[str, Any]:
    summary = audit_data.get('summary', {})
    main_key, main_threshold = find_main_threshold(summary)
    switch_n = finite_or_none(summary.get('switch')) or 0.0
    ca_contact_switch_n = finite_or_none(summary.get('ca_contact_switch')) or 0.0
    pocket_switch_n = finite_or_none(summary.get('pocket_switch')) or 0.0
    fk_available = finite_or_none(summary.get('fk_atom14_available')) or 0.0
    fk_contact = finite_or_none(main_threshold.get('fk_atom14_contact')) or 0.0
    ca_fk_overlap = finite_or_none(main_threshold.get('ca_fk_overlap')) or 0.0
    translated_fk_fraction = finite_or_none(main_threshold.get('translated_fk_fraction_of_original_fk'))
    translated_ok = translated_fk_fraction is not None and translated_fk_fraction <= translated_contact_max_fraction
    checks = {
        'switch_subset_nonempty': pass_if(switch_n > 0, f'switch residues={int(switch_n)}'),
        'ca_contact_switch_subset_nonempty': pass_if(ca_contact_switch_n > 0, f'CA-contact switch residues={int(ca_contact_switch_n)}'),
        'pocket_switch_subset_nonempty': pass_if(pocket_switch_n > 0, f'pocket-switch residues={int(pocket_switch_n)}'),
        'fk_atom14_available': pass_if(fk_available > 0, f'FK-derived atom14 available samples={int(fk_available)}'),
        'fk_contact_nonempty': pass_if(fk_contact > 0, f'FK atom14 contacts at {main_key}={int(fk_contact)}'),
        'ca_fk_overlap_nonempty': pass_if(ca_fk_overlap > 0, f'CA/FK contact overlap at {main_key}={int(ca_fk_overlap)}'),
        'translated_away_contact_depleted': pass_if(
            translated_ok,
            f'translated FK/original FK contact fraction={translated_fk_fraction}',
        ),
        'raw_atom14_not_required_for_contact_supervision': pass_if(
            True,
            'raw atom14 may be absent; FK-derived atom14 contact labels are audited separately',
        ),
    }
    checks['all_pass'] = all(item['pass'] for item in checks.values())
    checks['main_threshold_key'] = main_key
    return checks


def diagnostic_checks(
    diagnostic: Dict[str, Any],
    translated_lift_min: float,
    chemistry_flat_abs_max: float,
) -> Dict[str, Any]:
    deltas = diagnostic.get('chi1_accuracy_deltas', {})
    candidate_lifts = diagnostic.get('candidate_rotamer_lifts', {})
    translated_switch = get_nested_float(deltas, ('translated_away', 'switch'))
    translated_pocket = get_nested_float(deltas, ('translated_away', 'pocket'))
    translated_contact = get_nested_float(deltas, ('translated_away', 'ligand_facing_apo_ca'))
    proximity_candidates = [v for v in (translated_switch, translated_pocket, translated_contact) if v is not None]
    chi1_proximity_signal = bool(proximity_candidates and max(proximity_candidates) >= translated_lift_min)

    translated_candidate_lifts = {
        subset: get_nested_float(candidate_lifts, ('translated_away', subset))
        for subset in CANDIDATE_LIFT_SUBSETS
    }
    candidate_proximity_values = [v for v in translated_candidate_lifts.values() if v is not None]
    candidate_proximity_signal = bool(
        candidate_proximity_values and max(candidate_proximity_values) >= translated_lift_min
    )
    proximity_signal = bool(chi1_proximity_signal or candidate_proximity_signal)

    chemistry_controls = {
        control: get_nested_float(deltas, (control, 'switch'))
        for control in ('no_ligand', 'scrambled_types', 'batch_shuffled_ligand')
    }
    chemistry_flat = all(
        value is not None and abs(value) <= chemistry_flat_abs_max
        for value in chemistry_controls.values()
    )
    chemistry_positive = all(
        value is not None and value > chemistry_flat_abs_max
        for value in chemistry_controls.values()
    )
    candidate_chemistry_controls = {
        control: {
            subset: get_nested_float(candidate_lifts, (control, subset))
            for subset in CANDIDATE_LIFT_SUBSETS
        }
        for control in ('no_ligand', 'scrambled_types', 'batch_shuffled_ligand')
    }
    candidate_chemistry_switch = {
        control: values.get('switch')
        for control, values in candidate_chemistry_controls.items()
    }
    candidate_chemistry_positive = all(
        value is not None and value > chemistry_flat_abs_max
        for value in candidate_chemistry_switch.values()
    )
    strict_chemistry_positive = bool(chemistry_positive and candidate_chemistry_positive)
    return {
        'proximity_signal': proximity_signal,
        'chi1_proximity_signal': chi1_proximity_signal,
        'candidate_proximity_signal': candidate_proximity_signal,
        'translated_switch_lift': translated_switch,
        'translated_pocket_lift': translated_pocket,
        'translated_ligand_facing_lift': translated_contact,
        'translated_candidate_lifts': translated_candidate_lifts,
        'chemistry_control_switch_lifts': chemistry_controls,
        'candidate_chemistry_control_lifts': candidate_chemistry_controls,
        'chemistry_controls_flat': chemistry_flat,
        'chemistry_controls_positive': chemistry_positive,
        'candidate_chemistry_controls_positive': candidate_chemistry_positive,
        'paper_ready_ligand_causal': bool(proximity_signal and strict_chemistry_positive),
        'current_failure_pattern': bool(proximity_signal and not strict_chemistry_positive),
    }


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    audit_path = Path(args.audit_json)
    if not audit_path.exists():
        raise FileNotFoundError(f'audit JSON not found: {audit_path}')
    audit_data = load_json(audit_path)

    diagnostic_args = list(args.diagnostic or [])
    if args.baseline_json:
        diagnostic_args.append(f'baseline={args.baseline_json}')
    if args.translated_rerank_json:
        diagnostic_args.append(f'translated_rerank={args.translated_rerank_json}')
    if args.shuffled_rerank_json:
        diagnostic_args.append(f'shuffled_rerank={args.shuffled_rerank_json}')
    if not diagnostic_args:
        raise ValueError('provide at least one --diagnostic label=path or one explicit diagnostic JSON argument')

    diagnostics: Dict[str, Any] = {}
    diagnostic_decisions: Dict[str, Any] = {}
    for item in diagnostic_args:
        label, path = parse_labeled_path(item)
        if not path.exists():
            raise FileNotFoundError(f'diagnostic JSON not found for {label}: {path}')
        summary = summarize_diagnostic(path)
        diagnostics[label] = summary
        diagnostic_decisions[label] = diagnostic_checks(
            summary,
            translated_lift_min=args.translated_lift_min,
            chemistry_flat_abs_max=args.chemistry_flat_abs_max,
        )

    label_checks = audit_checks(audit_data, translated_contact_max_fraction=args.translated_contact_max_fraction)
    failure_pattern_seen = any(decision['current_failure_pattern'] for decision in diagnostic_decisions.values())
    any_paper_ready = any(decision['paper_ready_ligand_causal'] for decision in diagnostic_decisions.values())
    if not label_checks['all_pass']:
        recommendation = 'fix_contact_audit_or_dataset_labels_before_new_objectives'
    elif any_paper_ready:
        recommendation = 'review_positive_chemistry_signal_before_stage2_use'
    elif failure_pattern_seen:
        recommendation = 'proceed_to_typed_candidate_rotamer_ligand_interaction_energy'
    else:
        recommendation = 'rerun_or_fix_diagnostics_before_architecture_change'

    return {
        'report_type': 'stage1_ligand_causality_validation',
        'audit_json': str(audit_path),
        'audit_summary': audit_data.get('summary', {}),
        'diagnostics': diagnostics,
        'go_no_go': {
            'label_layer_checks': label_checks,
            'diagnostic_checks': diagnostic_decisions,
            'paper_ready_ligand_causal_signal_found': any_paper_ready,
            'current_candidate_rerank_failure_pattern_validated': failure_pattern_seen,
            'recommendation': recommendation,
            'next_direction_default': 'typed_candidate_rotamer_ligand_interaction_compatibility_energy',
            'stage2_posture': 'zero-prior fallback plus ablation-only soft features until chemistry controls pass',
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize Stage-1 ligand-causality validation outputs')
    parser.add_argument('--audit_json', required=True)
    parser.add_argument('--diagnostic', action='append', default=[],
                        help='diagnostic JSON as label=path; can be repeated')
    parser.add_argument('--baseline_json', default=None)
    parser.add_argument('--translated_rerank_json', default=None)
    parser.add_argument('--shuffled_rerank_json', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--translated_lift_min', type=float, default=0.03)
    parser.add_argument('--chemistry_flat_abs_max', type=float, default=0.005)
    parser.add_argument('--translated_contact_max_fraction', type=float, default=0.10)
    args = parser.parse_args()

    report = build_report(args)
    if args.output_json:
        output_json = Path(args.output_json)
    elif args.output_dir:
        output_json = Path(args.output_dir) / 'causality_validation_report.json'
    else:
        output_json = Path('causality_validation_report.json')
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print('=== Stage-1 ligand-causality validation summary ===')
    print(f'Output JSON: {output_json}')
    print(f"Recommendation: {report['go_no_go']['recommendation']}")
    print(f"Failure pattern validated: {report['go_no_go']['current_candidate_rerank_failure_pattern_validated']}")
    print(f"Paper-ready ligand-causal signal found: {report['go_no_go']['paper_ready_ligand_causal_signal_found']}")


if __name__ == '__main__':
    main()
