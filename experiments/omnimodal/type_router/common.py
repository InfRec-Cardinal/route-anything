"""Shared contract for task classification followed by a global model lookup."""
from pathlib import Path
import sys

from model_policy import require_current_table

SCHEMA_VERSION = 'task-capability-router-v1'


def task_payload(row, dimensions):
    """The complete classifier input; no candidate models or observed outcomes."""
    return {
        'request': row['input_text'],
        'output_contract': row.get('output_contract', {}),
        'benchmark': row['benchmark'],
        'metric_id': row.get('metric_id'),
        'candidate_capabilities': [
            {key: dimension[key] for key in ('id', 'name', 'definition') if key in dimension}
            for dimension in dimensions
        ],
    }


def lookup_model(table, benchmark, metric_id, capability_id):
    """Use one capability-to-model table across all benchmarks and metrics.

    The task's benchmark and metric are classifier context only. They never
    select a different table entry for the same predicted capability.
    """
    require_current_table(table)
    if table.get('scope') != 'global':
        raise ValueError('Capability routing requires one global capability-to-model table.')
    return next((entry for entry in table['entries']
                 if entry['capability_id'] == capability_id), None)
