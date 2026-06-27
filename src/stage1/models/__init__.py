"""Stage-1 模型集合。"""

from .ligand_condition import (
    LigandConditioner,
    LigandConditionerConfig,
    create_ligand_conditioner,
)

from .adapter import (
    ESMAdapter,
    ESMLayerFusionAdapter,
    create_esm_adapter,
    create_esm_layer_fusion_adapter,
)

from .torsion_head import (
    TorsionHead,
    create_torsion_head,
)

from .delta_z_predictor import DeltaZPredictor

__all__ = [
    'LigandConditioner',
    'LigandConditionerConfig',
    'create_ligand_conditioner',
    'ESMAdapter',
    'ESMLayerFusionAdapter',
    'create_esm_adapter',
    'create_esm_layer_fusion_adapter',
    'TorsionHead',
    'create_torsion_head',
    'DeltaZPredictor',
]

try:
    from .ipa import (
        FlashIPAModule,
        FlashIPAModuleConfig,
        create_flashipa_module,
    )

    from .stage1_model import (
        Stage1Model,
        Stage1ModelConfig,
        create_stage1_model,
    )

    __all__.extend([
        'FlashIPAModule',
        'FlashIPAModuleConfig',
        'create_flashipa_module',
        'Stage1Model',
        'Stage1ModelConfig',
        'create_stage1_model',
    ])
except ModuleNotFoundError as exc:
    if exc.name != 'flash_ipa':
        raise
