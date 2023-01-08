import torch.nn as nn
from mup import MuReadout
from mup.init import normal_
from transformers.modeling_utils import PreTrainedModel
from transformers.models.electra import load_tf_weights_in_electra

from bio_lm.model.electra.attention import ElectraSelfAttention
from bio_lm.model.deberta.attention import DisentangledSelfAttention
from bio_lm.model.electra.config import ElectraConfig


class ElectraPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ElectraConfig
    load_tf_weights = load_tf_weights_in_electra
    base_model_prefix = "electra"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [
        r"electra\.embeddings_project\.weight",
        r"electra\.embeddings_project\.bias",
    ]

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module, readout_zero_init=False, query_zero_init=False):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            ### muP: swap constant std normal init with normal_ from `mup.init`.
            ### Because `_init_weights` is called in `__init__`, before `infshape` is set,
            ### we need to manually call `self.apply(self._init_weights)` after calling
            ### `set_base_shape(model, base)`
            if isinstance(module, MuReadout) and readout_zero_init:
                module.weight.data.zero_()
            else:
                if hasattr(module.weight, "infshape"):
                    normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                else:
                    module.weight.data.normal_(
                        mean=0.0, std=self.config.initializer_range
                    )
            ### End muP
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        ### muP
        if isinstance(module, ElectraSelfAttention):
            if query_zero_init:
                module.query.weight.data[:] = 0

        if isinstance(module, DisentangledSelfAttention):
            if query_zero_init:
                module.query_proj.weight.data[:] = 0
