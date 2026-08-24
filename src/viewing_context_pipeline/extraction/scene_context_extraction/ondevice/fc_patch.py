import torch
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed

def convert_to_fc_patch(model):
    class Qwen3VLVisionPatchEmbedFC(torch.nn.Module):
        def __init__(self, original) -> None:
            super().__init__()
            assert isinstance(original, Qwen3VLVisionPatchEmbed)

            patch_size = original.patch_size
            temporal_patch_size = original.temporal_patch_size
            in_channels = original.in_channels
            embed_dim = original.embed_dim
            
            in_features_fc = in_channels * temporal_patch_size * patch_size * patch_size
            out_features_fc = embed_dim

            assert original.proj.stride == (temporal_patch_size, patch_size, patch_size)

            self.proj = torch.nn.Linear(in_features = in_features_fc, out_features = out_features_fc, bias=True)
            self.proj.weight.data = original.proj.weight.data.view(out_features_fc, in_features_fc)
            self.proj.bias.data = original.proj.bias.data

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            target_dtype = self.proj.weight.dtype
            hidden_states = self.proj(hidden_states.to(dtype=target_dtype))
            return hidden_states

    assert isinstance(model, Qwen3VLForConditionalGeneration)
    original = model.model.visual.patch_embed 
    assert isinstance(original, Qwen3VLVisionPatchEmbed)
    model.model.visual.patch_embed = Qwen3VLVisionPatchEmbedFC(original)
