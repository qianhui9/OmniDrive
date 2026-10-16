import importlib.util
from pathlib import Path
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture
def llm():
    # Load the actual native multimodal assembler without importing all of mmdet.
    path = Path('projects/mmdet3d_plugin/models/dense_heads/llava_arch.py')
    spec = importlib.util.spec_from_file_location('m3_test_native_arch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class NativeLLM(LlamaForCausalLM, module.LlavaMetaForCausalLM):
        def get_model(self):
            return self.model

        def generate(self, inputs=None, images=None, **kwargs):
            if images is not None:
                _, p, a, _, embeds, _ = self.prepare_inputs_labels_for_multimodal(
                    inputs, kwargs.pop('position_ids', None), kwargs.pop('attention_mask', None),
                    None, None, images, None)
                return super().generate(inputs_embeds=embeds, position_ids=p, attention_mask=a, **kwargs)
            return super().generate(inputs_embeds=self.get_input_embeddings()(inputs), **kwargs)
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=256, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = NativeLLM(config).eval()
    model.hidden_size = config.hidden_size
    return model


@pytest.fixture
def carriers():
    return torch.randn(1, 9, 32)


@pytest.fixture
def context():
    return dict(scene_id='s', frame_id='f', camera_ids=['front'], timestamp=1.)
