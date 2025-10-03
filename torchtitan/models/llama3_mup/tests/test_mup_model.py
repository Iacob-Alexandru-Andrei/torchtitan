# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
from torchtitan.models.llama3_mup.model.mup_args import TransformerModelArgs
from torchtitan.models.llama3_mup.model.mup_model import Transformer, TransformerBlock


class TestMuPLlamaModel(unittest.TestCase):
    def setUp(self):
        self.mup_config = {
            "mup_enabled": True,
            "mup_width_multiplier": 2.0,
            "mup_input_alpha": 2.0,
            "mup_output_alpha": 2.0,
            "completep_depth_alpha_enabled": True,
            "completep_depth_multiplier": 2.0,
            "completep_depth_alpha_exp": 0.5,
        }
        self.init_config = {"init_std": 0.02, "emb_init_std": 0.01}
        self.model_args = TransformerModelArgs(
            dim=128,
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            max_seq_len=256,
            use_embedding_norm=True,
            use_peri_norm=True,
            mup_config=self.mup_config,
            init_config=self.init_config,
        )
        self.model = Transformer(self.model_args)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, Transformer)
        # Check if peri_norm layers are created
        for layer in self.model.layers.values():
            self.assertTrue(hasattr(layer, "post_attn_norm"))
            self.assertTrue(hasattr(layer, "post_ffn_norm"))
        # Check if embedding norm is created
        self.assertIsNotNone(self.model.tok_embeddings.norm)

    def test_forward_pass(self):
        input_ids = torch.randint(0, self.model_args.vocab_size, (2, 128))
        output = self.model.forward(input_ids)
        self.assertEqual(
            output.shape, (2, 128, self.model_args.vocab_size)
        )

    def test_weight_initialization(self):
        # A simple check to ensure no errors during init
        self.model.init_weights()


if __name__ == "__main__":
    unittest.main()