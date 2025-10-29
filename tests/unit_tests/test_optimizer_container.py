import unittest

import torch
import torch.nn as nn

from torchtitan.components.optimizer import OptimizersContainer


class TestOptimizersContainerStateDict(unittest.TestCase):
    def _build_container(self) -> OptimizersContainer:
        torch.manual_seed(0)
        model_parts = [nn.Linear(4, 4) for _ in range(2)]
        container = OptimizersContainer(
            model_parts,
            torch.optim.AdamW,
            {"lr": 1e-3},
        )

        container.zero_grad()
        loss = 0.0
        for model in model_parts:
            data = torch.randn(8, 4)
            target = torch.randn(8, 4)
            loss = loss + torch.nn.functional.mse_loss(model(data), target)
        loss.backward()
        container.step()
        return container

    def test_state_dict_uses_unique_prefixes(self) -> None:
        container = self._build_container()

        state_dict = container.state_dict()
        prefixes = {key.split(".", 1)[0] for key in state_dict if "." in key}

        self.assertGreaterEqual(len(prefixes), 2)
        self.assertTrue(all(prefix.startswith("model_part_") for prefix in prefixes))

    def test_prefixed_state_dict_round_trip(self) -> None:
        container = self._build_container()
        original_state = container.state_dict()

        prefixed_keys = {key.split(".", 1)[0] for key in original_state if "." in key}
        self.assertEqual(len(prefixed_keys), 2)

        prefix_order = sorted(prefixed_keys)
        prefix_to_index = {prefix: idx for idx, prefix in enumerate(prefix_order)}

        updated_state = {}
        for key, value in original_state.items():
            if isinstance(value, torch.Tensor):
                prefix = key.split(".", 1)[0]
                fill_value = prefix_to_index[prefix] + 1
                updated_state[key] = torch.full_like(value, fill_value)
            else:
                updated_state[key] = value

        container.load_state_dict(updated_state)
        reloaded_state = container.state_dict()

        for key, value in reloaded_state.items():
            if not isinstance(value, torch.Tensor):
                continue
            prefix = key.split(".", 1)[0]
            fill_value = prefix_to_index[prefix] + 1
            expected = torch.full_like(value, fill_value)
            self.assertTrue(torch.allclose(value, expected))


if __name__ == "__main__":
    unittest.main()
