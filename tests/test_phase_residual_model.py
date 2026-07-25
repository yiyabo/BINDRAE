import sys
import unittest
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[1]
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists() and str(flash_ipa_path) not in sys.path:
    sys.path.insert(0, str(flash_ipa_path))

from flash_ipa.rigid import Rigid, Rotation

from src.stage2.models.torsion_flow import TorsionFlowNet, TorsionFlowNetConfig


class PhaseResidualModelTest(unittest.TestCase):
    def test_low_rank_decoder_replaces_independent_residual_heads(self):
        config = TorsionFlowNetConfig(
            c_s=64,
            c_p=32,
            c_hidden=32,
            no_heads=4,
            depth=1,
            no_qk_points=2,
            no_v_points=4,
            d_lig=16,
            num_heads_cross=4,
            time_dim=16,
            head_hidden=32,
            dropout=0.0,
            phase_residual_enabled=True,
            phase_residual_blockwise=True,
            phase_residual_decoder_mode="low_rank",
            phase_residual_rank=4,
        )
        model = TorsionFlowNet(config)
        self.assertIsNotNone(model.low_rank_residual_decoder)
        self.assertEqual(model.low_rank_residual_decoder.rank, 4)
        self.assertIsNone(model.residual_rotation_head)
        self.assertIsNone(model.residual_translation_head)
        self.assertIsNone(model.residual_chi_head)
        for head in model.low_rank_residual_decoder.coefficient_heads.values():
            self.assertEqual(head[-1].weight.abs().sum().item(), 0.0)
            self.assertEqual(head[-1].bias.abs().sum().item(), 0.0)

        with self.assertRaisesRegex(ValueError, "requires blockwise"):
            TorsionFlowNet(
                TorsionFlowNetConfig(
                    phase_residual_enabled=True,
                    phase_residual_decoder_mode="low_rank",
                )
            )

    def test_blockwise_heads_have_independent_closed_gates(self):
        config = TorsionFlowNetConfig(
            c_s=64,
            c_p=32,
            c_hidden=32,
            no_heads=4,
            depth=1,
            no_qk_points=2,
            no_v_points=4,
            d_lig=16,
            num_heads_cross=4,
            time_dim=16,
            head_hidden=32,
            dropout=0.0,
            phase_residual_enabled=True,
            phase_residual_blockwise=True,
        )
        model = TorsionFlowNet(config)
        self.assertIsNone(model.residual_gate_mlp)
        self.assertIsNone(model.residual_rigid_head)
        self.assertEqual(model.residual_rotation_head[-1].out_features, 3)
        self.assertEqual(model.residual_translation_head[-1].out_features, 3)
        self.assertEqual(model.residual_chi_head[-1].out_features, 4)
        for head in (
            model.residual_rotation_head,
            model.residual_translation_head,
            model.residual_chi_head,
        ):
            self.assertEqual(head[-1].weight.abs().sum().item(), 0.0)
            self.assertEqual(head[-1].bias.abs().sum().item(), 0.0)
        rotation_gate = torch.sigmoid(
            model.residual_rotation_gate_mlp[-1].bias
        ).item()
        translation_gate = torch.sigmoid(
            model.residual_translation_gate_mlp[-1].bias
        ).item()
        chi_gate = torch.sigmoid(model.residual_chi_gate_mlp[-1].bias).item()
        self.assertAlmostEqual(rotation_gate, chi_gate, places=7)
        self.assertLess(translation_gate, rotation_gate / 10.0)

    def test_blockwise_active_blocks_are_explicit(self):
        config = TorsionFlowNetConfig(
            c_s=64,
            c_p=32,
            c_hidden=32,
            no_heads=4,
            depth=1,
            no_qk_points=2,
            no_v_points=4,
            d_lig=16,
            num_heads_cross=4,
            time_dim=16,
            head_hidden=32,
            dropout=0.0,
            phase_residual_enabled=True,
            phase_residual_blockwise=True,
            phase_residual_active_blocks="rotation_chi",
        )
        model = TorsionFlowNet(config)
        self.assertEqual(model.phase_residual_active_blocks, {"rotation", "chi"})

        with self.assertRaisesRegex(ValueError, "only available for blockwise"):
            TorsionFlowNet(
                TorsionFlowNetConfig(
                    phase_residual_enabled=True,
                    phase_residual_active_blocks="rotation",
                )
            )

    def test_blockwise_gate_biases_are_configurable(self):
        config = TorsionFlowNetConfig(
            c_s=64,
            c_p=32,
            c_hidden=32,
            no_heads=4,
            depth=1,
            no_qk_points=2,
            no_v_points=4,
            d_lig=16,
            num_heads_cross=4,
            time_dim=16,
            head_hidden=32,
            dropout=0.0,
            phase_residual_enabled=True,
            phase_residual_blockwise=True,
            phase_residual_rotation_gate_bias=0.0,
            phase_residual_translation_gate_bias=-4.0,
            phase_residual_chi_gate_bias=1.0,
        )
        model = TorsionFlowNet(config)
        self.assertEqual(model.residual_rotation_gate_mlp[-1].bias.item(), 0.0)
        self.assertEqual(model.residual_translation_gate_mlp[-1].bias.item(), -4.0)
        self.assertEqual(model.residual_chi_gate_mlp[-1].bias.item(), 1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "FlashIPA forward requires CUDA")
    def test_dedicated_heads_are_zero_initialized_and_trainable(self):
        torch.manual_seed(3)
        device = torch.device("cuda")
        config = TorsionFlowNetConfig(
            c_s=64,
            c_p=32,
            c_hidden=32,
            no_heads=4,
            depth=1,
            no_qk_points=2,
            no_v_points=4,
            d_lig=16,
            num_heads_cross=4,
            time_dim=16,
            head_hidden=32,
            dropout=0.0,
            phase_residual_enabled=True,
        )
        model = TorsionFlowNet(config).to(device)
        batch_size, n_res, n_lig = 1, 12, 3
        rotations = torch.eye(3, device=device).view(1, 1, 3, 3).expand(
            batch_size,
            n_res,
            3,
            3,
        ).clone()
        rigids = Rigid(
            rots=Rotation(rot_mats=rotations),
            trans=torch.randn(batch_size, n_res, 3, device=device) * 0.1,
        )
        node_mask = torch.tensor(
            [[1] * 11 + [0]],
            dtype=torch.bool,
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                chi=torch.randn(batch_size, n_res, 4, device=device),
                rigids=rigids,
                esm=torch.randn(batch_size, n_res, 1280, device=device),
                lig_points=torch.randn(batch_size, n_lig, 3, device=device),
                lig_types=torch.randn(batch_size, n_lig, 20, device=device),
                lig_mask=torch.ones(
                    batch_size,
                    n_lig,
                    dtype=torch.bool,
                    device=device,
                ),
                w_res=torch.rand(batch_size, n_res, device=device),
                t=torch.tensor([0.4], device=device),
                node_mask=node_mask,
            )

        self.assertEqual(output["residual_chi"].shape, (batch_size, n_res, 4))
        self.assertEqual(output["residual_rigid_rot"].shape, (batch_size, n_res, 3))
        self.assertEqual(output["residual_rigid_trans"].shape, (batch_size, n_res, 3))
        self.assertEqual(output["residual_chi"].abs().max().item(), 0.0)
        self.assertEqual(output["residual_rigid_rot"].abs().max().item(), 0.0)
        self.assertEqual(output["residual_rigid_trans"].abs().max().item(), 0.0)
        self.assertEqual(output["residual_gate"][:, -1].abs().sum().item(), 0.0)

        loss = (
            output["residual_chi"].sum()
            + output["residual_rigid_rot"].sum()
            + output["residual_rigid_trans"].sum()
        )
        loss.backward()
        for head in (model.residual_chi_head, model.residual_rigid_head):
            gradient = head[-1].weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(gradient.abs().sum().item(), 0.0)
        missing_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing_gradients, [])

    @unittest.skipUnless(torch.cuda.is_available(), "FlashIPA forward requires CUDA")
    def test_low_rank_decoder_cuda_forward_and_coefficient_gradients(self):
        torch.manual_seed(7)
        device = torch.device("cuda")
        config = TorsionFlowNetConfig(
            c_s=64,
            c_p=32,
            c_hidden=32,
            no_heads=4,
            depth=1,
            no_qk_points=2,
            no_v_points=4,
            d_lig=16,
            num_heads_cross=4,
            time_dim=16,
            head_hidden=32,
            dropout=0.0,
            phase_residual_enabled=True,
            phase_residual_blockwise=True,
            phase_residual_decoder_mode="low_rank",
            phase_residual_rank=4,
        )
        model = TorsionFlowNet(config).to(device)
        batch_size, n_res, n_lig = 1, 12, 3
        rotations = torch.eye(3, device=device).view(1, 1, 3, 3).expand(
            batch_size,
            n_res,
            3,
            3,
        ).clone()
        rigids = Rigid(
            rots=Rotation(rot_mats=rotations),
            trans=torch.randn(batch_size, n_res, 3, device=device) * 0.1,
        )
        node_mask = torch.tensor(
            [[1] * 11 + [0]],
            dtype=torch.bool,
            device=device,
        )
        chi_mask = node_mask.unsqueeze(-1).expand(-1, -1, 4).clone()
        endpoint_delta_f = torch.randn(
            batch_size,
            n_res,
            6,
            device=device,
        )
        endpoint_delta_chi = torch.randn(
            batch_size,
            n_res,
            4,
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                chi=torch.randn(batch_size, n_res, 4, device=device),
                rigids=rigids,
                esm=torch.randn(batch_size, n_res, 1280, device=device),
                lig_points=torch.randn(batch_size, n_lig, 3, device=device),
                lig_types=torch.randn(batch_size, n_lig, 20, device=device),
                lig_mask=torch.ones(
                    batch_size,
                    n_lig,
                    dtype=torch.bool,
                    device=device,
                ),
                w_res=torch.rand(batch_size, n_res, device=device),
                t=torch.tensor([0.4], device=device),
                node_mask=node_mask,
                chi_mask=chi_mask,
                residual_basis_rigids=rigids,
                residual_endpoint_delta_f=endpoint_delta_f,
                residual_endpoint_delta_chi=endpoint_delta_chi,
            )

        self.assertEqual(output["residual_chi"].abs().max().item(), 0.0)
        self.assertEqual(output["residual_rigid_rot"].abs().max().item(), 0.0)
        self.assertEqual(output["residual_rigid_trans"].abs().max().item(), 0.0)
        for key in (
            "residual_rotation_coefficients",
            "residual_translation_coefficients",
            "residual_chi_coefficients",
        ):
            self.assertEqual(output[key].shape, (batch_size, 4))

        loss = (
            output["residual_chi"].sum()
            + output["residual_rigid_rot"].sum()
            + output["residual_rigid_trans"].sum()
        )
        loss.backward()
        for head in model.low_rank_residual_decoder.coefficient_heads.values():
            gradient = head[-1].weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(gradient.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
