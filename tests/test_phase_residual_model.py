import unittest

import torch

from flash_ipa.rigid import Rigid, Rotation

from src.stage2.models.torsion_flow import TorsionFlowNet, TorsionFlowNetConfig


class PhaseResidualModelTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
