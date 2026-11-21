import hashlib
import unittest
from pathlib import Path

import numpy as np

from fused_op import (
    FP16,
    FP32,
    ComputeNode,
    View,
    mask_uniform,
    replay_capture,
)
from read_capture import read_capture


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "samples" / "capture.json"

ATOL = RTOL = 0.002  # 协议: 乱序累加比较容差


def make_node(*, data, token_ids, valid, weights, seed=28, drop=0.0, eps=1e-5,
              scratch_budget=None):
    node = ComputeNode(
        data=data,
        token_ids=token_ids,
        valid=valid,
        seed=seed,
        drop=drop,
        eps=eps,
        scratch_budget=scratch_budget,
    )
    for version, values in weights.items():
        node.add_weight(version, values)
    return node


class MaskTests(unittest.TestCase):
    def test_blake2b_high_53_bits(self):
        digest = hashlib.blake2b(b"28:1:7:0", digest_size=8).digest()
        self.assertEqual(digest.hex(), "483fba2d913a2ec2")
        expected = (int.from_bytes(digest, "big") >> 11) / float(1 << 53)
        self.assertEqual(mask_uniform(28, 1, 7, 0), expected)

    def test_keyed_by_logical_position(self):
        # 同一逻辑位置结果一致; step/token/hidden 任一不同则数值不同
        self.assertEqual(mask_uniform(28, 2, 12, 3), mask_uniform(28, 2, 12, 3))
        self.assertNotEqual(mask_uniform(28, 1, 7, 0), mask_uniform(28, 2, 7, 0))
        self.assertNotEqual(mask_uniform(28, 1, 7, 0), mask_uniform(28, 1, 12, 0))
        self.assertNotEqual(mask_uniform(28, 1, 7, 0), mask_uniform(28, 1, 7, 1))
        for token in (7, 12, 100):
            for hidden in range(4):
                uniform = mask_uniform(28, 1, token, hidden)
                self.assertGreaterEqual(uniform, 0.0)
                self.assertLess(uniform, 1.0)


class ForwardRoundingTests(unittest.TestCase):
    def run_single(self, x, r, w=(1.0,)):
        node = make_node(
            data={"x": list(x), "r": list(r)},
            token_ids=[5],
            valid=[True],
            weights={"v": list(w)},
        )
        node.register_step(1, "v", [5])
        node.bind(1, View("x", 0, (1, len(x)), (1, 1)),
                  View("r", 0, (1, len(r)), (1, 1)))
        return (node, *node.forward(1, [0]))

    def test_residual_rounds_to_nearest_even(self):
        # 1 + 2**-11 恰好是一半, 最近偶数舍入到 1.0
        _, _, residual = self.run_single([1.0], [2**-11])
        self.assertEqual(residual[0, 0], np.float16(1.0))
        # 1 + 3*2**-11 处于 1+2**-10 与 1+2**-9 之间, 舍入到偶数尾数 1+2**-9
        _, _, residual = self.run_single([1.0], [3 * 2**-11])
        self.assertEqual(residual[0, 0], np.float16(1 + 2**-9))

    def test_norm_and_gradient_use_stored_value(self):
        # 和 1+2**-11 的储存值为 1.0; 归一化与梯度必须以储存值计算
        node, fused, residual = self.run_single([1.0], [2**-11])
        self.assertEqual(residual[0, 0], np.float16(1.0))
        stored = FP32(np.float16(1.0))
        mu = FP32(np.sqrt(FP32(stored * stored) + FP32(1e-5)))
        self.assertEqual(fused[0, 0], np.float16(stored / mu * FP32(1.0)))
        node.backward(1, "g", [0], [[1.0]], [[0.0]])
        published = node.published_gradients(1)
        dyw = FP32(FP32(1.0) * FP32(1.0))
        dot = FP32(dyw * stored)
        factor = FP32(dot / (FP32(1) * mu * mu * mu))
        expected = FP32(dyw / mu - stored * factor)
        self.assertEqual(published["data"]["x"][0], expected)
        self.assertEqual(published["data"]["r"][0], expected)
        self.assertEqual(published["weight"][0], FP32(FP32(1.0) * stored / mu))

    def test_inverted_dropout_scale(self):
        # drop=0.25 时保留位置缩放 4/3 (FP32), 舍弃位置为零
        node = make_node(
            data={"x": [3.0], "r": [0.25]},
            token_ids=[7],
            valid=[True],
            weights={"v": [1.0]},
            drop=0.25,
        )
        node.register_step(1, "v", [7])
        node.bind(1, View("x", 0, (1, 1), (1, 1)), View("r", 0, (1, 1), (1, 1)))
        _, residual = node.forward(1, [0])
        if mask_uniform(28, 1, 7, 0) >= 0.25:
            expected = FP32(FP32(3.0) * (FP32(1.0) / FP32(0.75))) + FP32(0.25)
        else:
            expected = FP32(0.25)
        self.assertEqual(residual[0, 0], np.float16(expected))


class PaddingRowTests(unittest.TestCase):
    def test_bypass_zero_output_and_zero_gradient(self):
        node = make_node(
            data={"x": [1, 2, 3, 4, 5, 6, 7, 8],
                  "r": [10, 20, 30, 40, 50, 60, 70, 80]},
            token_ids=[7, 100],
            valid=[True, False],
            weights={"v": [1, 1, 1, 1]},
        )
        node.register_step(1, "v", [7])
        node.bind(1, View("x", 0, (2, 4), (4, 1)), View("r", 0, (2, 4), (4, 1)))
        fused, residual = node.forward(1, [0, 1])
        # 填充行残差保持原值, 归一化输出为零
        np.testing.assert_array_equal(residual[1],
                                      np.array([50, 60, 70, 80], dtype=FP16))
        np.testing.assert_array_equal(fused[1], np.zeros(4, dtype=FP16))
        # 填充行的非有限反向数值不废弃训练步, 其梯度为零
        result = node.backward(
            1, "g", [0, 1],
            [[1, 1, 1, 1], [None, None, None, None]],
            [[0, 0, 0, 0], [None, None, None, None]],
        )
        self.assertEqual(result, "complete")
        published = node.published_gradients(1)
        np.testing.assert_array_equal(published["data"]["r"][4:],
                                      np.zeros(4, dtype=FP32))
        np.testing.assert_array_equal(published["data"]["x"][4:],
                                      np.zeros(4, dtype=FP32))


class OverlapMergeTests(unittest.TestCase):
    def test_gradients_merge_to_owner_and_offset(self):
        data_ab = [0.5, 1, 2, -2, 4, 6, 3, 2, 1, 3, 4, 5, 5, 7, -2, 8]
        weights = {"v": [1, 1.5, 2, 0.5]}
        rows = [0, 1]
        main_grads = [[1, 0, 2, 1], [0, 1, 0, 1]]
        residual_grads = [[0, 0, 1, 0], [1, 0, 0, 1]]

        def run(shared_storage):
            if shared_storage:
                data = {"ab": data_ab}
                view_main = View("ab", 0, (3, 4), (4, 1))
                view_residual = View("ab", 2, (3, 4), (1, 3))
            else:
                data = {"ab": data_ab, "res": data_ab}
                view_main = View("ab", 0, (3, 4), (4, 1))
                view_residual = View("res", 2, (3, 4), (1, 3))
            node = make_node(
                data=data,
                token_ids=[7, 12, 100],
                valid=[True, True, False],
                weights=weights,
                drop=0.25,
            )
            node.register_step(1, "v", [7, 12])
            node.bind(1, view_main, view_residual)
            node.forward(1, rows)
            node.backward(1, "g", rows, main_grads, residual_grads)
            return node.published_gradients(1)

        merged = run(True)
        split = run(False)
        # 共享存储时所有者梯度 = 主支路与残差梯度按物理偏移相加
        np.testing.assert_allclose(
            merged["data"]["ab"],
            split["data"]["ab"] + split["data"]["res"],
            atol=ATOL, rtol=RTOL,
        )
        np.testing.assert_allclose(merged["weight"], split["weight"],
                                   atol=ATOL, rtol=RTOL)


class LayoutAndSplitTests(unittest.TestCase):
    """截取微批与布局变化应产生相符的前向输出与最终所有者梯度。"""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.tokens, self.hidden = 5, 4
        self.x = rng.normal(size=(self.tokens, self.hidden)).astype(FP16)
        self.r = rng.normal(size=(self.tokens, self.hidden)).astype(FP16)
        self.token_ids = [3, 5, 8, 13, 21]
        self.valid = [True, True, False, True, True]
        self.expected = [3, 5, 13, 21]
        self.weights = {"v": [1.0, 0.5, 1.5, 2.0]}
        self.drop = 0.3
        dy = rng.normal(size=(self.tokens, self.hidden)).astype(FP16)
        dres = rng.normal(size=(self.tokens, self.hidden)).astype(FP16)
        self.dy = dy.astype(np.float32).tolist()
        self.dres = dres.astype(np.float32).tolist()
        self.dy[2] = [None] * self.hidden  # 填充行的非有限应被忽略
        self.dres[2] = [None] * self.hidden

    def build(self, layout):
        if layout == "contiguous":
            data = {"x": self.x.reshape(-1).tolist(),
                    "r": self.r.reshape(-1).tolist()}
            strides = (self.hidden, 1)
        else:
            data = {"x": self.x.T.reshape(-1).tolist(),
                    "r": self.r.T.reshape(-1).tolist()}
            strides = (1, self.tokens)
        node = make_node(
            data=data,
            token_ids=self.token_ids,
            valid=self.valid,
            weights=self.weights,
            drop=self.drop,
        )
        node.register_step(1, "v", self.expected)
        node.bind(1, View("x", 0, (self.tokens, self.hidden), strides),
                  View("r", 0, (self.tokens, self.hidden), strides))
        return node

    def gather(self, grad, layout):
        if layout == "contiguous":
            return grad.reshape(self.tokens, self.hidden)
        return grad.reshape(self.hidden, self.tokens).T

    def test_layout_and_microbatch_split_invariance(self):
        whole = self.build("contiguous")
        fused_all, residual_all = whole.forward(1, [0, 1, 2, 3, 4])
        whole.backward(1, "g", [0, 1, 2, 3, 4], self.dy, self.dres)
        self.assertEqual(whole.step_status(1), "complete")

        split = self.build("transposed")
        fused_parts, residual_parts = {}, {}
        for rows in ([4, 2, 0], [3, 1]):  # 乱序且含填充行的微批拆分
            fused, residual = split.forward(1, rows)
            for position, row in enumerate(rows):
                fused_parts[row] = fused[position]
                residual_parts[row] = residual[position]
        for row in range(self.tokens):
            np.testing.assert_array_equal(fused_parts[row], fused_all[row])
            np.testing.assert_array_equal(residual_parts[row], residual_all[row])
        split.backward(1, "g1", [3, 0],
                       [self.dy[3], self.dy[0]], [self.dres[3], self.dres[0]])
        split.backward(1, "g2", [4, 1, 2],
                       [self.dy[4], self.dy[1], self.dy[2]],
                       [self.dres[4], self.dres[1], self.dres[2]])
        self.assertEqual(split.step_status(1), "complete")

        whole_pub = whole.published_gradients(1)
        split_pub = split.published_gradients(1)
        for name in ("x", "r"):
            np.testing.assert_allclose(
                self.gather(split_pub["data"][name], "transposed"),
                self.gather(whole_pub["data"][name], "contiguous"),
                atol=ATOL, rtol=RTOL,
            )
        np.testing.assert_allclose(split_pub["weight"], whole_pub["weight"],
                                   atol=ATOL, rtol=RTOL)

    def test_padding_compaction_keeps_mask_and_outputs(self):
        # 压紧填充行: 只交付有效行的微批与完整微批逐行一致
        full = self.build("contiguous")
        fused_all, residual_all = full.forward(1, [0, 1, 2, 3, 4])
        compact_rows = [0, 1, 3, 4]
        compact = self.build("contiguous")
        fused, residual = compact.forward(1, compact_rows)
        for position, row in enumerate(compact_rows):
            np.testing.assert_array_equal(fused[position], fused_all[row])
            np.testing.assert_array_equal(residual[position], residual_all[row])


class WeightVersionTests(unittest.TestCase):
    def test_inflight_step_keeps_pinned_weight(self):
        node = make_node(
            data={"x": [1, 2], "r": [0.5, -0.5]},
            token_ids=[7],
            valid=[True],
            weights={"r1": [2.0, 0.5]},
        )
        node.register_step(1, "r1", [7])
        node.bind(1, View("x", 0, (1, 2), (2, 1)), View("r", 0, (1, 2), (2, 1)))
        node.forward(1, [0])
        # 训练器把新版权重发给接下来的一批
        node.add_weight("r2", [3.0, 1.0])
        node.register_step(2, "r2", [7])
        node.bind(2, View("x", 0, (1, 2), (2, 1)), View("r", 0, (1, 2), (2, 1)))
        node.forward(2, [0])
        # 在途梯度仍以旧版本重算并交付给旧版
        node.backward(1, "g1", [0], [[1.0, 1.0]], [[0.0, 0.0]])
        node.backward(2, "g2", [0], [[1.0, 1.0]], [[0.0, 0.0]])

        xhat = np.array([1.5, 1.5], dtype=FP32)  # drop=0: 1+0.5, 2-0.5
        mu = FP32(np.sqrt(FP32(np.dot(xhat, xhat)) / FP32(2) + FP32(1e-5)))

        def expected(weight):
            w = np.asarray(weight, dtype=FP32)
            dy = np.ones(2, dtype=FP32)
            dyw = dy * w
            dot = FP32(np.dot(dyw, xhat))
            factor = FP32(dot / (FP32(2) * mu * mu * mu))
            dxhat = dyw / mu - xhat * factor
            return dxhat, dy * xhat / mu

        dx1, dw1 = expected([2.0, 0.5])
        dx2, dw2 = expected([3.0, 1.0])
        pub1 = node.published_gradients(1)
        pub2 = node.published_gradients(2)
        np.testing.assert_array_equal(pub1["data"]["x"], dx1)
        np.testing.assert_array_equal(pub1["data"]["r"], dx1)
        np.testing.assert_array_equal(pub1["weight"], dw1)
        np.testing.assert_array_equal(pub2["data"]["x"], dx2)
        np.testing.assert_array_equal(pub2["weight"], dw2)
        self.assertEqual(pub1["weight_version"], "r1")
        self.assertEqual(pub2["weight_version"], "r2")


class StepLifecycleTests(unittest.TestCase):
    def test_capture_replay_lifecycle(self):
        node, records = replay_capture(read_capture(CAPTURE))
        # 步 1 有效行含非有限反向数值 -> 废弃, 不发布任何更新
        self.assertEqual(node.step_status(1), "aborted")
        self.assertIsNone(node.published_gradients(1))
        # 其它合法训练步仍能完成并发布
        self.assertEqual(node.step_status(2), "complete")
        published = node.published_gradients(2)
        self.assertEqual(set(published["data"]), {"ab", "z"})
        self.assertEqual(published["weight_version"], "r2")
        self.assertEqual(published["data"]["ab"].shape, (16,))
        self.assertEqual(published["data"]["z"].shape, (12,))
        self.assertEqual(published["weight"].shape, (4,))
        self.assertTrue(np.all(np.isfinite(published["data"]["ab"])))
        results = [record[-1] for record in records if record[0] == "backward"]
        self.assertEqual(results, ["awaiting_gradients", "aborted", "complete"])

    def test_partial_delivery_is_not_published(self):
        capture = read_capture(CAPTURE)
        views = {name: View.from_spec(spec)
                 for name, spec in capture["views"].items()}
        node = make_node(
            data=capture["data"],
            token_ids=capture["token_ids"],
            valid=capture["valid"],
            weights=capture["weight"],
            seed=capture["seed"],
            drop=capture["drop"],
            eps=capture["eps"],
            scratch_budget=capture["scratch_floats"],
        )
        node.register_step(1, "r1", [7, 12])
        node.bind(1, views["a"], views["b"])
        node.forward(1, [0, 2])
        node.forward(1, [1])
        # 前向答复后: 可重算, 未发布
        self.assertEqual(node.step_status(1), "recomputable")
        self.assertIsNone(node.published_gradients(1))
        node.backward(1, "g1", [1], [[1, 0, 2, 1]], [[0, 0, 1, 0]])
        # 部分交付: 须等待梯度交齐, 不发布部分更新
        self.assertEqual(node.step_status(1), "awaiting_gradients")
        self.assertIsNone(node.published_gradients(1))
        info = node.step_info(1)
        self.assertEqual(info["pending_tokens"], [7])
        self.assertEqual(info["delivered_tokens"], [12])
        node.backward(1, "g0", [0], [[1, None, 1, 0]], [[0, 1, 0, 0]])
        self.assertEqual(node.step_status(1), "aborted")
        self.assertIsNone(node.published_gradients(1))

    def test_duplicate_delivery_makes_no_extra_update(self):
        capture = read_capture(CAPTURE)
        first, _ = replay_capture(capture)
        repeated = read_capture(CAPTURE)
        repeated["events"] = repeated["events"] + [repeated["events"][-1]]
        second, records = replay_capture(repeated)
        self.assertEqual(records[-1][-1], "complete")  # 已关闭, 忽略
        for name, grad in first.published_gradients(2)["data"].items():
            np.testing.assert_array_equal(
                grad, second.published_gradients(2)["data"][name])
        np.testing.assert_array_equal(first.published_gradients(2)["weight"],
                                      second.published_gradients(2)["weight"])

    def test_duplicate_grad_id_before_completion(self):
        node = make_node(
            data={"x": [1, 2], "r": [0.5, -0.5]},
            token_ids=[7, 12],
            valid=[True, True],
            weights={"v": [1.0]},
        )
        node.register_step(1, "v", [7, 12])
        node.bind(1, View("x", 0, (2, 1), (1, 1)), View("r", 0, (2, 1), (1, 1)))
        node.forward(1, [0, 1])
        self.assertEqual(node.backward(1, "g", [0], [[1.0]], [[0.0]]),
                         "awaiting_gradients")
        self.assertEqual(node.backward(1, "g", [0], [[1.0]], [[0.0]]),
                         "duplicate")
        after = node.step_info(1)
        self.assertEqual(after["delivered_tokens"], [7])
        node.backward(1, "g2", [1], [[1.0]], [[0.0]])
        published = node.published_gradients(1)
        # 行 0 只计入一次
        xhat0 = FP32(np.float16(FP32(1.0) + FP32(0.5)))
        mu0 = FP32(np.sqrt(FP32(xhat0 * xhat0) + FP32(1e-5)))
        dot0 = FP32(FP32(1.0) * xhat0)
        factor0 = FP32(dot0 / (FP32(1) * mu0 * mu0 * mu0))
        expected0 = FP32(FP32(1.0) / mu0 - xhat0 * factor0)
        self.assertEqual(published["data"]["x"][0], expected0)


class CaptureValueTests(unittest.TestCase):
    """以协议口径独立重算采集中的前向与所有者梯度。"""

    def test_capture_forward_values(self):
        capture = read_capture(CAPTURE)
        _, records = replay_capture(capture)
        fused, residual = records[2][3], records[2][4]  # forward 步1 行[0,2]
        ab = np.array(capture["data"]["ab"], dtype=FP16)
        # 步1 token7 四个位置全部保留
        self.assertTrue(all(mask_uniform(28, 1, 7, h) >= 0.25 for h in range(4)))
        x = ab[0:4].astype(FP32)  # 视图 a 行0
        r = ab[[2, 5, 8, 11]].astype(FP32)  # 视图 b 行0
        scale = FP32(1.0) / FP32(1.0 - FP32(0.25))
        xhat = (x * scale + r).astype(FP16).astype(FP32)
        mu = FP32(np.sqrt(FP32(np.dot(xhat, xhat)) / FP32(4) + FP32(1e-5)))
        w = np.array(capture["weight"]["r1"], dtype=FP32)
        np.testing.assert_array_equal(residual[0], xhat.astype(FP16))
        np.testing.assert_array_equal(fused[0], ((xhat / mu) * w).astype(FP16))
        # 行2 是填充行: 残差原样绕过, 归一化输出为零
        np.testing.assert_array_equal(residual[1], ab[[4, 7, 10, 13]])
        np.testing.assert_array_equal(fused[1], np.zeros(4, dtype=FP16))

    def test_capture_step2_owner_gradients(self):
        capture = read_capture(CAPTURE)
        node, _ = replay_capture(capture)
        published = node.published_gradients(2)
        z = np.array(capture["data"]["z"], dtype=FP16)
        ab = np.array(capture["data"]["ab"], dtype=FP16)
        w = np.array(capture["weight"]["r2"], dtype=FP32)
        eps = FP32(capture["eps"])
        drop = capture["drop"]
        keep = FP32(1.0) / FP32(1.0 - FP32(drop))
        grad_main = {0: [2, 1, 1, 0], 1: [1, 2, 1, 3]}
        grad_res = {0: [0, 0, 1, 0], 1: [1, 0, 0, 1]}
        exp_z = np.zeros(12, dtype=FP32)
        exp_ab = np.zeros(16, dtype=FP32)
        exp_w = np.zeros(4, dtype=FP32)
        for row, token in ((0, 7), (1, 12)):
            scale = np.array(
                [keep if mask_uniform(28, 2, token, h) >= drop else FP32(0.0)
                 for h in range(4)],
                dtype=FP32,
            )
            x = z[row * 4: row * 4 + 4].astype(FP32)  # 视图 c 行
            res_offsets = [2 + row + 3 * h for h in range(4)]  # 视图 b 行
            r = ab[res_offsets].astype(FP32)
            xhat = (x * scale + r).astype(FP16).astype(FP32)
            mu = FP32(np.sqrt(FP32(np.dot(xhat, xhat)) / FP32(4) + eps))
            dy = np.array(grad_main[row], dtype=FP32)
            dres = np.array(grad_res[row], dtype=FP32)
            dyw = dy * w
            dot = FP32(np.dot(dyw, xhat))
            factor = FP32(dot / (FP32(4) * mu * mu * mu))
            dout = dyw / mu - xhat * factor + dres
            exp_z[row * 4: row * 4 + 4] += dout * scale
            np.add.at(exp_ab, res_offsets, dout)
            exp_w += dy * xhat / mu
        np.testing.assert_array_equal(published["data"]["z"], exp_z)
        np.testing.assert_array_equal(published["data"]["ab"], exp_ab)
        np.testing.assert_array_equal(published["weight"], exp_w)

    def test_replay_is_deterministic(self):
        capture = read_capture(CAPTURE)
        first, first_records = replay_capture(capture)
        second, second_records = replay_capture(capture)
        for rec_a, rec_b in zip(first_records, second_records):
            if rec_a[0] == "forward":
                np.testing.assert_array_equal(rec_a[3], rec_b[3])
                np.testing.assert_array_equal(rec_a[4], rec_b[4])
        for step in (1, 2):
            pub_a = first.published_gradients(step)
            pub_b = second.published_gradients(step)
            self.assertEqual(pub_a is None, pub_b is None)
            if pub_a is not None:
                for name, grad in pub_a["data"].items():
                    np.testing.assert_array_equal(grad, pub_b["data"][name])


class ScratchTests(unittest.TestCase):
    def test_peak_within_budget_and_released_between_requests(self):
        capture = read_capture(CAPTURE)
        node, _ = replay_capture(capture)
        self.assertLessEqual(node.scratch.peak, capture["scratch_floats"])
        # 反传请求之间不保留完整未交付中间量
        self.assertEqual(node.scratch.current, 0)

    def test_budget_is_enforced(self):
        node = make_node(
            data={"x": [1, 2, 3, 4], "r": [1, 1, 1, 1]},
            token_ids=[7],
            valid=[True],
            weights={"v": [1, 1, 1, 1]},
            scratch_budget=2,
        )
        node.register_step(1, "v", [7])
        node.bind(1, View("x", 0, (1, 4), (4, 1)), View("r", 0, (1, 4), (4, 1)))
        with self.assertRaisesRegex(RuntimeError, "预算"):
            node.forward(1, [0])
        self.assertEqual(node.scratch.current, 0)  # 失败后同样释放


class ProtocolErrorTests(unittest.TestCase):
    def setUp(self):
        self.node = make_node(
            data={"x": [1, 2], "r": [3, 4]},
            token_ids=[7],
            valid=[True],
            weights={"v": [1, 1]},
        )

    def test_unknown_step(self):
        with self.assertRaisesRegex(ValueError, "未知训练步"):
            self.node.forward(9, [0])

    def test_duplicate_step_registration(self):
        self.node.register_step(1, "v", [7])
        with self.assertRaisesRegex(ValueError, "已登记"):
            self.node.register_step(1, "v", [7])

    def test_unknown_weight_version(self):
        with self.assertRaisesRegex(ValueError, "未知权重版本"):
            self.node.register_step(1, "nope", [7])

    def test_view_out_of_bounds(self):
        self.node.register_step(1, "v", [7])
        with self.assertRaisesRegex(ValueError, "越界"):
            self.node.bind(1, View("x", 0, (1, 2), (0, 4)),
                           View("r", 0, (1, 2), (0, 4)))

    def test_row_out_of_range(self):
        self.node.register_step(1, "v", [7])
        self.node.bind(1, View("x", 0, (1, 2), (2, 1)),
                       View("r", 0, (1, 2), (2, 1)))
        with self.assertRaisesRegex(ValueError, "行号越界"):
            self.node.forward(1, [1])

    def test_forward_on_closed_step(self):
        self.node.register_step(1, "v", [7])
        self.node.bind(1, View("x", 0, (1, 2), (2, 1)),
                       View("r", 0, (1, 2), (2, 1)))
        self.node.forward(1, [0])
        self.node.backward(1, "g", [0], [[1, 1]], [[0, 0]])
        self.assertEqual(self.node.step_status(1), "complete")
        with self.assertRaisesRegex(ValueError, "已关闭"):
            self.node.forward(1, [0])


if __name__ == "__main__":
    unittest.main()
