"""融合 dropout + 残差加法 + RMSNorm 的前反向接口(按交接协议)。

数值口径(见 交接协议.md):

* 掩码按逻辑 (seed, step, token, hidden) 归属: 取 ``blake2b(digest_size=8)``
  对 ASCII 文本 ``seed:step:token:hidden`` 的高 53 位形成 ``[0,1)`` 的数,
  小于 ``drop`` 舍弃, 保留者按倒置 dropout 乘 ``1/(1-drop)``。掩码只依赖逻辑
  位置, 不因压紧填充行、改变微批拆分或使用非连续视图而变动。
* 主支路输入先做倒置 dropout 得到 FP32 值, 与残差相加后按最近偶数舍入至
  FP16; 该残差作为副输出交付(填充行原样绕过)。RMS 与缩放读取储存值在
  FP32 下计算, 归一化输出舍入 FP16。
* 反向在释放大中间量后重算, 只依据本次前向对应的输入值、逻辑随机位置与
  步骤登记的权重版本; 梯度以储存值计算, 舍入采用直通约定, 主体和副输出的
  上游梯度都进入。
* 两支路物理存储重叠时, 梯度合并到真正的所有者与偏移; 微批可乱序完成,
  重复提交不造成额外更新; 有效行含非有限反向数值时整步废弃, 不发布任何
  部分更新, 其它已固定版本的合法训练步不受影响。
* ``scratch_floats`` 只计新建临时浮点元素(含接收梯度的精度转换与标量工作区),
  交付张量和原存储不计; 每次请求结束时临时量全部释放, 请求之间不保留完整
  未交付中间量。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FP16",
    "FP32",
    "ComputeNode",
    "ScratchLedger",
    "StepStatus",
    "View",
    "mask_uniform",
    "replay_capture",
]

FP16 = np.float16
FP32 = np.float32

_MASK_SHIFT = 64 - 53
_MASK_DENOM = float(1 << 53)


def mask_uniform(seed: int, step: int, token: int, hidden: int) -> float:
    """逻辑位置 ``(seed, step, token, hidden)`` 对应的 ``[0,1)`` 均匀数。

    取 ``blake2b(digest_size=8)`` 对 ASCII 文本 ``seed:step:token:hidden``
    的高 53 位除以 ``2**53``。只依赖逻辑位置, 与微批行号、填充行压紧和
    视图布局无关。
    """
    text = f"{seed}:{step}:{token}:{hidden}".encode("ascii")
    digest = hashlib.blake2b(text, digest_size=8).digest()
    return (int.from_bytes(digest, "big") >> _MASK_SHIFT) / _MASK_DENOM


@dataclass(frozen=True)
class View:
    """``[storage, offset, shape, strides]`` 视图。

    偏移与步幅均按元素计, ``shape`` 为 ``[token, hidden]``; 相同 storage 可
    被不同视图引用并允许交叠。
    """

    storage: str
    offset: int
    shape: Tuple[int, int]
    strides: Tuple[int, int]

    @classmethod
    def from_spec(cls, spec: Sequence) -> "View":
        try:
            storage, offset, shape, strides = spec
            tokens, hidden = shape
            stride_token, stride_hidden = strides
        except (TypeError, ValueError) as exc:
            raise ValueError(f"无效的视图规格: {spec!r}") from exc
        return cls(
            str(storage),
            int(offset),
            (int(tokens), int(hidden)),
            (int(stride_token), int(stride_hidden)),
        )

    @property
    def tokens(self) -> int:
        return self.shape[0]

    @property
    def hidden(self) -> int:
        return self.shape[1]

    def offset_at(self, row: int, col: int) -> int:
        return self.offset + row * self.strides[0] + col * self.strides[1]

    def row_offsets(self, row: int) -> np.ndarray:
        cols = np.arange(self.hidden, dtype=np.int64)
        return self.offset + row * self.strides[0] + cols * self.strides[1]

    def check_bounds(self, storage_len: int) -> None:
        corners = [
            self.offset_at(r, c)
            for r in (0, self.tokens - 1)
            for c in (0, self.hidden - 1)
        ]
        if min(corners) < 0 or max(corners) >= storage_len:
            raise ValueError(
                "视图越界: "
                f"storage={self.storage} offset={self.offset} "
                f"shape={self.shape} strides={self.strides} 长度={storage_len}"
            )


class ScratchLedger:
    """临时浮点元素账本。

    只计新建的临时浮点元素; 交付张量(融合输出、交付残差、所有者梯度累积)
    和原存储不计。``current`` 在每次请求结束时回到零, 证明请求之间没有保留
    完整未交付中间量。
    """

    def __init__(self, budget: Optional[int] = None):
        self.budget = None if budget is None else int(budget)
        self.current = 0
        self.peak = 0

    def alloc(self, count: int) -> None:
        count = int(count)
        new_total = self.current + count
        if self.budget is not None and new_total > self.budget:
            raise RuntimeError(
                f"临时浮点元素超出预算: {new_total} > {self.budget}"
            )
        self.current = new_total
        if self.current > self.peak:
            self.peak = self.current

    def release(self, count: int) -> None:
        self.current -= int(count)


class _ScratchSession:
    """一次请求内的临时量会话; 退出时全部释放。"""

    def __init__(self, ledger: ScratchLedger):
        self._ledger = ledger
        self._counts: List[int] = []

    def array(self, count: int) -> np.ndarray:
        count = int(count)
        self._ledger.alloc(count)
        self._counts.append(count)
        return np.empty(count, dtype=FP32)

    def reserve(self, count: int) -> None:
        """只记账不分配, 用于 mu/dot/factor 等标量临时量。"""
        count = int(count)
        self._ledger.alloc(count)
        self._counts.append(count)

    def __enter__(self) -> "_ScratchSession":
        return self

    def __exit__(self, *exc) -> bool:
        for count in self._counts:
            self._ledger.release(count)
        self._counts.clear()
        return False


class StepStatus(str, Enum):
    """训练步状态, 供训练端区分可重算、等待交齐与已终止。"""

    REGISTERED = "registered"  #: 已登记并绑定, 尚未前向
    RECOMPUTABLE = "recomputable"  #: 前向已答复且中间量已释放, 反传可重算
    AWAITING_GRADIENTS = "awaiting_gradients"  #: 部分梯度已交付, 须等待交齐
    COMPLETE = "complete"  #: 梯度交齐, 更新已发布
    ABORTED = "aborted"  #: 有效行含非有限反向数值, 已因数值问题终止


@dataclass
class _StepState:
    step_id: int
    weight_version: str
    weight: np.ndarray  # 登记时固定的权重副本, 在途梯度不得以新权重重算
    expected_tokens: frozenset
    view_main: Optional[View] = None
    view_residual: Optional[View] = None
    status: StepStatus = StepStatus.REGISTERED
    deliveries: set = field(default_factory=set)  # 已接收的梯度提交号
    delivered_tokens: set = field(default_factory=set)  # 已交付梯度的逻辑 token
    data_grads: Dict[str, np.ndarray] = field(default_factory=dict)
    weight_grad: Optional[np.ndarray] = None


def _is_finite(value) -> bool:
    if value is None:  # null 在梯度资料表示非有限
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class ComputeNode:
    """按交接协议执行融合算子并交付梯度的小内存计算节点。"""

    def __init__(
        self,
        *,
        data: Dict[str, Sequence[float]],
        token_ids: Sequence[int],
        valid: Sequence[bool],
        seed: int,
        drop: float,
        eps: float,
        scratch_budget: Optional[int] = None,
    ):
        self.storage = {
            name: np.asarray(values, dtype=FP16) for name, values in data.items()
        }
        self.token_ids = [int(token) for token in token_ids]
        self.valid = [bool(flag) for flag in valid]
        self.seed = int(seed)
        self.drop = float(drop)
        if not 0.0 <= self.drop < 1.0:
            raise ValueError("drop 必须在 [0,1) 内")
        # 倒置 dropout 的 FP32 保留缩放
        self.keep_scale = FP32(1.0) / FP32(1.0 - self.drop)
        self.eps = FP32(eps)
        self.scratch = ScratchLedger(scratch_budget)
        self._weights: Dict[str, np.ndarray] = {}
        self._steps: Dict[int, _StepState] = {}

    # ------------------------------------------------------------------ 登记

    def add_weight(self, version: str, values: Sequence[float]) -> None:
        """接收训练器发来的一个权重版本(版本不可变)。"""
        version = str(version)
        if version in self._weights:
            raise ValueError(f"权重版本已存在: {version}")
        self._weights[version] = np.asarray(values, dtype=FP32)

    def register_step(
        self, step_id: int, weight_version: str, expected_tokens: Sequence[int]
    ) -> None:
        """登记训练步: 固定应收 token 与训练版本(权重取副本)。"""
        step_id = int(step_id)
        if step_id in self._steps:
            raise ValueError(f"训练步已登记: {step_id}")
        weight_version = str(weight_version)
        if weight_version not in self._weights:
            raise ValueError(f"未知权重版本: {weight_version}")
        self._steps[step_id] = _StepState(
            step_id=step_id,
            weight_version=weight_version,
            weight=self._weights[weight_version].copy(),
            expected_tokens=frozenset(int(token) for token in expected_tokens),
        )

    def bind(self, step_id: int, view_main: View, view_residual: View) -> None:
        """固定同一步观测的主支路与残差视图。"""
        st = self._require_step(step_id)
        if st.view_main is not None:
            raise ValueError(f"训练步 {step_id} 已绑定视图")
        if view_main.shape != view_residual.shape:
            raise ValueError("主支路与残差视图形状不一致")
        if view_main.hidden != st.weight.shape[0]:
            raise ValueError("隐藏维与权重长度不一致")
        if view_main.tokens > len(self.token_ids):
            raise ValueError("视图行数超出局部 token 数")
        for view in (view_main, view_residual):
            if view.storage not in self.storage:
                raise ValueError(f"未知存储: {view.storage}")
            view.check_bounds(len(self.storage[view.storage]))
        st.view_main = view_main
        st.view_residual = view_residual
        st.weight_grad = np.zeros(st.weight.shape, dtype=FP32)
        for view in (view_main, view_residual):
            st.data_grads.setdefault(
                view.storage, np.zeros(len(self.storage[view.storage]), dtype=FP32)
            )

    # ------------------------------------------------------------------ 前向

    def forward(
        self, step_id: int, rows: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """按微批行号前向, 返回 ``(融合输出, 交付残差)``, 均为 FP16。

        填充行残差原样绕过、归一化输出为零。大中间量在答复后释放, 反向只
        能依据本次前向对应的输入值、逻辑随机位置与步骤权重版本重算。
        """
        st = self._require_bound(step_id)
        if st.status in (StepStatus.COMPLETE, StepStatus.ABORTED):
            raise ValueError(f"训练步 {step_id} 已关闭 ({st.status.value})")
        rows = self._check_rows(st, rows)
        hidden = st.view_main.hidden
        fused = np.empty((len(rows), hidden), dtype=FP16)  # 交付: 融合输出
        residual = np.empty((len(rows), hidden), dtype=FP16)  # 交付: 下游残差
        with _ScratchSession(self.scratch) as scratch:
            scale = scratch.array(hidden)
            xhat = scratch.array(hidden)
            ybuf = scratch.array(hidden)
            scratch.reserve(1)  # mu 标量工作区
            for index, row in enumerate(rows):
                if not self.valid[row]:
                    # 填充行: 残差保持原值, 归一化输出为零
                    residual[index] = self.storage[st.view_residual.storage][
                        st.view_residual.row_offsets(row)
                    ]
                    fused[index] = FP16(0.0)
                    continue
                self._mask_scale_row(st.step_id, self.token_ids[row], scale)
                mu = self._stored_residual_row(st, row, scale, xhat)
                residual[index] = xhat.astype(FP16)  # 储存值即交付残差
                np.divide(xhat, mu, out=ybuf)  # RMS 在 FP32 下计算
                ybuf *= st.weight
                fused[index] = ybuf.astype(FP16)  # 最近偶数舍入至 FP16
        if st.status is StepStatus.REGISTERED:
            st.status = StepStatus.RECOMPUTABLE
        return fused, residual

    # ------------------------------------------------------------------ 反向

    def backward(
        self,
        step_id: int,
        grad_id: str,
        rows: Sequence[int],
        main_grads: Sequence[Sequence],
        residual_grads: Sequence[Sequence],
    ) -> str:
        """按微批行号交付一次上游梯度, 返回处理结果状态。

        主体和副输出的上游梯度都进入; 梯度以储存值重算, 舍入直通。微批可
        乱序完成, 重复提交不造成额外更新; 有效行含非有限数值时整步废弃且
        不发布任何部分更新。
        """
        st = self._require_bound(step_id)
        if st.status in (StepStatus.COMPLETE, StepStatus.ABORTED):
            return st.status.value  # 已关闭: 忽略
        grad_id = str(grad_id)
        if grad_id in st.deliveries:
            return "duplicate"  # 重复提交不造成额外更新
        rows = self._check_rows(st, rows)
        main_grads = self._check_grads(st, main_grads, len(rows))
        residual_grads = self._check_grads(st, residual_grads, len(rows))
        for index, row in enumerate(rows):
            if not self.valid[row]:
                continue  # 填充行梯度为零, 其中的非有限不废弃训练步
            values = (*main_grads[index], *residual_grads[index])
            if any(not _is_finite(value) for value in values):
                st.status = StepStatus.ABORTED
                st.deliveries.add(grad_id)
                return StepStatus.ABORTED.value
        hidden = st.view_main.hidden
        main_acc = st.data_grads[st.view_main.storage]
        residual_acc = st.data_grads[st.view_residual.storage]
        with _ScratchSession(self.scratch) as scratch:
            scale = scratch.array(hidden)
            xhat = scratch.array(hidden)
            dy = scratch.array(hidden)  # 接收梯度的 FP32 转换
            dres = scratch.array(hidden)
            work = scratch.array(hidden)
            scratch.reserve(3)  # mu/dot/factor 标量工作区
            for index, row in enumerate(rows):
                if not self.valid[row]:
                    continue
                token = self.token_ids[row]
                if token not in st.expected_tokens or token in st.delivered_tokens:
                    continue
                self._mask_scale_row(st.step_id, token, scale)
                mu = self._stored_residual_row(st, row, scale, xhat)
                dy[:] = main_grads[index]
                dres[:] = residual_grads[index]
                np.multiply(dy, st.weight, out=work)
                dot = FP32(np.dot(work, xhat))
                factor = FP32(dot / (FP32(hidden) * mu * mu * mu))
                for h in range(hidden):
                    # RMSNorm 反向(FP32, 以储存值计算)
                    work[h] = work[h] / mu - xhat[h] * factor
                work += dres  # 副输出上游梯度进入, 舍入直通
                main_offsets = st.view_main.row_offsets(row)
                residual_offsets = st.view_residual.row_offsets(row)
                for h in range(hidden):
                    # 合并到真正的所有者与偏移(同一存储自然相加)
                    main_acc[main_offsets[h]] += work[h] * scale[h]
                    residual_acc[residual_offsets[h]] += work[h]
                    st.weight_grad[h] += dy[h] * xhat[h] / mu
                st.delivered_tokens.add(token)
        st.deliveries.add(grad_id)
        if st.expected_tokens <= st.delivered_tokens:
            st.status = StepStatus.COMPLETE
        else:
            st.status = StepStatus.AWAITING_GRADIENTS
        return st.status.value

    # ------------------------------------------------------------------ 查询

    def step_status(self, step_id: int) -> str:
        return self._require_step(step_id).status.value

    def step_info(self, step_id: int) -> dict:
        """训练端据此区分可重算、须等待梯度交齐与已终止的训练步。"""
        st = self._require_step(step_id)
        return {
            "step": st.step_id,
            "status": st.status.value,
            "weight_version": st.weight_version,
            "expected_tokens": sorted(st.expected_tokens),
            "delivered_tokens": sorted(st.delivered_tokens),
            "pending_tokens": sorted(st.expected_tokens - st.delivered_tokens),
        }

    def published_gradients(self, step_id: int) -> Optional[dict]:
        """梯度交齐后发布所有者梯度; 未交齐或已废弃时返回 ``None``。"""
        st = self._require_step(step_id)
        if st.status is not StepStatus.COMPLETE:
            return None
        return {
            "step": st.step_id,
            "weight_version": st.weight_version,
            "data": {name: grad.copy() for name, grad in st.data_grads.items()},
            "weight": st.weight_grad.copy(),
        }

    # ------------------------------------------------------------------ 内部

    def _require_step(self, step_id: int) -> _StepState:
        try:
            return self._steps[int(step_id)]
        except KeyError:
            raise ValueError(f"未知训练步: {step_id}") from None

    def _require_bound(self, step_id: int) -> _StepState:
        st = self._require_step(step_id)
        if st.view_main is None:
            raise ValueError(f"训练步 {step_id} 尚未绑定视图")
        return st

    def _check_rows(self, st: _StepState, rows: Sequence[int]) -> List[int]:
        rows = [int(row) for row in rows]
        limit = min(st.view_main.tokens, len(self.token_ids))
        for row in rows:
            if not 0 <= row < limit:
                raise ValueError(f"行号越界: {row}")
        return rows

    def _check_grads(self, st: _StepState, grads, count: int) -> List[list]:
        grads = [list(row) for row in grads]
        if len(grads) != count:
            raise ValueError("梯度行数与微批行号不一致")
        hidden = st.view_main.hidden
        for row in grads:
            if len(row) != hidden:
                raise ValueError("梯度宽度与隐藏维不一致")
        return grads

    def _mask_scale_row(self, step_id: int, token: int, out: np.ndarray) -> None:
        for h in range(out.shape[0]):
            uniform = mask_uniform(self.seed, step_id, token, h)
            out[h] = self.keep_scale if uniform >= self.drop else FP32(0.0)

    def _stored_residual_row(
        self, st: _StepState, row: int, scale: np.ndarray, xhat: np.ndarray
    ) -> FP32:
        """重算一行的储存残差(FP32 读回), 返回 RMS 分母 ``mu``。

        只依据本次前向对应的输入值(原存储)、逻辑随机位置与步骤权重版本。
        """
        main = self.storage[st.view_main.storage]
        residual = self.storage[st.view_residual.storage]
        main_offsets = st.view_main.row_offsets(row)
        residual_offsets = st.view_residual.row_offsets(row)
        for h in range(xhat.shape[0]):
            dropped = FP32(main[main_offsets[h]]) * scale[h]  # 倒置 dropout
            summed = dropped + FP32(residual[residual_offsets[h]])  # FP32 累积
            xhat[h] = FP32(FP16(summed))  # 最近偶数舍入至 FP16 后读回
        mean_square = FP32(np.dot(xhat, xhat)) / FP32(xhat.shape[0])
        return FP32(np.sqrt(mean_square + self.eps))


def replay_capture(capture: dict) -> Tuple[ComputeNode, list]:
    """按采集事件顺序回放, 返回 ``(节点, 记录列表)``。"""
    node = ComputeNode(
        data=capture["data"],
        token_ids=capture["token_ids"],
        valid=capture["valid"],
        seed=capture["seed"],
        drop=capture["drop"],
        eps=capture["eps"],
        scratch_budget=capture.get("scratch_floats"),
    )
    for version, values in capture["weight"].items():
        node.add_weight(version, values)
    views = {name: View.from_spec(spec) for name, spec in capture["views"].items()}
    records = []
    for event in capture["events"]:
        kind = event[0]
        if kind == "step":
            _, step_id, weight_version, tokens = event
            node.register_step(step_id, weight_version, tokens)
            records.append(("step", step_id, weight_version, list(tokens)))
        elif kind == "bind":
            _, step_id, main_name, residual_name = event
            node.bind(step_id, views[main_name], views[residual_name])
            records.append(("bind", step_id, main_name, residual_name))
        elif kind == "forward":
            _, step_id, rows = event
            fused, residual = node.forward(step_id, rows)
            records.append(("forward", step_id, list(rows), fused, residual))
        elif kind == "backward":
            _, step_id, grad_id, rows, main_grads, residual_grads = event
            result = node.backward(
                step_id, grad_id, rows, main_grads, residual_grads
            )
            records.append(("backward", step_id, grad_id, result))
        else:
            raise ValueError(f"未知事件: {kind}")
    return node, records


def _format_array(array: np.ndarray) -> str:
    return np.array2string(array, precision=6, suppress_small=False)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from read_capture import read_capture

    parser = argparse.ArgumentParser(description="回放交接采集并交付梯度")
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    node, records = replay_capture(read_capture(args.capture))
    for index, record in enumerate(records, start=1):
        kind = record[0]
        if kind == "forward":
            _, step_id, rows, fused, residual = record
            print(f"{index:02} forward 步={step_id} 行={rows}")
            print(f"   融合输出 {_format_array(fused)}")
            print(f"   交付残差 {_format_array(residual)}")
        elif kind == "backward":
            _, step_id, grad_id, result = record
            print(f"{index:02} backward 步={step_id} 提交={grad_id} -> {result}")
        elif kind == "step":
            _, step_id, weight_version, tokens = record
            print(f"{index:02} step 步={step_id} 版本={weight_version} token={tokens}")
        else:
            _, step_id, main_name, residual_name = record
            print(f"{index:02} bind 步={step_id} 主={main_name} 残={residual_name}")
    for step_id in sorted(node._steps):
        info = node.step_info(step_id)
        published = node.published_gradients(step_id)
        print(
            f"步 {step_id}: {info['status']} 版本={info['weight_version']} "
            f"待交={info['pending_tokens']}"
        )
        if published is None:
            print("   未发布更新")
        else:
            for name, grad in sorted(published["data"].items()):
                print(f"   所有者 {name}: {_format_array(grad)}")
            print(f"   权重梯度: {_format_array(published['weight'])}")
    budget = node.scratch.budget
    print(f"临时浮点元素: 峰值={node.scratch.peak} 预算={budget}")
