# 低精度残差归一化的前反向重算与梯度交付

`samples/capture.json` 是离线合成的交接采集，包含原始数据和接收顺序。[交接协议](交接协议.md) 描述各字段及上下游共同使用的数值口径。`read_capture` 可读取采集文件并列出操作顺序。

`fused_op` 提供融合 dropout、残差加法与 RMSNorm 的前后向接口，供内存较小的计算节点按交接协议交付梯度：

- `View` 描述 `[storage, offset, shape, strides]` 视图，支持转置、带步长与交叠的非连续布局，索引越界即拒绝。
- `ComputeNode.forward(step, rows)` 按微批行号返回 `(融合输出, 交付残差)`。倒置 dropout 在 FP32 下进行，与残差相加后按最近偶数舍入至 FP16 作为副输出；RMS 与缩放读取储存值在 FP32 下计算并舍入 FP16。掩码只按逻辑 `(seed, step, token, hidden)` 归属，不因压紧填充行、微批拆分或视图布局而变；填充行残差原样绕过、输出与梯度为零。
- `ComputeNode.backward(step, grad_id, rows, main_grads, residual_grads)` 在释放大中间量后重算，只依据本次前向的输入值、逻辑随机位置与步骤登记的权重版本；主体与副输出的上游梯度都进入，按物理存储合并到真正的所有者与偏移。微批可乱序完成，重复提交不造成额外更新；有效行含非有限数值时整步废弃且不发布任何部分更新。
- `step_status` / `step_info` 区分可重算、须等待梯度交齐与已因数值问题终止的训练步；`published_gradients` 仅在梯度交齐后发布所有者梯度与权重梯度。
- 临时浮点元素由 `ScratchLedger` 记账（交付张量与原存储不计），每次请求结束全部释放，峰值不得超过采集给出的 `scratch_floats`。

环境使用 Python 3.11 和锁定的 NumPy。首次在本地准备虚拟环境后运行以下命令。

```bash
python -m pip install -r requirements.txt
python -m read_capture samples/capture.json
python -m fused_op samples/capture.json
python -m unittest discover -s tests -v
```
