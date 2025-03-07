from typing import TYPE_CHECKING, List, Optional

import torch
import torch.utils.data.dataloader
import tqdm
from compressed_tensors.utils import get_execution_device

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs
from llmcompressor.utils.helpers import calibration_forward_context

if TYPE_CHECKING:
    from llmcompressor.modifiers import Modifier

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    sequential_targets: List[str],
    ignore: List[str],
    callback_modifier: Optional["Modifier"] = None,
):
    """
    按照以下步骤运行顺序数据管道：
    1.根据 `sequential_targets` 将模型划分为子图
    2.数据按顺序通过每个子图。数据通过每个子图两次，一次是为了触发校准钩子，第二次是为了捕捉通过钩子发生量化后的激活。
    3. 每个子图之间的中间激活被缓存，并在每个批次之间卸载到 CPU，以节省内存。

    该管道要求模型可追溯数据加载器的数据。这对于使用视觉数据集的视觉语言模型来说可能是个问题，因为模型中需要进行专门的输入处理。

    如果跟踪失败，将引发 torch.fx.proxy.TraceError。
    可以通过封装不可跟踪函数（请参阅 llmcompressor.transformers.tracing）来使模型可跟踪


    Run a sequential data pipeline according to the following steps:

    1. The model is partitioned into subgraphs according to `sequential_targets`
    2. Data passes through each subgraph sequentially. Data is passed through each
        subgraph twice, once to trigger calibration hooks, then a second time in order
        to capture activations after quantization has occurred through the hooks.
    3. The intermediate activations between each subgraph are cached and offloaded to
        the cpu between each batch in order to save memory

    This pipeline requires that the model be traceable with respect to data from the
    data loader. This may be an issue for vision language models with vision datasets,
    due to specialized input processing in the model.

    In the event that tracing fails, a torch.fx.proxy.TraceError will be raised. A model
    can be made traceable by wrapping the untraceable functions (see
    llmcompressor.transformers.tracing)

    :param model: model being calibrated
    :param dataloader: loads data for calibration
    :param sequential_targets: patterns which match to the layer modules of the model
    :param ignore: patterns which match to modules which should be ignored by tracing
    """
    # trace subgraphs
    sample_input = next(iter(dataloader))
    subgraphs = trace_subgraphs(model, sample_input, sequential_targets, ignore)

    with calibration_forward_context(model):
        # prepare intermediates cache
        model_device = get_execution_device(model)
        intermediates = IntermediatesCache.from_dataloader(dataloader, model_device)

        num_subgraphs = len(subgraphs)
        for subgraph_index, subgraph in enumerate(subgraphs):
            # prepare tqdm description texts
            calib_desc = f"({subgraph_index + 1}/{num_subgraphs}): Calibrating"
            prop_desc = f"({subgraph_index + 1}/{num_subgraphs}): Propagating"

            # compile subgraph forward function
            forward_function = subgraph.compile_forward()

            # do an preliminary pass to trigger modifier hooks
            for batch_index in tqdm.tqdm(range(len(dataloader)), desc=calib_desc):
                inputs = intermediates.fetch(batch_index, subgraph.input_names)
                forward_function(model, **inputs)

            # TODO: replace with a lifecycle event
            if callback_modifier:
                callback_modifier.on_sequential_batch_end()

            # this pass does not trigger modifier hooks
            # and is only used for capturing outputs from the newly compressed modules
            with HooksMixin.disable_hooks():
                for batch_index in tqdm.tqdm(range(len(dataloader)), desc=prop_desc):
                    inputs = intermediates.fetch(batch_index, subgraph.input_names)
                    output = forward_function(model, **inputs)

                    if subgraph_index < num_subgraphs - 1:
                        intermediates.update(batch_index, output)
                        intermediates.delete(batch_index, subgraph.consumed_names)
