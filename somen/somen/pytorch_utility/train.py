import cProfile
import io
import pstats
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from ignite.engine import Events
from ignite.engine.engine import Engine
from ignite.handlers import Checkpoint, DiskSaver
from ignite.utils import convert_tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset

from somen.pytorch_utility import misc
from somen.pytorch_utility.extensions.batch_evaluator import BatchEvaluator
from somen.pytorch_utility.extensions.best_value_snapshot import BestValueSnapshot
from somen.pytorch_utility.extensions.distributed_batch_evaluator import DistributedBatchEvaluator
from somen.pytorch_utility.extensions.distributed_evaluator import DistributedEvaluator
from somen.pytorch_utility.extensions.evaluator import Evaluator
from somen.pytorch_utility.extensions.extension import Extension
from somen.pytorch_utility.extensions.extension_saver import ExtensionSaver
from somen.pytorch_utility.extensions.log_report import LogReport
from somen.pytorch_utility.extensions.lr_scheduler import LRScheduler
from somen.pytorch_utility.extensions.print_report import PrintReport


def _get_defalut_update_fn(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    label_indices: Sequence[int] = [-1],
    non_blocking: bool = False,
) -> Callable:
    def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()

        batch = [convert_tensor(v, device=device, non_blocking=non_blocking) for v in batch]
        inputs = [v for i, v in enumerate(batch) if i not in label_indices and i - len(batch) not in label_indices]
        labels = [batch[i] for i in label_indices]

        y_pred = model(*inputs)
        if isinstance(y_pred, tuple):
            loss = loss_fn(*y_pred, *labels)
        else:
            loss = loss_fn(y_pred, *labels)

        loss.backward()
        optimizer.step()

        return loss.item()

    return _update


def train(
    model: torch.nn.Module,
    params: Mapping[str, Any],
    train_set: Dataset,
    valid_sets: Optional[Sequence[Dataset]] = None,
    valid_names: Optional[Sequence[str]] = None,
    working_dir: Union[str, Path] = "working/",
    load_best: bool = False,
    collate_fn=None,
    ext_extensions: Optional[Sequence[Extension]] = None,
) -> None:
    """

    Args:
        model
        params
        train_set
        working_dir
        load_best
        collect_fn
        handlers: Sequence of (event, handler, args) tuples.

    Returns:
        model: trained model
    """
    working_dir = Path(working_dir)

    # Pickup training parameters
    # optimizer
    optimizer: Union[torch.optim.Optimizer, str] = params.get("optimizer", "Adam")
    optimizer_params: Mapping[str, Any] = params.get("optimizer_params", {"lr": 1e-3})

    # LR scheduler
    lr_scheduler: Optional[str] = params.get("lr_scheduler", None)
    lr_scheduler_params: Optional[Mapping[str, Any]] = params.get("lr_scheduler_params", None)

    # device
    device = params.get("device", "cuda")

    # settings about distributed data parallel
    local_rank: Optional[int] = params.get("local_rank", None)
    if local_rank is not None:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        distributed = True
        global_rank = torch.distributed.get_rank()
        is_main_process = global_rank == 0
    else:
        distributed = False
        global_rank = 0
        is_main_process = True
    sync_batch_norm: bool = params.get("sync_batch_norm", True)
    if local_rank is not None and device == "cuda":
        device = f"cuda:{local_rank}"

    # data loader
    batch_size = params.get("batch_size", 128)
    num_workers = params.get("num_workers", 0)
    pin_memory = params.get("pin_memory", True)

    # trainer
    assert "objective" in params
    objective: Union[Callable, str] = params["objective"]
    label_indices: Sequence[int] = params.get("label_indices", [-1])
    convert_batch_non_blocking: bool = params.get("convert_batch_non_blocking", False)
    get_update_fn = params.get("get_update_fn", None)
    if get_update_fn is None:
        get_update_fn = _get_defalut_update_fn
    train_sampler: Optional[torch.utils.data.Sampler] = params.get("train_sampler", None)

    # evaluator
    metric = params.get("metric", "loss")
    batch_eval = params.get("batch_eval", False)
    take_best_value_snapshot = params.get("take_best_value_snapshot", True)
    maximize = params.get("maximize", False)
    eval_event = misc.str_to_event(params.get("eval_event", "1 epoch"))

    # log report
    log_event = misc.str_to_event(params.get("log_event", "1 epoch"))
    print_report = params.get("print_report", True)

    # trainer snapshot
    take_trainer_snapshot = params.get("take_trainer_snapshot", True)
    trainer_snapshot_event = misc.str_to_event(params.get("trainer_snapshot_event", "1 epoch"))
    trainer_snapshot_n_saved = params.get("trainer_snapshot_n_saved", 1)

    # resume
    resume = params.get("resume", False)

    # number of epochs
    nb_epoch = params.get("nb_epoch", 100)

    # profiling settings
    benchmark_mode: bool = params.get("benchmark_mode", False)
    benchmark_iter: int = params.get("benchmark_iteration", 50)
    benchmark_iter_event = misc.str_to_event(f"{benchmark_iter} iteration")
    enable_cprofile: bool = params.get("enable_cprofile", False)

    # Set up optimizer
    if isinstance(optimizer, str):
        optimizer = getattr(torch.optim, optimizer)(model.parameters(), **optimizer_params)

    # Make a dataloader from train_set
    if train_sampler is None and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        sampler=train_sampler,
    )

    # Build Trainer Engine
    if callable(objective):
        loss_fn = objective
    else:
        if objective == "mse":
            loss_fn = F.mse_loss
        elif objective == "logloss":
            loss_fn = F.nll_loss
        else:
            raise ValueError(f"Given objective '{objective}' is not supported yet.")

    # move to device
    if device.startswith("cuda:"):
        torch.cuda.set_device(device)
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if distributed:
        if sync_batch_norm:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        update_fn = get_update_fn(
            DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True),
            optimizer,
            loss_fn,
            device,
            label_indices,
            convert_batch_non_blocking,
        )
    else:
        update_fn = get_update_fn(model, optimizer, loss_fn, device, label_indices, convert_batch_non_blocking)

    trainer = Engine(update_fn)

    # prepare extensions
    printed_entries = ["elapsed_time", "epoch", "iteration", "train/loss"]
    extensions: List[Extension] = [] if ext_extensions is None else list(ext_extensions)

    # Evaluator
    if valid_sets is not None:
        assert len(valid_sets) > 0
        if valid_names is None:
            if len(valid_sets) == 1:
                valid_names = ["valid"]
            else:
                valid_names = [f"valid{i}" for i in range(len(valid_sets))]

        if len(valid_sets) > 1:
            print(f"Multiple validatio sets are given. `{valid_names[0]}` is used as the best value trigger.")

        # metric: Union[str, Sequence[Union[str, Tuple[str, Callable]], Mapping[str, Callable]]
        if isinstance(metric, str):
            metric = [metric]

        if isinstance(metric, dict):
            metric_functions = metric
        elif isinstance(metric, Sequence):
            metric_functions = {}
            for e in metric:
                if isinstance(e, tuple):
                    metric_name, metric_fn = e
                else:
                    metric_name = e
                    if metric_name == "loss":
                        metric_fn = loss_fn
                    else:
                        metric_fn = misc.get_metric_func(metric_name)
                metric_functions[metric_name] = metric_fn

        # the first metric is used for best value snapshot
        best_value_metric = next(iter(metric_functions.keys()))

        for valid_set, valid_name in zip(valid_sets, valid_names):
            prefix = valid_name + "/"

            evaluator_kwargs = {
                "model": model,
                "prefix": prefix,
                "metric_functions": metric_functions,
                "label_indices": label_indices,
                "device": device,
                "non_blocking": convert_batch_non_blocking,
                "call_event": eval_event,
            }

            if distributed:
                world_size = torch.distributed.get_world_size()
                example_per_process = (len(valid_set) + world_size - 1) // world_size
                sections = [example_per_process * r for r in range(world_size)] + [len(valid_set)]
                valid_loader = torch.utils.data.DataLoader(
                    torch.utils.data.Subset(valid_set, range(sections[global_rank], sections[global_rank + 1])),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    pin_memory=pin_memory,
                )
                if batch_eval:
                    evaluator = DistributedBatchEvaluator(
                        rank=global_rank, data_loader=valid_loader, **evaluator_kwargs
                    )
                else:
                    evaluator = DistributedEvaluator(rank=global_rank, data_loader=valid_loader, **evaluator_kwargs)
            else:
                valid_loader = torch.utils.data.DataLoader(
                    valid_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    pin_memory=pin_memory,
                )
                if batch_eval:
                    evaluator = BatchEvaluator(data_loader=valid_loader, **evaluator_kwargs)
                else:
                    evaluator = Evaluator(data_loader=valid_loader, **evaluator_kwargs)

            extensions.append(evaluator)
            for key in metric_functions.keys():
                printed_entries.append(prefix + key)

        if take_best_value_snapshot:
            extensions.append(
                BestValueSnapshot(
                    model, working_dir / "best.pth", valid_names[0] + "/" + best_value_metric, maximize, eval_event
                )
            )

    # LR scheduler
    if lr_scheduler is not None or lr_scheduler_params is not None:
        extensions.append(LRScheduler(optimizer, optimizer_params["lr"], nb_epoch, lr_scheduler, lr_scheduler_params))
        printed_entries.append("lr")

    # LogReport
    extensions.append(LogReport(working_dir / "log.json", nb_epoch, log_event))
    if print_report:
        extensions.append(PrintReport(printed_entries))

    # Attach extensions to trainer
    extensions = sorted(extensions, key=lambda x: x.priority, reverse=True)

    @trainer.on(Events.STARTED)
    def _extension_started(engine: Engine) -> None:
        for e in extensions:
            if e.main_process_only and not is_main_process:
                continue
            e.started(engine)

    @trainer.on(Events.ITERATION_COMPLETED)
    def _extension_iteration_completed(engine: Engine) -> None:
        engine.state.metrics["observation"] = {"train/loss": engine.state.output}
        for e in extensions:
            if e.main_process_only and not is_main_process:
                continue
            e.iteration_completed(engine)

    if train_sampler is not None and hasattr(train_sampler, "set_epoch"):

        @trainer.on(Events.EPOCH_STARTED)
        def _set_epoch_distributed_sampler(engine: Engine) -> None:
            train_sampler.set_epoch(engine.state.epoch)

    for e in extensions:
        if e.main_process_only and not is_main_process:
            continue
        if benchmark_mode:
            trainer.add_event_handler(benchmark_iter_event, e)
        else:
            trainer.add_event_handler(e.call_event, e)

    if benchmark_mode:
        # Add handler for stop training
        @trainer.on(Events.ITERATION_COMPLETED(once=benchmark_iter))
        def _end_benchmark(engine: Engine) -> None:
            print("Stop benchmark!")
            engine.terminate()

    # Snapshot for resume
    objects_to_checkpoint = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "extensions": ExtensionSaver(extensions),
    }
    if take_trainer_snapshot and is_main_process:
        trainer.add_event_handler(
            trainer_snapshot_event,
            Checkpoint(
                to_save=objects_to_checkpoint,
                save_handler=DiskSaver(working_dir, require_empty=False),
                n_saved=trainer_snapshot_n_saved,
            ),
        )

    # Resume
    if resume:
        check_points = list(working_dir.glob("checkpoint*.pth"))
        if len(check_points) > 0:
            check_points = sorted(check_points, key=lambda x: x.stat().st_mtime)
            print(f"Checkpoints are found. `{check_points[-1]}` will be loaded.")
            Checkpoint.load_objects(
                to_load=objects_to_checkpoint, checkpoint=torch.load(check_points[-1], map_location=device)
            )
        else:
            print("No Checkpoint is found.")

    if enable_cprofile and is_main_process:
        pr = cProfile.Profile()
        pr.enable()

    # Start training!
    trainer.run(train_loader, max_epochs=nb_epoch)

    if enable_cprofile and is_main_process:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats()
        print(s.getvalue())

        pr.dump_stats(working_dir / "benchmark.cprofile")

    # load a model snapshot which achieve the best validation score
    if take_best_value_snapshot and load_best:
        if distributed:
            torch.distributed.barrier()
        model.load_state_dict(torch.load(working_dir / "best.pth"))

    # # learning rate decay
    # learning_rate_decay = params.get("learning_rate_decay", -1)
    # learning_rate_decay_trigger = params.get("learning_rate_decay_trigger", "1 epoch")
    # learning_rate_decay_trigger = misc.str_to_trigger(learning_rate_decay_trigger)
    # if learning_rate_decay >= 0:
    #     trainer.extend(
    #         extensions.ExponentialShift(decay_attr, learning_rate_decay), trigger=learning_rate_decay_trigger
    #     )

    # # L2 regularization
    # reg_l2 = params.get("reg_l2", -1)
    # reg_l2_ignore_biases = params.get("reg_l2_ignore_biases", False)
    # if reg_l2 >= 0:
    #     if reg_l2_ignore_biases:
    #         for param in learning_model.params():
    #             if param.name != "b":  # バイアス以外だったら
    #                 param.update_rule.add_hook(chainer.optimizer_hooks.WeightDecay(reg_l2))
    #     else:
    #         optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(reg_l2))

    # # learning rate step shift
    # learning_rate_step_shift = params.get("learning_rate_step_shift", False)
    # if learning_rate_step_shift:
    #     shift_epochs = [nb_epoch // 2, nb_epoch * 3 // 4]
    #     trainer.extend(
    #         extensions.ExponentialShift(decay_attr, 0.1),
    #         trigger=training.triggers.ManualScheduleTrigger(shift_epochs, "epoch"),
    #     )

    # # log learning rate
    # trainer.extend(extensions.observe_lr(observation_key=decay_attr))

    # # progress bar
    # print_progress = params.get("print_progress", False)
    # if print_progress:
    #     trainer.extend(extensions.ProgressBar())

    # # clean up snapshots
    # remove_trainer_snapshot = params.get("remove_trainer_snapshot", True)
    # if take_trainer_snapshot and remove_trainer_snapshot:
    #     os.remove(trainer_path)

    return model
