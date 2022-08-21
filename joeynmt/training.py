# coding: utf-8
"""
Training module
"""
import argparse
import heapq
import logging
import math
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Tuple
import torch
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from batch import Batch
from builders import build_gradient_clipper, build_optimizer, build_scheduler
from data import load_data, make_data_iter
from helpers import (
    delete_ckpt,
    load_checkpoint,
    load_config,
    log_cfg,
    make_logger,
    make_model_dir,
    parse_train_args,
    set_seed,
    store_attention_plots,
    symlink_update,
    write_list_to_file,
)
from model import Model, _DataParallel, build_model
from prediction import predict, test
import torch
import torch.nn.functional as F
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
    SubsetRandomSampler
)
from data import collate_fn
from functools import partial

# for fp16 training
try:
    from apex import amp

    amp.register_half_function(torch, "einsum")
except ImportError as no_apex:  # noqa: F841
    # error handling in TrainManager object construction
    pass

logger = logging.getLogger(__name__)


class TrainManager:
    """
    Manages training loop, validations, learning rate scheduling
    and early stopping.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, model: Model, cfg: dict) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.
        Note: no need to pass batch_class here. see make_data_iter()

        :param model: torch module defining the model
        :param cfg: dictionary containing the training configurations
        """
        (  # pylint: disable=unbalanced-tuple-unpacking
            model_dir,
            load_model,
            load_encoder,
            load_decoder,
            loss_type,
            label_smoothing,
            normalization,
            learning_rate_min,
            keep_best_ckpts,
            logging_freq,
            validation_freq,
            log_valid_sents,
            early_stopping_metric,
            seed,
            shuffle,
            epochs,
            max_updates,
            batch_size,
            batch_type,
            batch_multiplier,
            device,
            n_gpu,
            num_workers,
            fp16,
            reset_best_ckpt,
            reset_scheduler,
            reset_optimizer,
            reset_iter_state,
        ) = parse_train_args(cfg["training"])

        # logging and storing
        self.model_dir = model_dir
        self.tb_writer = SummaryWriter(log_dir=(model_dir / "tensorboard").as_posix())
        self.logging_freq = logging_freq
        self.validation_freq = validation_freq
        self.log_valid_sents = log_valid_sents

        # model
        self.model = model
        self.model.log_parameters_list()
        self.model.loss_function = (loss_type, label_smoothing)
        logger.info(self.model)

        # CPU / GPU
        self.device = device
        self.n_gpu = n_gpu
        self.num_workers = num_workers
        if self.device.type == "cuda":
            self.model.to(self.device)

        # optimization
        self.clip_grad_fun = build_gradient_clipper(config=cfg["training"])
        self.optimizer = build_optimizer(config=cfg["training"],
                                         parameters=self.model.parameters())

        # save/delete checkpoints
        self.num_ckpts = keep_best_ckpts
        self.ckpt_queue: List[Tuple[float, Path]] = []  # heap queue

        # early_stopping
        self.early_stopping_metric = early_stopping_metric
        # early_stopping_metric decides on how to find the early stopping point: ckpts
        # are written when there's a new high/low score for this metric. If we schedule
        # after loss/ppl, we want to minimize the score, else we want to maximize it.
        if self.early_stopping_metric in ["ppl", "loss"]:  # lower is better
            self.minimize_metric = True
        elif self.early_stopping_metric in ["acc", "bleu", "chrf"]:  # higher is better
            self.minimize_metric = False

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=cfg["training"],
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=cfg["model"]["encoder"]["hidden_size"],
        )

        # data & batch handling
        self.seed = seed
        self.shuffle = shuffle
        self.epochs = epochs
        self.max_updates = max_updates
        self.max_updates = max_updates
        self.batch_size = batch_size
        self.batch_type = batch_type
        self.learning_rate_min = learning_rate_min
        self.batch_multiplier = batch_multiplier
        self.normalization = normalization

        # Placeholder so that we can use the train_iter in other functions.
        self.train_iter, self.train_iter_state = None, None

        # initialize training statistics
        self.stats = self.TrainStatistics(
            steps=0,
            is_min_lr=False,
            is_max_update=False,
            total_tokens=0,
            best_ckpt_iter=0,
            best_ckpt_score=np.inf if self.minimize_metric else -np.inf,
            minimize_metric=self.minimize_metric,
        )

        # fp16
        self.fp16 = fp16
        if self.fp16:
            if "apex" not in sys.modules:
                # pylint: disable=used-before-assignment
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex "
                    "to use fp16 training.") from no_apex  # noqa: F821

            self.model, self.optimizer = amp.initialize(self.model,
                                                        self.optimizer,
                                                        opt_level="O1")
            # opt level: one of {"O0", "O1", "O2", "O3"}
            # see https://nvidia.github.io/apex/amp.html#opt-levels

        # load model parameters
        if load_model is not None:
            self.init_from_checkpoint(
                load_model,
                reset_best_ckpt=reset_best_ckpt,
                reset_scheduler=reset_scheduler,
                reset_optimizer=reset_optimizer,
                reset_iter_state=reset_iter_state,
            )
        for layer_name, load_path in [
            ("encoder", load_encoder),
            ("decoder", load_decoder),
        ]:
            if load_path is not None:
                self.init_layers(path=load_path, layer=layer_name)

        # gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            self.model = _DataParallel(self.model)

        # config for generation
        self.valid_cfg = cfg["testing"].copy()
        self.valid_cfg["beam_size"] = 1  # greedy decoding during train loop
        # in greedy decoding, we use the same batch_size as the one in training
        self.valid_cfg["batch_size"] = self.batch_size
        self.valid_cfg["batch_type"] = self.batch_type
        # no further exploration during training
        self.valid_cfg["n_best"] = 1
        # self.valid_cfg["return_attention"] = False  # don't override this param
        self.valid_cfg["return_prob"] = "none"
        self.valid_cfg["generate_unk"] = True
        self.valid_cfg["repetition_penalty"] = -1  # turn off
        self.valid_cfg["no_repeat_ngram_size"] = -1  # turn off

    def _save_checkpoint(self, new_best: bool, score: float) -> None:
        """
        Save the model's current parameters and the training state to a checkpoint.

        The training state contains the total number of training steps, the total number
        of training tokens, the best checkpoint score and iteration so far, and
        optimizer and scheduler states.

        :param new_best: This boolean signals which symlink we will use for the new
            checkpoint. If it is true, we update best.ckpt.
        :param score: Validation score which is used as key of heap queue. if score is
            float('nan'), the queue won't be updated.
        """
        model_path = Path(self.model_dir) / f"{self.stats.steps}.ckpt"
        model_state_dict = (self.model.module.state_dict() if isinstance(
            self.model, torch.nn.DataParallel) else self.model.state_dict())
        state = {
            "steps":
            self.stats.steps,
            "total_tokens":
            self.stats.total_tokens,
            "best_ckpt_score":
            self.stats.best_ckpt_score,
            "best_ckpt_iteration":
            self.stats.best_ckpt_iter,
            "model_state":
            model_state_dict,
            "optimizer_state":
            self.optimizer.state_dict(),
            "scheduler_state":
            (self.scheduler.state_dict() if self.scheduler is not None else None),
            "amp_state":
            amp.state_dict() if self.fp16 else None,
            "train_iter_state":
            (self.train_iter.batch_sampler.sampler.generator.get_state()),
        }
        torch.save(state, model_path.as_posix())

        # update symlink
        symlink_target = Path(f"{self.stats.steps}.ckpt")
        # last symlink
        last_path = Path(self.model_dir) / "latest.ckpt"
        prev_path = symlink_update(symlink_target, last_path)  # update always
        # best symlink
        best_path = Path(self.model_dir) / "best.ckpt"
        if new_best:
            prev_path = symlink_update(symlink_target, best_path)
            assert best_path.resolve().stem == str(self.stats.best_ckpt_iter)

        # push to and pop from the heap queue
        to_delete = None
        if not math.isnan(score) and self.num_ckpts > 0:
            if len(self.ckpt_queue) < self.num_ckpts:  # no pop, push only
                heapq.heappush(self.ckpt_queue, (score, model_path))
            else:  # push + pop the worst one in the queue
                if self.minimize_metric:
                    # pylint: disable=protected-access
                    heapq._heapify_max(self.ckpt_queue)
                    to_delete = heapq._heappop_max(self.ckpt_queue)
                    heapq.heappush(self.ckpt_queue, (score, model_path))
                    # pylint: enable=protected-access
                else:
                    to_delete = heapq.heappushpop(self.ckpt_queue, (score, model_path))

            if to_delete is not None:
                assert to_delete[1] != model_path  # don't delete the last ckpt
                if to_delete[1].stem != best_path.resolve().stem:
                    delete_ckpt(to_delete[1])  # don't delete the best ckpt

            assert len(self.ckpt_queue) <= self.num_ckpts

            # remove old symlink target if not in queue after push/pop
            if prev_path is not None and prev_path.stem not in [
                    c[1].stem for c in self.ckpt_queue
            ]:
                delete_ckpt(prev_path)

    def init_from_checkpoint(
        self,
        path: Path,
        reset_best_ckpt: bool = False,
        reset_scheduler: bool = False,
        reset_optimizer: bool = False,
        reset_iter_state: bool = False,
    ) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint,
                                use for domain adaptation with a new dev
                                set or when using a new metric for fine-tuning.
        :param reset_scheduler: reset the learning rate scheduler, and do not
                                use the one stored in the checkpoint.
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint.
        :param reset_iter_state: reset the sampler's internal state and do not
                                use the one stored in the checkpoint.
        """
        logger.info("Loading model from %s", path)
        model_checkpoint = load_checkpoint(path=path, device=self.device)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            logger.info("Reset optimizer.")

        if not reset_scheduler:
            if (model_checkpoint["scheduler_state"] is not None
                    and self.scheduler is not None):
                self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])
        else:
            logger.info("Reset scheduler.")

        if not reset_best_ckpt:
            self.stats.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.stats.best_ckpt_iter = model_checkpoint["best_ckpt_iteration"]
        else:
            logger.info("Reset tracking of the best checkpoint.")

        if not reset_iter_state:
            # restore counters
            assert "train_iter_state" in model_checkpoint
            self.stats.steps = model_checkpoint["steps"]
            self.stats.total_tokens = model_checkpoint["total_tokens"]
            self.train_iter_state = model_checkpoint["train_iter_state"]
        else:
            # reset counters if explicitly 'train_iter_state: True' in config
            logger.info("Reset data iterator (random seed: {%d}).", self.seed)

        # move to gpu
        if self.device.type == "cuda":
            self.model.to(self.device)

        # fp16
        if self.fp16 and model_checkpoint.get("amp_state", None) is not None:
            amp.load_state_dict(model_checkpoint["amp_state"])

    def init_layers(self, path: Path, layer: str) -> None:
        """
        Initialize encoder decoder layers from a given checkpoint file.

        :param path: path to checkpoint
        :param layer: layer name; 'encoder' or 'decoder' expected
        """
        assert path is not None
        layer_state_dict = OrderedDict()
        logger.info("Loading %s laysers from %s", layer, path)
        ckpt = load_checkpoint(path=path, device=self.device)
        for k, v in ckpt["model_state"].items():
            if k.startswith(layer):
                layer_state_dict[k] = v
        self.model.load_state_dict(layer_state_dict, strict=False)

    def train_and_validate(self, train_data: Dataset, valid_data: Dataset) -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        # pylint: disable=too-many-branches,too-many-statements
        self.train_iter = make_data_iter(
            dataset=train_data,
            batch_size=self.batch_size,
            batch_type=self.batch_type,
            seed=self.seed,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            device=self.device,
            pad_index=self.model.pad_index,
        )

        if self.train_iter_state is not None:
            self.train_iter.batch_sampler.sampler.generator.set_state(
                self.train_iter_state.cpu())

        #################################################################
        # simplify accumulation logic:
        #################################################################
        # for epoch in range(epochs):
        #     self.model.zero_grad()
        #     epoch_loss = 0.0
        #     batch_loss = 0.0
        #     for i, batch in enumerate(self.train_iter):
        #
        #         # gradient accumulation:
        #         # loss.backward() inside _train_step()
        #         batch_loss += self._train_step(inputs)
        #
        #         if (i + 1) % self.batch_multiplier == 0:
        #             self.optimizer.step()     # update!
        #             self.model.zero_grad()    # reset gradients
        #             self.steps += 1           # increment counter
        #
        #             epoch_loss += batch_loss  # accumulate batch loss
        #             batch_loss = 0            # reset batch loss
        #
        #     # leftovers are just ignored.
        #################################################################

        logger.info(
            "Train stats:\n"
            "\tdevice: %s\n"
            "\tn_gpu: %d\n"
            "\t16-bits training: %r\n"
            "\tgradient accumulation: %d\n"
            "\tbatch size per device: %d\n"
            "\teffective batch size (w. parallel & accumulation): %d",
            self.device.type,  # next(self.model.parameters()).device
            self.n_gpu,
            self.fp16,
            self.batch_multiplier,
            self.batch_size // self.n_gpu if self.n_gpu > 1 else self.batch_size,
            self.batch_size * self.batch_multiplier,
        )

        try:
            for epoch_no in range(self.epochs):
                logger.info("EPOCH %d", epoch_no + 1)

                if self.scheduler_step_at == "epoch":
                    self.scheduler.step(epoch=epoch_no)

                self.model.train()

                # Reset statistics for each epoch.
                start = time.time()
                total_valid_duration = 0
                start_tokens = self.stats.total_tokens
                self.model.zero_grad()
                epoch_loss = 0
                total_batch_loss = 0
                total_n_correct = 0

                # subsample train data each epoch
                if train_data.random_subset > 0:
                    try:
                        train_data.reset_random_subset()
                        train_data.sample_random_subset(seed=epoch_no)
                        logger.info(
                            "Sample random subset from dev set: n=%d, seed=%d",
                            len(train_data),
                            epoch_no,
                        )
                    except AssertionError as e:
                        logger.warning(e)

                batch: Batch  # yield a joeynmt Batch object
                for i, batch in enumerate(self.train_iter):
                    # sort batch now by src length and keep track of order
                    batch.sort_by_src_length()

                    # get batch loss
                    norm_batch_loss, n_correct = self._train_step(batch)
                    total_batch_loss += norm_batch_loss
                    total_n_correct += n_correct

                    # update!
                    if (i + 1) % self.batch_multiplier == 0:
                        # clip gradients (in-place)
                        if self.clip_grad_fun is not None:
                            if self.fp16:
                                self.clip_grad_fun(
                                    parameters=amp.master_params(self.optimizer))
                            else:
                                self.clip_grad_fun(parameters=self.model.parameters())

                        # make gradient step
                        self.optimizer.step()

                        # decay lr
                        if self.scheduler_step_at == "step":
                            self.scheduler.step(self.stats.steps)

                        # reset gradients
                        self.model.zero_grad()

                        # increment step counter
                        self.stats.steps += 1
                        if self.stats.steps >= self.max_updates:
                            self.stats.is_max_update = True

                        # log learning progress
                        if self.stats.steps % self.logging_freq == 0:
                            elapsed = time.time() - start - total_valid_duration
                            elapsed_tok = self.stats.total_tokens - start_tokens
                            token_accuracy = total_n_correct / elapsed_tok
                            self.tb_writer.add_scalar("train/batch_loss",
                                                      total_batch_loss,
                                                      self.stats.steps)
                            self.tb_writer.add_scalar("train/batch_acc", token_accuracy,
                                                      self.stats.steps)
                            logger.info(
                                "Epoch %3d, "
                                "Step: %8d, "
                                "Batch Loss: %12.6f, "
                                "Batch Acc: %.6f, "
                                "Tokens per Sec: %8.0f, "
                                "Lr: %.6f",
                                epoch_no + 1,
                                self.stats.steps,
                                total_batch_loss,
                                token_accuracy,
                                elapsed_tok / elapsed,
                                self.optimizer.param_groups[0]["lr"],
                            )
                            start = time.time()
                            total_valid_duration = 0
                            start_tokens = self.stats.total_tokens

                        # update epoch_loss
                        epoch_loss += total_batch_loss  # accumulate loss
                        total_batch_loss = 0  # reset batch loss
                        total_n_correct = 0  # reset batch accuracy

                        # validate on the entire dev set
                        if self.stats.steps % self.validation_freq == 0:
                            valid_duration = self._validate(valid_data)
                            total_valid_duration += valid_duration

                        # check current_lr
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        if current_lr < self.learning_rate_min:
                            self.stats.is_min_lr = True

                        self.tb_writer.add_scalar("train/learning_rate", current_lr,
                                                  self.stats.steps)

                    if self.stats.is_min_lr or self.stats.is_max_update:
                        break

                if self.stats.is_min_lr or self.stats.is_max_update:
                    log_str = (f"minimum lr {self.learning_rate_min}"
                               if self.stats.is_min_lr else
                               f"maximum num. of updates {self.max_updates}")
                    logger.info("Training ended since %s was reached.", log_str)
                    break

                logger.info(
                    "Epoch %3d: total training loss %.2f",
                    epoch_no + 1,
                    epoch_loss,
                )
            else:
                logger.info("Training ended after %3d epochs.", epoch_no + 1)
            logger.info(
                "Best validation result (greedy) "
                "at step %8d: %6.2f %s.",
                self.stats.best_ckpt_iter,
                self.stats.best_ckpt_score,
                self.early_stopping_metric,
            )
        except KeyboardInterrupt:
            self._save_checkpoint(False, float("nan"))

        self.tb_writer.close()  # close Tensorboard writer

    def _train_step(self, batch: Batch) -> Tensor:
        """
        Train the model on one batch: Compute the loss.

        :param batch: training batch
        :return:
            - losses for batch (sum)
            - number of correct tokens for batch (sum)
        """
        # reactivate training
        self.model.train()

        # get loss (run as during training with teacher forcing)
        batch_loss, _, _, correct_tokens = self.model(return_type="loss", **vars(batch))

        # normalize batch loss
        norm_batch_loss = batch.normalize(
            batch_loss,
            normalization=self.normalization,
            n_gpu=self.n_gpu,
            n_accumulation=self.batch_multiplier,
        )

        # sum over multiple gpus
        n_correct_tokens = batch.normalize(correct_tokens, "sum", self.n_gpu)

        # accumulate gradients
        if self.fp16:
            with amp.scale_loss(norm_batch_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            norm_batch_loss.backward()

        # increment token counter
        self.stats.total_tokens += batch.ntokens

        return norm_batch_loss.item(), n_correct_tokens.item()

    def _validate(self, valid_data: Dataset):
        if valid_data.random_subset > 0:  # subsample validation set each valid step
            try:
                valid_data.reset_random_subset()
                valid_data.sample_random_subset(seed=self.stats.steps)
                logger.info(
                    "Sample random subset from dev set: n=%d, seed=%d",
                    len(valid_data),
                    self.stats.steps,
                )
            except AssertionError as e:
                logger.warning(e)

        valid_start_time = time.time()
        (
            valid_scores,
            valid_references,
            valid_hypotheses,
            valid_hypotheses_raw,
            valid_sequence_scores,  # pylint: disable=unused-variable
            valid_attention_scores,
        ) = predict(
            model=self.model,
            data=valid_data,
            compute_loss=True,
            device=self.device,
            n_gpu=self.n_gpu,
            normalization=self.normalization,
            cfg=self.valid_cfg,
        )
        valid_duration = time.time() - valid_start_time

        # for eval_metric in ['loss', 'ppl', 'acc'] + self.eval_metrics:
        for eval_metric, score in valid_scores.items():
            if not math.isnan(score):
                self.tb_writer.add_scalar(f"valid/{eval_metric}", score,
                                          self.stats.steps)

        ckpt_score = valid_scores[self.early_stopping_metric]

        if self.scheduler_step_at == "validation":
            self.scheduler.step(metrics=ckpt_score)

        # update new best
        new_best = self.stats.is_best(ckpt_score)
        if new_best:
            self.stats.best_ckpt_score = ckpt_score
            self.stats.best_ckpt_iter = self.stats.steps
            logger.info(
                "Hooray! New best validation result [%s]!",
                self.early_stopping_metric,
            )

        # save checkpoints
        is_better = (self.stats.is_better(ckpt_score, self.ckpt_queue)
                     if len(self.ckpt_queue) > 0 else True)
        if self.num_ckpts < 0 or is_better:
            self._save_checkpoint(new_best, ckpt_score)

        # append to validation report
        self._add_report(valid_scores=valid_scores, new_best=new_best)

        self._log_examples(
            references=valid_references,
            hypotheses=valid_hypotheses,
            hypotheses_raw=valid_hypotheses_raw,
            data=valid_data,
        )

        # store validation set outputs
        write_list_to_file(self.model_dir / f"{self.stats.steps}.hyps",
                           valid_hypotheses)

        # store attention plots for selected valid sentences
        if valid_attention_scores:
            store_attention_plots(
                attentions=valid_attention_scores,
                targets=valid_hypotheses_raw,
                sources=valid_data.get_list(lang=valid_data.src_lang, tokenized=True),
                indices=self.log_valid_sents,
                output_prefix=(self.model_dir / f"att.{self.stats.steps}").as_posix(),
                tb_writer=self.tb_writer,
                steps=self.stats.steps,
            )

        return valid_duration

    def _add_report(self, valid_scores: dict, new_best: bool = False) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_scores: validation evaluation score [eval_metric]
        :param new_best: whether this is a new best model
        """
        current_lr = self.optimizer.param_groups[0]["lr"]

        valid_file = self.model_dir / "validations.txt"
        with valid_file.open("a", encoding="utf-8") as opened_file:
            score_str = "\t".join([f"Steps: {self.stats.steps}"] + [
                f"{eval_metric}: {score:.5f}"
                for eval_metric, score in valid_scores.items() if not math.isnan(score)
            ] + [f"LR: {current_lr:.8f}", "*" if new_best else ""])
            opened_file.write(f"{score_str}\n")

    def _log_examples(
        self,
        hypotheses: List[str],
        references: List[str],
        hypotheses_raw: List[List[str]],
        data: Dataset,
    ) -> None:
        """
        Log the first `self.log_valid_sents` sentences from given examples.

        :param hypotheses: decoded hypotheses (list of strings)
        :param references: decoded references (list of strings)
        :param hypotheses_raw: raw hypotheses (list of list of tokens)
        :param data: Dataset
        """
        for p in self.log_valid_sents:
            if p >= len(hypotheses):
                continue
            logger.info("Example #%d", p)

            # tokenized text
            tokenized_src = data.get_item(idx=p, lang=data.src_lang)
            tokenized_trg = data.get_item(idx=p, lang=data.trg_lang)
            logger.debug("\tTokenized source:     %s", tokenized_src)
            logger.debug("\tTokenized reference:  %s", tokenized_trg)
            logger.debug("\tTokenized hypothesis: %s", hypotheses_raw[p])

            # detokenized text
            logger.info("\tSource:     %s", data.src[p])
            logger.info("\tReference:  %s", references[p])
            logger.info("\tHypothesis: %s", hypotheses[p])

    class TrainStatistics:

        def __init__(
            self,
            steps: int = 0,
            is_min_lr: bool = False,
            is_max_update: bool = False,
            total_tokens: int = 0,
            best_ckpt_iter: int = 0,
            best_ckpt_score: float = np.inf,
            minimize_metric: bool = True,
        ) -> None:
            self.steps = steps  # global update step counter
            self.is_min_lr = is_min_lr  # stop by reaching learning rate minimum
            self.is_max_update = is_max_update  # stop by reaching max num of updates
            self.total_tokens = total_tokens  # number of total tokens seen so far
            self.best_ckpt_iter = best_ckpt_iter  # store iteration point of best ckpt
            self.best_ckpt_score = best_ckpt_score  # initial values for best scores
            self.minimize_metric = minimize_metric  # minimize or maximize score

        def is_best(self, score):
            if self.minimize_metric:
                is_best = score < self.best_ckpt_score
            else:
                is_best = score > self.best_ckpt_score
            return is_best

        def is_better(self, score: float, heap_queue: list):
            assert len(heap_queue) > 0
            if self.minimize_metric:
                is_better = score < heapq.nlargest(1, heap_queue)[0][0]
            else:
                is_better = score > heapq.nsmallest(1, heap_queue)[0][0]
            return is_better


def train(cfg_file: str, skip_test: bool = False) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    :param skip_test: whether a test should be run or not after training
    """
    # read config file
    cfg = load_config(Path(cfg_file))

    # make logger
    model_dir = make_model_dir(
        Path(cfg["training"]["model_dir"]),
        overwrite=cfg["training"].get("overwrite", False),
    )
    joeynmt_version = make_logger(model_dir, mode="train")
    if "joeynmt_version" in cfg:
        assert str(joeynmt_version) == str(cfg["joeynmt_version"]), (
            f"You are using JoeyNMT version {joeynmt_version}, "
            f'but {cfg["joeynmt_version"]} is expected in the given config.')
    # TODO: save version number in model checkpoints

    # write all entries of config to the log
    log_cfg(cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, (model_dir / "config.yaml").as_posix())

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # load the data
    src_vocab, trg_vocab, train_data, dev_data, test_data = load_data(
        data_cfg=cfg["data"])

    # store the vocabs and tokenizers
    src_vocab.to_file(model_dir / "src_vocab.txt")
    if hasattr(train_data.tokenizer[train_data.src_lang], "copy_cfg_file"):
        train_data.tokenizer[train_data.src_lang].copy_cfg_file(model_dir)
    trg_vocab.to_file(model_dir / "trg_vocab.txt")
    if hasattr(train_data.tokenizer[train_data.trg_lang], "copy_cfg_file"):
        train_data.tokenizer[train_data.trg_lang].copy_cfg_file(model_dir)

    # build an encoder-decoder model
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, cfg=cfg)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    if not skip_test:
        # predict with the best model on validation and test
        # (if test data is available)

        ckpt = model_dir / f"{trainer.stats.best_ckpt_iter}.ckpt"
        output_path = model_dir / f"{trainer.stats.best_ckpt_iter:08d}.hyps"

        datasets_to_test = {
            "dev": dev_data,
            "test": test_data,
            "src_vocab": src_vocab,
            "trg_vocab": trg_vocab,
        }
        test(
            cfg_file,
            ckpt=ckpt.as_posix(),
            output_path=output_path.as_posix(),
            datasets=datasets_to_test,
        )
    else:
        logger.info("Skipping test after training")

        
        
        

'''
Each query strategy below returns a list of len=query_size with indices of 
samples that are to be queried.

Arguments:
- model (torch.nn.Module): not needed for `random_query`
- device (torch.device): not needed for `random_query`
- dataloader (torch.utils.data.DataLoader)
- query_size (int): number of samples to be queried for labels (default=10)

'''
def random_query(data_loader, query_size=10):
    
    sample_idx = []
    
    # Because the data has already been shuffled inside the data loader,
    # we can simply return the `query_size` first samples from it
    for batch in data_loader:
        
        _, _, idx = batch
        sample_idx.extend(idx.tolist())

        if len(sample_idx) >= query_size:
            break
    
    return sample_idx[0:query_size]

def least_confidence_query(model, device, data_loader, query_size=10):

    confidences = []
    indices = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
        
            data, _, idx = batch
            logits = model(data.to(device))
            probabilities = F.softmax(logits, dim=1)
            
            # Keep only the top class confidence for each sample
            most_probable = torch.max(probabilities, dim=1)[0]
            confidences.extend(most_probable.cpu().tolist())
            indices.extend(idx.tolist())
            
    conf = np.asarray(confidences)
    ind = np.asarray(indices)
    sorted_pool = np.argsort(conf)
    # Return the indices corresponding to the lowest `query_size` confidences
    return ind[sorted_pool][0:query_size]

def margin_query(model, device, data_loader, query_size=10):
    
    margins = []
    indices = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
        
            data, _, idx = batch
            logits = model(data.to(device))
            probabilities = F.softmax(logits, dim=1)
            
            # Select the top two class confidences for each sample
            toptwo = torch.topk(probabilities, 2, dim=1)[0]
            
            # Compute the margins = differences between the two top confidences
            differences = toptwo[:,0]-toptwo[:,1]
            margins.extend(torch.abs(differences).cpu().tolist())
            indices.extend(idx.tolist())

    margin = np.asarray(margins)
    index = np.asarray(indices)
    sorted_pool = np.argsort(margin)
    # Return the indices corresponding to the lowest `query_size` margins
    return index[sorted_pool][0:query_size]


def query_the_oracle(model, device, dataset, query_size=10, query_strategy='random', 
                     interactive=True, pool_size=0, batch_size=128, num_workers=4):
    
    unlabeled_idx = np.nonzero(dataset.unlabeled_mask)[0]
    
    # Select a pool of samples to query from
    if pool_size > 0:    
        pool_idx = random.sample(range(1, len(unlabeled_idx)), pool_size)
        pool_loader = DataLoader(
                                dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(unlabeled_idx[pool_idx]),
                                collate_fn=partial(
                                    collate_fn,
                                    src_process=dataset.sequence_encoder[dataset.src_lang],
                                    trg_process=dataset.sequence_encoder[dataset.trg_lang],
                                    pad_index=pad_index,
                                    device=device,
                                    has_trg=dataset.has_trg,
                                    is_train=dataset.split == "train",
                                ),
                                num_workers=num_workers,
                            )
        # pool_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
        #                                       sampler=SubsetRandomSampler(unlabeled_idx[pool_idx]))
    else:
        # pool_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
        #                                       sampler=SubsetRandomSampler(unlabeled_idx))
        pool_loader = DataLoader(
                                dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(unlabeled_idx),
                                collate_fn=partial(
                                    collate_fn,
                                    src_process=dataset.sequence_encoder[dataset.src_lang],
                                    trg_process=dataset.sequence_encoder[dataset.trg_lang],
                                    pad_index=pad_index,
                                    device=device,
                                    has_trg=dataset.has_trg,
                                    is_train=dataset.split == "train",
                                ),
                                num_workers=num_workers,
                            )
       
    if query_strategy == 'margin':
        sample_idx = margin_query(model, device, pool_loader, query_size)
    elif query_strategy == 'least_confidence':
        sample_idx = least_confidence_query(model, device, pool_loader, query_size)
    else:
        sample_idx = random_query(pool_loader, query_size)
    
    # Query the samples, one at a time
    for sample in sample_idx:
        
        if interactive:
            dataset.display(sample)
            print("What is the translation for this sentence?")
            new_label = int(input())
            dataset.update_label(sample, new_label)
            
        else:
            dataset.label_from_file(sample)

def train_model_ac(cfg_file: str, skip_test: bool = False) -> Any:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    :param skip_test: whether a test should be run or not after training
    :return:
        - model
    """
    # read config file
    cfg = load_config(Path(cfg_file))

    # make logger
    model_dir = make_model_dir(
        Path(cfg["training"]["model_dir"]),
        overwrite=cfg["training"].get("overwrite", False),
    )
    joeynmt_version = make_logger(model_dir, mode="train")
    if "joeynmt_version" in cfg:
        assert str(joeynmt_version) == str(cfg["joeynmt_version"]), (
            f"You are using JoeyNMT version {joeynmt_version}, "
            f'but {cfg["joeynmt_version"]} is expected in the given config.')
    # TODO: save version number in model checkpoints

    # write all entries of config to the log
    log_cfg(cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, (model_dir / "config.yaml").as_posix())

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # load the data
    src_vocab, trg_vocab, train_data, dev_data, test_data = load_data(
        data_cfg=cfg["data"])

    # store the vocabs and tokenizers
    src_vocab.to_file(model_dir / "src_vocab.txt")
    if hasattr(train_data.tokenizer[train_data.src_lang], "copy_cfg_file"):
        train_data.tokenizer[train_data.src_lang].copy_cfg_file(model_dir)
    trg_vocab.to_file(model_dir / "trg_vocab.txt")
    if hasattr(train_data.tokenizer[train_data.trg_lang], "copy_cfg_file"):
        train_data.tokenizer[train_data.trg_lang].copy_cfg_file(model_dir)

    # build an encoder-decoder model
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    
    query_the_oracle(model, torch.device("cuda"), train_data, query_size=10, query_strategy='random', 
                     interactive=True, pool_size=0, batch_size=128, num_workers=4)
    
     # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, cfg=cfg)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)
    
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    args = parser.parse_args()
    cfg_file = "/home/ubuntu/joeynmt_kriti/test/data/models/v1_enhi_25_transformer_23.04bleu/enhi_transformer_t1/config.yaml"
    ckpt = "/home/ubuntu/joeynmt_kriti/test/data/models/v1_enhi_25_transformer_23.04bleu/enhi_transformer_t1/81000.ckpt"
    train_model_ac(cfg_file=args.config)
