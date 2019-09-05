import argparse
import logging
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
from data.loader import data_loader
from models import TrajectoryGenerator
from utils import (
    displacement_error,
    final_displacement_error,
    get_dset_path,
    int_tuple,
    l2_loss,
    relative_to_abs,
)

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")

parser.add_argument("--dataset_name", default="zara2", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)

parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_epochs", default=400, type=int)

parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")

parser.add_argument(
    "--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size"
)
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma"
)
parser.add_argument(
    "--hidden-units",
    type=str,
    default="16",
    help="Hidden units in each hidden layer, splitted with comma",
)
parser.add_argument(
    "--graph_network_out_dims",
    type=int,
    default=32,
    help="dims of every node after through GAT module",
)
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)


parser.add_argument(
    "--lr",
    default=1e-3,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)

parser.add_argument("--best_k", default=20, type=int)
parser.add_argument("--print_every", default=10, type=int)
parser.add_argument("--use_gpu", default=1, type=int)
parser.add_argument("--gpu_num", default="0", type=str)

parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)


best_ade = 100


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, "train")
    val_path = get_dset_path(args.dataset_name, "test")

    logging.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logging.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    writer = SummaryWriter()

    n_units = (
        [args.traj_lstm_hidden_size]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.graph_lstm_hidden_size]
    )
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    model = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        traj_lstm_input_size=args.traj_lstm_input_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=n_heads,
        graph_network_out_dims=args.graph_network_out_dims,
        dropout=args.dropout,
        alpha=args.alpha,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
    )
    model.cuda()
    optimizer = optim.Adam(
        [
            {"params": model.traj_lstm_model.parameters(), "lr": 1e-2},
            {"params": model.traj_hidden2pos.parameters()},
            {"params": model.gatencoder.parameters(), "lr": 3e-2},
            {"params": model.graph_lstm_model.parameters(), "lr": 1e-2},
            {"params": model.traj_gat_hidden2pos.parameters()},
            {"params": model.pred_lstm_model.parameters()},
            {"params": model.pred_hidden2pos.parameters()},
        ],
        lr=args.lr,
    )
    global best_ade
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("Restoring from checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    training_step = 1
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        if epoch < 150:
            training_step = 1
        elif epoch < 250:
            training_step = 2
        else:
            if epoch == 250:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 5e-3
            training_step = 3
        train(args, model, train_loader, optimizer, epoch, training_step, writer)
        if training_step == 3:
            ade = validate(args, model, val_loader, epoch, writer)
            is_best = ade < best_ade
            best_ade = min(ade, best_ade)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_ade": best_ade,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                f"./checkpoint/checkpoint{epoch}.pth.tar",
            )
    writer.close()


def train(args, model, train_loader, optimizer, epoch, training_step, writer):
    losses = utils.AverageMeter("Loss", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = [tensor.cuda() for tensor in batch]
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            non_linear_ped,
            loss_mask,
            seq_start_end,
        ) = batch
        optimizer.zero_grad()
        loss = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = []
        loss_mask = loss_mask[:, args.obs_len :]

        if training_step == 1 or training_step == 2:
            model_input = obs_traj_rel
            pred_traj_fake_rel = model(
                model_input, obs_traj, seq_start_end, 1, training_step
            )
            l2_loss_rel.append(
                l2_loss(pred_traj_fake_rel, model_input, loss_mask, mode="raw")
            )
        else:
            model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
            for _ in range(args.best_k):
                pred_traj_fake_rel = model(model_input, obs_traj, seq_start_end, 0)
                l2_loss_rel.append(
                    l2_loss(
                        pred_traj_fake_rel,
                        model_input[-args.pred_len :],
                        loss_mask,
                        mode="raw",
                    )
                )

        l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
            _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
            _l2_loss_rel = torch.min(_l2_loss_rel) / (
                (pred_traj_fake_rel.shape[0]) * (end - start)
            )
            l2_loss_sum_rel += _l2_loss_rel

        loss += l2_loss_sum_rel
        losses.update(loss.item(), obs_traj.shape[1])
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_every == 0:
            progress.display(batch_idx)
    writer.add_scalar("train_loss", losses.avg, epoch)


def validate(args, model, val_loader, epoch, writer):
    ade = utils.AverageMeter("ADE", ":.6f")
    fde = utils.AverageMeter("FDE", ":.6f")
    progress = utils.ProgressMeter(len(val_loader), [ade, fde], prefix="Test: ")

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch
            loss_mask = loss_mask[:, args.obs_len :]
            pred_traj_fake_rel = model(obs_traj_rel, obs_traj, seq_start_end)

            pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len :]
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel_predpart, obs_traj[-1])
            ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
            ade_ = ade_ / (obs_traj.shape[1] * args.pred_len)

            fde_ = fde_ / (obs_traj.shape[1])
            ade.update(ade_, obs_traj.shape[1])
            fde.update(fde_, obs_traj.shape[1])

            if i % args.print_every == 0:
                progress.display(i)

        logging.info(
            " * ADE  {ade.avg:.3f} FDE  {fde.avg:.3f}".format(ade=ade, fde=fde)
        )
        writer.add_scalar("val_ade", ade.avg, epoch)
    return ade.avg


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    return ade, fde


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, filename)
        logging.info("-------------- lower ade ----------------")
        shutil.copyfile(filename, "model_best.pth.tar")


if __name__ == "__main__":
    args = parser.parse_args()
    utils.set_logger(os.path.join(args.log_dir, "train.log"))
    checkpoint_dir = "./checkpoint"
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)
    main(args)
