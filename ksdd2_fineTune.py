import os
import sys

LOG_WANDB = False

import copy
import json
from pathlib import Path

if LOG_WANDB:
    import wandb

from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, seed_everything

from torchmetrics import AveragePrecision, Metric
from anomalib.utils.metrics import AUROC, AUPRO

from datamodules import ksdd2, sensum
from datamodules.ksdd2 import KSDD2, NumSegmented
from datamodules.sensum import Sensum
from datamodules.mvtec import MVTec
from datamodules.visa import Visa

from model.supersimplenet import SuperSimpleNet

from common.visualizer import Visualizer
from common.results_writer import ResultsWriter
from common.loss import calc_loss


def train(
    model: SuperSimpleNet,
    epochs: int,
    datamodule: LightningDataModule,
    device: str,
    image_metrics: dict[str, Metric],
    pixel_metrics: dict[str, Metric],
    th: float = 0.5,
    clip_grad: bool = True,
    eval_step_size: int = 4,
):
    model.to(device)
    optimizer, scheduler = model.get_optimizers()
    bestLoss = float('inf')

    model.train()
    train_loader = datamodule.train_dataloader()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(
            total=len(train_loader),
            desc=str(epoch) + "/" + str(epochs),
            miniters=int(1),
            unit="batch",
        ) as prog_bar:
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()

                image_batch = batch["image"].to(device)

                # Fetch object mask if available
                object_mask = batch.get("object_mask", None)
                if object_mask is not None:
                    object_mask = object_mask.to(device).float()
                    object_mask = F.interpolate(
                        object_mask.unsqueeze(1),
                        size=(model.fh, model.fw),
                        mode="bilinear",
                        align_corners=False,
                    )

                # best downsampling proposed by DestSeg
                mask = batch["mask"].to(device).type(torch.float32)
                mask = F.interpolate(
                    mask.unsqueeze(1),
                    size=(model.fh, model.fw),
                    mode="bilinear",
                    align_corners=False,
                )
                mask = torch.where(
                    mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
                )

                label = batch["label"].to(device).type(torch.float32)

                anomaly_map, score, mask, label = model.forward(
                    image_batch, mask, label, object_mask
                )

                # adjusted truncated l1: mask + flipped sign (ano->pos, good->neg)

                loss = calc_loss(anomaly_map, mask, th, score, label, object_mask)

                loss.backward()

                if clip_grad:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                else:
                    norm = None

                optimizer.step()

                total_loss += loss.detach().cpu().item()

                output = {
                    "batch_loss": np.round(loss.data.cpu().detach().numpy(), 5),
                    "avg_loss": np.round(total_loss / (i + 1), 5),
                    "norm": norm,
                }

                prog_bar.set_postfix(**output)
                prog_bar.update(1)

            if (epoch + 1) % eval_step_size == 0:
                results, loss = test(
                    model=model,
                    datamodule=datamodule,
                    device=device,
                    image_metrics=image_metrics,
                    pixel_metrics=pixel_metrics,
                    th=th,
                    normalize=True,
                )
                print("\n\t[test loss : ",loss,"]\n")
                if(bestLoss > loss):
                    bestLoss = loss
                    model.save_model(
                        Path(f"./best_loss_weight")
                    )
                    print(f"\n\t[S] Epoch {epoch + 1}: New best loss {bestLoss}. Model saved.\n")
                if LOG_WANDB:
                    wandb.log({**results, **output})
            else:
                if LOG_WANDB:
                    wandb.log(output)
        scheduler.step()

    return results


@torch.no_grad()
def test(
    model: SuperSimpleNet,
    datamodule: LightningDataModule,
    device: str,
    image_metrics: dict[str, Metric],
    pixel_metrics: dict[str, Metric],
    th: float = 0.5,
    normalize: bool = True,
    image_save_path: Path = None,
    score_save_path: Path = None,
):
    model.to(device)
    model.eval()

    # for anomaly map max as image score
    seg_image_metrics = {}

    for m_name, metric in image_metrics.items():
        metric.cpu()
        metric.reset()

        seg_image_metrics[f"seg-{m_name}"] = copy.deepcopy(metric)

    for metric in pixel_metrics.values():
        metric.cpu()
        metric.reset()

    test_loader = datamodule.test_dataloader()
    results = {
        "anomaly_map": [],
        "gt_mask": [],
        "score": [],
        "seg_score": [],
        "label": [],
        "image_path": [],
        "mask_path": [],
        "loss" : [],
    }
    for batch in tqdm(test_loader, position=0, leave=True):
        image_batch = batch["image"].to(device)

        # Fetch object mask if available
        object_mask = batch.get("object_mask", None)

        if object_mask is not None:
            object_mask = object_mask.to(device).float()
            object_mask = F.interpolate(
                object_mask.unsqueeze(1),
                size=(model.fh, model.fw),
                mode="bilinear",
                align_corners=False,
            )

        anomaly_map, anomaly_score = model.forward(image_batch, object_mask=object_mask)

        # Apply object mask to anomaly map and ground truth if provided
        if object_mask is not None:
            object_mask = F.interpolate(
                object_mask, size=anomaly_map.shape[-2:], mode="bilinear", align_corners=False
            )
            object_mask = object_mask.to(anomaly_map.device)
            anomaly_map = anomaly_map * object_mask

        score = torch.sigmoid(anomaly_score)
        label = batch["label"].detach().cpu()

        loss =  calc_loss(anomaly_map, batch["mask"].unsqueeze(1), th, score, label, object_mask)
    
        anomaly_map = anomaly_map.detach().cpu()
        anomaly_score = anomaly_score.detach().cpu()

        results["anomaly_map"].append(anomaly_map.detach().cpu())
        results["gt_mask"].append(batch["mask"].detach().cpu())

        results["loss"].append(loss.detach().cpu())
        results["score"].append(score)
        results["seg_score"].append(
            anomaly_map.reshape(anomaly_map.shape[0], -1).max(dim=1).values
        )
        results["label"].append(label)

        results["image_path"].extend(batch["image_path"])
        results["mask_path"].extend(batch["mask_path"])

    results["anomaly_map"] = torch.cat(results["anomaly_map"])
    results["score"] = torch.cat(results["score"])
    results["seg_score"] = torch.cat(results["seg_score"])
    results["gt_mask"] = torch.cat(results["gt_mask"])
    results["label"] = torch.cat(results["label"])
    results["loss"] = sum(results['loss'])

    # normalize
    if normalize:
        results["score"] = (results["score"] - results["score"].min()) / (
            results["score"].max() - results["score"].min()
        )
        results["seg_score"] = (results["seg_score"] - results["seg_score"].min()) / (
            results["seg_score"].max() - results["seg_score"].min()
        )

    results_dict = {}
    for name, metric in image_metrics.items():
        metric.update(results["score"], results["label"])
        results_dict[name] = metric.to(device).compute().item()
        metric.to("cpu")

    for name, metric in seg_image_metrics.items():
        metric.update(results["seg_score"], results["label"])
        results_dict[name] = metric.to(device).compute().item()
        metric.to("cpu")

    for name, metric in pixel_metrics.items():
        try:
            # avoid nan in early stages
            am = results["anomaly_map"]
            am[am != am] = 0
            results["anomaly_map"] = am

            metric.update(results["anomaly_map"], results["gt_mask"].type(torch.float32))
            results_dict[name] = metric.to(device).compute().item()
        except RuntimeError:
            # AUPRO in some cases with early predictions crashes cuda, so just skip it in that case
            results_dict[name] = 0
        metric.to("cpu")

    for name, value in results_dict.items():
        print(f"{name}: {value} ", end="")
    print()

    if image_save_path:
        print("Visualizing")
        visualizer = Visualizer(image_save_path)
        visualizer.visualize(results)

    score_dict = {}
    if score_save_path:
        # save both segscore and score to json
        for img_path, score, seg_score, label in zip(
            results["image_path"],
            results["score"],
            results["seg_score"],
            results["label"],
        ):
            img_path = Path(img_path)

            anomaly_type = img_path.parent.name
            if anomaly_type not in score_dict:
                score_dict[anomaly_type] = {"good": {}, "bad": {}}

            # since some datasets (sensum) can have same names in bad and good
            if label == 1:
                kind = "bad"
            else:
                kind = "good"

            score_dict[anomaly_type][kind][img_path.stem] = {
                "score": score.item(),
                "seg_score": seg_score.item(),
            }

        score_save_path.mkdir(exist_ok=True, parents=True)
        with open(score_save_path / "scores.json", "w") as f:
            json.dump(score_dict, f)

    return results_dict , results['loss'].item()


def train_and_eval(model, datamodule, config, device):
    if LOG_WANDB:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.init(project=config["wandb_project"], config=config, name=config["name"])

    image_metrics = {
        "I-AUROC": AUROC(),
        "AP-det": AveragePrecision(num_classes=1),
    }
    pixel_metrics = {
        "P-AUROC": AUROC(),
        "AUPRO": AUPRO(),
        "AP-loc": AveragePrecision(num_classes=1),
    }

    train(
        model=model,
        epochs=config["epochs"],
        datamodule=datamodule,
        device=device,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        clip_grad=config["clip_grad"],
        eval_step_size=config["eval_step_size"],
    )
    if LOG_WANDB:
        wandb.finish()

    try:
        model.save_model(
            Path(config["results_save_path"])
            / config["setup_name"]
            / "checkpoints"
            / config["dataset"]
            / config["category"],
        )
    except Exception as e:
        print("Error saving checkpoint" + str(e))

    results, _ = test(
        model=model,
        datamodule=datamodule,
        device=device,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        normalize=True,
        image_save_path=Path(config["results_save_path"])
        / config["setup_name"]
        / "visual"
        / config["dataset"]
        / config["category"],
        score_save_path=Path(config["results_save_path"])
        / config["setup_name"]
        / "scores"
        / config["dataset"]
        / config["category"],
    )

    return results


def main_ksdd2(device, config):
    config = copy.deepcopy(config)
    config["dataset"] = "ksdd2"
    config["category"] = "ksdd2"
    config["name"] = f"ksdd2_{config['setup_name']}"

    results_writer = ResultsWriter(
        metrics=[
            "AP-det",
            "AP-loc",
            "P-AUROC",
            "I-AUROC",
            "AUPRO",
            "seg-AP-det",
            "seg-I-AUROC",
        ]
    )

    seed_everything(config["seed"], workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = SuperSimpleNet(image_size=ksdd2.get_default_resolution(), config=config)
    model.load_state_dict(torch.load(config['weight'], map_location=device), strict=False)
    print("weights loaded")

    datamodule = KSDD2(
        root=Path(config["datasets_folder"]) / "KolektorSDD2",
        image_size=ksdd2.get_default_resolution(),
        train_batch_size=config["batch"],
        eval_batch_size=config["batch"],
        num_workers=config["num_workers"],
        num_segmented=NumSegmented.N246,
        seed=config["seed"],
        flips=config["flips"],
    )
    datamodule.setup()

    results = train_and_eval(
        model=model, datamodule=datamodule, config=config, device=device
    )

    results_writer.add_result(
        category="ksdd2",
        last=results,
    )
    results_writer.save(
        Path(config["results_save_path"]) / config["setup_name"] / config["dataset"]
    )

def run_sup(data_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "weight":"./weights.pt",
        "wandb_project": "icpr",
        "datasets_folder": Path("./datasets"),
        "num_workers": 1,
        "setup_name": "superSimpleNet",
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise": True,
        "perlin": True,
        "no_anomaly": "empty",
        "bad": True,
        "overlap": False,
        "noise_std": 0.015,
        "perlin_thr": 0.6,
        "seed": 456654,
        "batch": 32,
        "epochs": 300,
        "flips": True,
        "seg_lr": 0.0002,
        "dec_lr": 0.0002,
        "adapt_lr": 0.0001,
        "gamma": 0.4,
        "stop_grad": False,
        "clip_grad": True,
        "eval_step_size": 4,
        "results_save_path": Path("./results"),
    }

    if data_name == "ksdd2":
        main_ksdd2(device=device, config=config)

def main():
    run_sup("ksdd2")


if __name__ == "__main__":
    main()
