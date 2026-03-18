"""
Evaluate a trained model on benchmark datasets or run image-list inference.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import yaml
from tqdm import tqdm

from abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from utils import get_test_metrics

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent

DEFAULT_DETECTOR_PATH = SCRIPT_DIR / "config" / "detector" / "clip.yaml"
DEFAULT_TEST_CONFIG_PATH = SCRIPT_DIR / "config" / "test_config.yaml"
DEFAULT_WEIGHTS_PATH = SCRIPT_DIR / "df40_weights" / "train_on_fs" / "clip.pth"
DEFAULT_IMAGE_LIST_PATH = "../val_images.txt"
DEFAULT_OUTPUT_FILE = "submission_clip_large_df40.txt"

parser = argparse.ArgumentParser(description="Evaluate DeepfakeBench models.")
parser.add_argument(
    "--detector_path",
    type=str,
    default=str(DEFAULT_DETECTOR_PATH),
    help="Path to detector YAML file.",
)
parser.add_argument(
    "--test_dataset",
    nargs="+",
    default=None,
    help="Dataset names for JSON-based benchmark testing mode.",
)
parser.add_argument(
    "--weights_path",
    type=str,
    default=str(DEFAULT_WEIGHTS_PATH),
    help="Checkpoint path (README default for CLIP: training/df40_weights/train_on_fs/clip.pth).",
)
parser.add_argument(
    "--image_list_path",
    type=str,
    default=DEFAULT_IMAGE_LIST_PATH,
    help="Txt path with one image path per line. Set empty string to use --test_dataset mode.",
)
parser.add_argument(
    "--output_file",
    type=str,
    default=DEFAULT_OUTPUT_FILE,
    help="Output file for image-list inference probabilities.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=None,
    help="Override test batch size.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=None,
    help="Override dataloader workers.",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
on_2060 = torch.cuda.is_available() and "2060" in torch.cuda.get_device_name(0)


def resolve_existing_path(path_str):
    path = Path(path_str).expanduser()
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"Path does not exist: {path}")

    search_roots = [Path.cwd(), SCRIPT_DIR, PROJECT_DIR, WORKSPACE_DIR]
    checked = []
    for root in search_roots:
        candidate = (root / path).resolve()
        checked.append(str(candidate))
        if candidate.exists():
            return candidate

    checked_text = "\n".join(checked)
    raise FileNotFoundError(f"Could not resolve {path_str}. Checked:\n{checked_text}")


def init_seed(config):
    if config["manualSeed"] is None:
        config["manualSeed"] = random.randint(1, 10000)
    random.seed(config["manualSeed"])
    torch.manual_seed(config["manualSeed"])
    if config["cuda"]:
        torch.cuda.manual_seed_all(config["manualSeed"])


class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, resolution, mean, std):
        self.image_paths = image_paths
        self.resolution = int(resolution)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (self.resolution, self.resolution),
            interpolation=cv2.INTER_CUBIC,
        )
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = (image - self.mean) / self.std
        return image, 0, image_path

    @staticmethod
    def collate_fn(batch):
        images, labels, image_paths = zip(*batch)
        data_dict = {
            "image": torch.stack(images, dim=0),
            "label": torch.LongTensor(labels),
            "mask": None,
            "landmark": None,
            "image_paths": list(image_paths),
        }
        return data_dict


def read_image_paths_from_txt(image_list_path):
    txt_path = resolve_existing_path(image_list_path)
    base_dir = txt_path.parent
    image_paths = []
    missing_paths = []

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_token = line.split()[0]
            image_path = Path(image_token).expanduser()
            if not image_path.is_absolute():
                image_path = (base_dir / image_path).resolve()
            if image_path.exists():
                image_paths.append(str(image_path))
            else:
                missing_paths.append(str(image_path))

    if len(image_paths) == 0:
        raise ValueError(f"No valid images found in: {txt_path}")
    if len(missing_paths) > 0:
        preview = "\n".join(missing_paths[:10])
        raise FileNotFoundError(
            f"Found {len(missing_paths)} missing image paths in {txt_path}.\n"
            f"First missing paths:\n{preview}"
        )

    return image_paths


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # Update config per dataset to avoid mutating shared config
        dataset_config = config.copy()
        dataset_config["test_dataset"] = test_name
        test_set = DeepfakeAbstractBaseDataset(
            config=dataset_config,
            mode="test",
        )
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config["test_batchSize"],
            shuffle=False,
            num_workers=int(config["workers"]),
            collate_fn=test_set.collate_fn,
            drop_last=False,
        )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config["test_dataset"]:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def prepare_image_list_loader(config, image_list_path):
    image_paths = read_image_paths_from_txt(image_list_path)
    test_set = ImageListDataset(
        image_paths=image_paths,
        resolution=config["resolution"],
        mean=config["mean"],
        std=config["std"],
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config["test_batchSize"],
        shuffle=False,
        num_workers=int(config["workers"]),
        collate_fn=test_set.collate_fn,
        drop_last=False,
    )
    return test_data_loader


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for _, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = (
            data_dict["image"],
            data_dict["label"],
            data_dict["mask"],
            data_dict["landmark"],
        )
        label = torch.where(data_dict["label"] != 0, 1, 0)
        # move data to GPU
        data_dict["image"], data_dict["label"] = data.to(device), label.to(device)
        if mask is not None:
            data_dict["mask"] = mask.to(device)
        if landmark is not None:
            data_dict["landmark"] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict["label"].cpu().detach().numpy())
        prediction_lists += list(predictions["prob"].cpu().detach().numpy())
        feature_lists += list(predictions["feat"].cpu().detach().numpy())

    return np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)


def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    metrics_all_datasets = {}
    for key, loader in test_data_loaders.items():
        data_dict = loader.dataset.data_dict
        predictions_nps, label_nps, _ = test_one_dataset(model, loader)

        metric_one_dataset = get_test_metrics(
            y_pred=predictions_nps,
            y_true=label_nps,
            img_names=data_dict["image"],
        )
        metrics_all_datasets[key] = metric_one_dataset

        tqdm.write(f"dataset: {key}")
        for metric_name, metric_value in metric_one_dataset.items():
            tqdm.write(f"{metric_name}: {metric_value}")

    return metrics_all_datasets


def infer_image_list(model, data_loader):
    model.eval()
    all_probs = []
    all_image_paths = []

    for data_dict in tqdm(data_loader, total=len(data_loader)):
        data_dict["image"] = data_dict["image"].to(device)
        data_dict["label"] = data_dict["label"].to(device)

        predictions = inference(model, data_dict)
        if "prob" in predictions:
            probs = predictions["prob"]
        elif "cls" in predictions:
            probs = torch.softmax(predictions["cls"], dim=1)[:, 1]
        else:
            raise ValueError("Model output should contain either 'prob' or 'cls'.")

        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_image_paths.extend(data_dict["image_paths"])

    return all_image_paths, np.asarray(all_probs, dtype=np.float32)


def save_inference_scores(probabilities, output_file):
    output_path = Path(output_file).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for prob in probabilities:
            f.write(f"{float(prob):.6f}\n")
    return output_path


def load_weights(model, weights_path):
    if not weights_path:
        raise ValueError("weights_path is empty.")

    resolved_weights_path = resolve_existing_path(weights_path)
    ckpt = torch.load(str(resolved_weights_path), map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    new_weights = {}
    for key, value in ckpt.items():
        new_key = key.replace("module.", "")
        if "base_model." in new_key:
            new_key = new_key.replace("base_model.", "backbone.")
        if "classifier." in new_key:
            new_key = new_key.replace("classifier.", "head.")
        new_weights[new_key] = value

    model.load_state_dict(new_weights, strict=True)
    print(f"===> Load checkpoint done: {resolved_weights_path}")


@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    detector_path = resolve_existing_path(args.detector_path)
    test_config_path = resolve_existing_path(str(DEFAULT_TEST_CONFIG_PATH))

    with open(detector_path, "r") as f:
        config = yaml.safe_load(f)
    with open(test_config_path, "r") as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    config["cuda"] = torch.cuda.is_available()

    if on_2060:
        config["lmdb_dir"] = r"./transform_2_lmdb"
        config["train_batchSize"] = 10
        config["workers"] = 0
    else:
        # Local fallback for non-2060 machines.
        config["workers"] = 8
        config["lmdb_dir"] = r"./data/LMDBs"

    # CLI overrides
    if args.test_dataset:
        config["test_dataset"] = args.test_dataset
    if args.batch_size is not None:
        config["test_batchSize"] = args.batch_size
    if args.num_workers is not None:
        config["workers"] = args.num_workers

    config["weights_path"] = args.weights_path

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config["cudnn"]:
        cudnn.benchmark = True

    # prepare the model (detector)
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)
    load_weights(model, args.weights_path)

    use_image_list_mode = bool(args.image_list_path and args.image_list_path.strip())

    if use_image_list_mode:
        test_data_loader = prepare_image_list_loader(config, args.image_list_path)
        _, all_probs = infer_image_list(model, test_data_loader)
        output_path = save_inference_scores(all_probs, args.output_file)
        print(f"===> Inference done on {len(all_probs)} images!")
        print(f"===> Saved probability scores to: {output_path}")
        print(
            f"===> Probability range: "
            f"[{float(all_probs.min()):.6f}, {float(all_probs.max()):.6f}]"
        )
    else:
        # Keep original benchmark behavior (dataset-json based evaluation with metrics)
        test_data_loaders = prepare_testing_data(config)
        _ = test_epoch(model, test_data_loaders)
        print("===> Test Done!")


if __name__ == "__main__":
    main()
