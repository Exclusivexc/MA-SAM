import os
import pickle
import numpy as np
import pandas as pd
from skimage.measure import block_reduce
from sklearn.cluster import KMeans
from tqdm import tqdm

""" 
    image: H, W, 1
    feature: 1, 256, 64, 64
    uncertinaty: H, W
"""


class Sample_Selector(object):
    def __init__(self, base_dir: str, select_ratio: float = 0.01):
        self.base_dir = base_dir
        self.files = os.listdir(base_dir)
        self.files = [
            file
            for file in self.files
            if "segmentation" not in file
            and "uncertainty" not in file
            and "feature" not in file
            and "csv" not in file
            and int(file[4:6]) < 50
        ]
        self.select_num = int(len(self.files) * select_ratio)

    def random_select(self, seed=2025):
        np.random.seed(seed)
        results = [
            os.path.join(self.base_dir, file)
            for file in np.random.choice(self.files, self.select_num)
        ]
        selected_samples = [
            {
                "image_pth": image_pth,
                "mask_pth": image_pth.replace(".pkl", "_segmentation.pkl"),
            }
            for image_pth in results
        ]

        # Save selected samples to a CSV file
        output_csv_file = os.path.join(
            self.base_dir, "csv_files", f"random_select_{self.select_num}_seed{seed}.csv"
        )
        selected_samples_df = pd.DataFrame(selected_samples)
        selected_samples_df.to_csv(output_csv_file, index=False)

        print(
            f"Successfully saved selected samples to {output_csv_file} with random method!"
        )

    def my_select(self):
        def average_pooling(
            uncertainty: np.ndarray, target_size: tuple = (64, 64)
        ) -> np.ndarray:
            """
            Perform average pooling to resize the uncertainty map to the target size.

            Args:
                uncertainty (np.ndarray): Input uncertainty map of shape (H, W).
                target_size (tuple): Target size (height, width), default is (64, 64).

            Returns:
                np.ndarray: Resized uncertainty map of shape (64, 64).
            """
            input_height, input_width = uncertainty.shape
            target_height, target_width = target_size

            # Calculate pooling block size
            block_size = (input_height // target_height, input_width // target_width)

            # Perform average pooling using block_reduce
            pooled_uncertainty = block_reduce(
                uncertainty, block_size=block_size, func=np.mean
            )

            return pooled_uncertainty

        def weighted_feature_fusion(
            feature: np.ndarray, pooled_uncertainty: np.ndarray
        ) -> np.ndarray:
            """
            Perform weighted fusion of feature maps using pooled uncertainty.

            Args:
                feature (np.ndarray): Input feature map of shape (1, 256, 64, 64).
                pooled_uncertainty (np.ndarray): Pooled uncertainty map of shape (64, 64).

            Returns:
                np.ndarray: Weighted feature of shape (1, 256).
            """
            # Normalize pooled_uncertainty to sum to 1
            weights = pooled_uncertainty / np.sum(pooled_uncertainty)

            # Reshape weights to match feature dimensions
            weights = weights.reshape(1, 1, 64, 64)

            # Perform weighted sum along spatial dimensions
            weighted_feature = np.sum(feature * weights, axis=(2, 3))  # Shape: (1, 256)

            return weighted_feature

        results = []  # To store weighted_feature and avg_uncertainty for each file

        for file in tqdm(self.files):
            image_file = os.path.join(self.base_dir, file)
            feature_file = image_file.replace(".pkl", "_feature.pkl")
            uncertianty_file = image_file.replace(".pkl", "_uncertainty_10.pkl")
            image, feature, uncertainty = (
                pickle.load(open(image_file, "rb")),
                pickle.load(open(feature_file, "rb")),
                pickle.load(open(uncertianty_file, "rb")),
            )

            pooled_uncertainty = (
                average_pooling(uncertainty, target_size=(64, 64)) + 0.5
            )
            avg_uncertainty = uncertainty.mean()

            weighted_feature = weighted_feature_fusion(feature, pooled_uncertainty)

            results.append(
                {
                    "image_pth": image_file,
                    "weighted_feature": weighted_feature,
                    "avg_uncertainty": avg_uncertainty,
                }
            )

        # Prepare data for clustering
        weighted_features = np.array(
            [result["weighted_feature"].flatten() for result in results]
        )  # Shape: (num_files, 256)

        # Perform KMeans clustering
        n_clusters = self.select_num
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
        clusters = kmeans.fit_predict(weighted_features)
        # Add cluster labels to results
        for i, result in enumerate(results):
            result["cluster"] = int(clusters[i])

        # Select one sample from each cluster
        selected_samples = []  # To store selected samples
        for cluster_id in range(n_clusters):
            cluster_samples = [
                result for result in results if result["cluster"] == cluster_id
            ]

            if not selected_samples:
                # For the first cluster, select the sample with avg_uncertainty closest to the median
                uncertainties = np.array(
                    [sample["avg_uncertainty"] for sample in cluster_samples]
                )
                median_uncertainty = np.median(uncertainties)
                closest_sample = cluster_samples[
                    np.argmin(np.abs(uncertainties - median_uncertainty))
                ]
            else:
                # For subsequent clusters, select the sample with avg_uncertainty farthest from already selected samples
                selected_uncertainties = np.array(
                    [sample["avg_uncertainty"] for sample in selected_samples]
                )
                distances = np.array(
                    [
                        np.min(
                            np.abs(selected_uncertainties - sample["avg_uncertainty"])
                        )
                        for sample in cluster_samples
                    ]
                )
                closest_sample = cluster_samples[np.argmax(distances)]

            selected_samples.append(
                {
                    "image_pth": closest_sample["image_pth"],
                    "mask_pth": closest_sample["image_pth"].replace(
                        ".pkl", "_segmentation.pkl"
                    ),
                    "avg_uncertainty": closest_sample["avg_uncertainty"],
                }
            )

        # Save selected samples to a CSV file
        output_csv_file = os.path.join(self.base_dir, "csv_files", f"my_select_{n_clusters}.csv")
        selected_samples_df = pd.DataFrame(selected_samples)
        selected_samples_df.to_csv(output_csv_file, index=False)

        print(
            f"Successfully saved selected samples to {output_csv_file} with my method!"
        )

if __name__ == "__main__":
    base_dir = "/media/ubuntu/maxiaochuan/MA-SAM/data/Promise12/2D_slices"
    select_ratios = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]
    for select_ratio in select_ratios:
        selector = Sample_Selector(base_dir, select_ratio=select_ratio)
        selector.random_select(seed=2023)
        selector.random_select(seed=2024)
        selector.random_select(seed=2025)
        selector.random_select(seed=2026)
        selector.random_select(seed=2027)