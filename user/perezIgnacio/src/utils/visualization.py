import matplotlib.pyplot as plt
import random
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler

def show_rgb(X, idx):
    """
    Display RGB visualization of a multispectral patch.
    """
    rgb = np.dstack([
        X[idx, 2],  # R
        X[idx, 1],  # G
        X[idx, 0],  # B
    ])

    # Applying contrast enhancement for visualization
    rgb = np.clip(rgb, 0, 0.4)/ 0.4

    plt.imshow(rgb)
    plt.axis("off")


def plot_by_class(X, y, classes, samples_per_class=5):
    """
    Plot sample patches for each class.
    """
    num_classes = len(classes)
    plt.figure(figsize=(2 * samples_per_class, 2 * num_classes))

    for class_id, class_name in classes.items():
        class_indices = (y == class_id).nonzero()[0]

        if len(class_indices) == 0:
            print(f"No samples found for class: {class_name}")
            continue

        # Randomly select samples_per_class or fewer if not enough
        selected_indices = random.sample(
            class_indices.tolist(),
            min(samples_per_class, len(class_indices))
        )

        for i, idx in enumerate(selected_indices):
            plt.subplot(num_classes, samples_per_class, class_id * samples_per_class + i + 1)
            show_rgb(X, idx=idx)
            if i == 0:
                plt.title(f'{class_name}', loc='left', color='red')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """
    Plot confusion matrix as a heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_band_statistics(X, band_names):
    """
    Plot mean and standard deviation for each band.
    """
    means = []
    stds = []

    for b in range(X.shape[1]):
        band = X[:, b, :, :].reshape(-1).astype(np.float32)
        means.append(band.mean().item())
        stds.append(band.std().item())

    bands = range(1, X.shape[1] + 1)

    plt.figure(figsize=(10, 4))
    plt.errorbar(bands, means, yerr=stds, fmt='o', capsize=4)
    plt.xlabel("Band")
    plt.ylabel("Value")
    plt.title("Per-band mean + std")
    plt.xticks(bands, band_names, rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_band_correlation(X, band_names):
    X_flattened = np.transpose(X, (1, 0, 2, 3)).reshape(X.shape[1], -1)
    corr_matrix = np.corrcoef(X_flattened)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        xticklabels=band_names,
        yticklabels=band_names,
        cmap="coolwarm",
        fmt=".2f",
        annot=True
    )
    plt.title("Sentinel-2 Band Correlation")
    plt.tight_layout()
    plt.show()


def plot_class_distribution(y, class_names, title="Class Distribution"):    
    labels_count = dict(sorted(Counter(y).items()))
    crop_names = [class_names[label] for label in labels_count.keys()]
    crop_counts = list(labels_count.values())

    plt.figure(figsize=(12, 6))
    sns.barplot(x=crop_names, y=crop_counts, hue=crop_names, palette='viridis', legend=False)
    plt.title(title)
    plt.xlabel('Potential')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_pca(X, scale=True):
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X    

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print("Explained variance per component:")
    for i, v in enumerate(explained):
        print(f"PC{i+1}: {v:.4f}")

    # Plot cumulative variance
    plt.figure()
    plt.plot(cumulative, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Pixel-Level PCA")
    plt.grid(True)
    plt.show()

def plot_mutual_info(X, y):
    mi_scores = mutual_info_classif(
        X,
        y,
        discrete_features=False,
        n_neighbors=3,
        random_state=42
    )

    print("Mutual Information per band:")
    for i, score in enumerate(mi_scores):
        print(f"Band {i}: {score:.4f}")

    plt.figure()
    plt.bar(range(len(mi_scores)), mi_scores)
    plt.xlabel("Band")
    plt.ylabel("Mutual Information")
    plt.title("Feature Importance via Mutual Information")
    plt.show()