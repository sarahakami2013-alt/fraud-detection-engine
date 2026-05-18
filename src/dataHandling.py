from __future__ import annotations
from config import (
    TRAIN_DIR, IMAGE_SIZE, RANDOM_SEED, TEST_SIZE,
    CATEGORY_TO_BINARY, IMAGE_EXTENSIONS,
)

# Some images in this dataset are large screenshots; disabling PIL's
# decompression-bomb guard is safe for a known, local, trusted dataset.
Image.MAX_IMAGE_PIXELS = None


def list_images() -> list[tuple]:
    """Return a list of (path, category) for every image under train/.

    The category is simply the name of the folder the image sits in.
    """
    items = []
    for category in CATEGORY_TO_BINARY:
        folder = TRAIN_DIR / category
        if not folder.exists():
            raise FileNotFoundError(
                f"Missing folder: {folder}\n"
                "Each team member must download the Exam Cheating Dataset from "
                "Kaggle into data/ (the data/ folder is gitignored)."
            )
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, category))
    return items


def image_to_features(img: Image.Image) -> np.ndarray:
    """Turn ONE open PIL image into a flat feature vector.

    Three simple steps:
      - convert to grayscale  (drop colour; keep shape/posture information)
      - resize to IMAGE_SIZE  (every image becomes the same size)
      - flatten               (the 2-D thumbnail becomes a 1-D vector)

    For 32x32 this yields a 1024-length vector of pixel values (0-255).
    Standardisation happens later, in split_and_balance().
    """
    thumbnail = img.convert("L").resize(IMAGE_SIZE)
    return np.asarray(thumbnail, dtype=np.float32).flatten()


def build_dataset() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load every image and build the model-ready dataset.

    Returns
    -------
    X : np.ndarray, shape (n_images, n_features)
        Pixel-value features, one row per image.
    y : np.ndarray, shape (n_images,)
        Binary labels: 0 = Normal, 1 = Fraud.
    meta : pd.DataFrame
        One row per image with path, category, label, and the original
        width/height/mode - used for the EDA / Data Strategy section.
    """
    items = list_images()
    features, labels, rows = [], [], []

    for path, category in items:
        try:
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
                vector = image_to_features(img)
        except Exception:
            # A corrupted / unreadable file is skipped rather than crashing
            # the whole run. build_dataset() prints how many were skipped.
            continue

        features.append(vector)
        label = CATEGORY_TO_BINARY[category]
        labels.append(label)
        rows.append({
            "path": str(path), "category": category, "label": label,
            "width": width, "height": height, "mode": mode,
        })

    X = np.vstack(features)
    y = np.array(labels)
    meta = pd.DataFrame(rows)

    n_skipped = len(items) - len(meta)
    if n_skipped:
        print(f"  note: skipped {n_skipped} unreadable image(s)")

    return X, y, meta


def split_and_balance(X: np.ndarray, y: np.ndarray) -> dict:
    """Split, standardise, and balance - in the correct, leak-free order.

    Steps
    -----
    1. Stratified train/test split (fixed seed) - class proportions preserved.
    2. StandardScaler fitted on the TRAINING set only, applied to both sets.
    3. SMOTE applied to the TRAINING set only, to balance Normal vs Fraud.
       The test set is left untouched so it still reflects reality.

    Returns
    -------
    dict with keys:
        X_train, y_train  - scaled AND SMOTE-balanced training data
        X_test,  y_test   - scaled test data (real, untouched distribution)
        y_train_raw       - training labels BEFORE SMOTE (for before/after counts)
        scaler            - the fitted StandardScaler (reusable later)
    """
    # 1. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED,
    )

    # 2. Standardise (fit on train only -> no leakage)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. SMOTE on the training set only
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    return {
        "X_train": X_train_balanced,
        "y_train": y_train_balanced,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "y_train_raw": y_train,
        "scaler": scaler,
    }