from __future__ import annotations
from config import (
    TRAIN_DIR, IMAGE_SIZE, RANDOM_SEED, TEST_SIZE,
    CATEGORY_TO_BINARY, IMAGE_EXTENSIONS,
)


Image.MAX_IMAGE_PIXELS = None


def list_images() -> list[tuple]:

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

    thumbnail = img.convert("L").resize(IMAGE_SIZE)
    return np.asarray(thumbnail, dtype=np.float32).flatten()


def build_dataset() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    items = list_images()
    features, labels, rows = [], [], []

    for path, category in items:
        try:
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
                vector = image_to_features(img)
        except Exception:

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