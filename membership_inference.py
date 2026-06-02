"""Membership Inference Attack (Shokri et al., 2016) - reproducible script.

This script runs a full membership inference pipeline on 20 Newsgroups:
1) Load and split data into target/shadow pools.
2) Vectorize text with TF-IDF.
3) Train target model.
4) Train shadow models and build attack dataset.
5) Train attack model.
6) Evaluate privacy risk (AUC and Attack Accuracy).

Why this script exists:
- The notebook environment can be unstable in low-memory setups.
- This file is a stable, executable baseline for repeatable experiments.
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import shuffle


# -----------------------------
# Config (easy to tweak)
# -----------------------------
SEED = 42
N_SHADOWS = 2
TARGET_TRAIN_SIZE = 500
SHADOW_TRAIN_SIZE = 500
MAX_FEATURES = 1000
MAX_TOTAL_SAMPLES = 3000

np.random.seed(SEED)


def train_model(X, y):
    """Train target/shadow model with intentionally weak regularization.

    C=10.0 keeps regularization relatively low, which often increases
    overfitting and makes membership inference signals easier to observe.
    """
    model = LogisticRegression(
        C=10.0,
        max_iter=200,
        solver="lbfgs",
    )
    model.fit(X, y)
    return model


def get_probs(model, X, n_classes):
    """Return probabilities with fixed class width (n_classes).

    Some models may not observe all classes in a split; this function places
    predicted columns into a full [n_samples, n_classes] matrix.
    """
    probs = model.predict_proba(X)
    full = np.zeros((X.shape[0], n_classes))
    for i, cls in enumerate(model.classes_):
        full[:, cls] = probs[:, i]
    return full


def load_and_split_data():
    """Load dataset and produce the 4-way split used by Shokri-style attacks."""
    print("Descargando 20 Newsgroups...")
    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        random_state=SEED,
    )

    X_all = np.array(data.data)
    y_all = np.array(data.target)

    # Memory guard for constrained environments (WSL/notebook crashes).
    X_all = X_all[:MAX_TOTAL_SAMPLES]
    y_all = y_all[:MAX_TOTAL_SAMPLES]
    n_classes = len(data.target_names)

    print(f"Total documentos: {len(X_all)}")
    print(f"Clases: {n_classes}")

    # Robust shuffle for object arrays (text), safer than manual permutation
    # in low-memory environments.
    X_all, y_all = shuffle(X_all, y_all, random_state=SEED)

    # 4-way split: target members/non-members + shadow in/out pools.
    t1 = TARGET_TRAIN_SIZE
    t2 = t1 + TARGET_TRAIN_SIZE
    t3 = t2 + SHADOW_TRAIN_SIZE * N_SHADOWS
    t4 = t3 + SHADOW_TRAIN_SIZE * N_SHADOWS

    X_target_train, y_target_train = X_all[:t1], y_all[:t1]
    X_target_test, y_target_test = X_all[t1:t2], y_all[t1:t2]
    X_shadow_pool_in, y_shadow_pool_in = X_all[t2:t3], y_all[t2:t3]
    X_shadow_pool_out, y_shadow_pool_out = X_all[t3:t4], y_all[t3:t4]

    print("\nSplits:")
    print(f"  target_train:    {len(X_target_train)}")
    print(f"  target_test:     {len(X_target_test)}")
    print(f"  shadow_pool_in:  {len(X_shadow_pool_in)}")
    print(f"  shadow_pool_out: {len(X_shadow_pool_out)}")

    print("\n=== MUESTRA DATOS CRUDOS (target_train) ===")
    for i in range(min(2, len(X_target_train))):
        snippet = X_target_train[i].replace("\n", " ")[:220]
        print(f"[{i}] clase={y_target_train[i]} | texto=\"{snippet}...\"")

    return (
        X_all,
        n_classes,
        X_target_train,
        y_target_train,
        X_target_test,
        y_target_test,
        X_shadow_pool_in,
        y_shadow_pool_in,
        X_shadow_pool_out,
        y_shadow_pool_out,
    )


def vectorize_data(X_all, X_target_train, X_target_test, X_shadow_pool_in, X_shadow_pool_out):
    """Fit a shared TF-IDF space and transform all splits."""
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 1),
        min_df=2,
    )
    vectorizer.fit(X_all)

    V_target_train = vectorizer.transform(X_target_train)
    V_target_test = vectorizer.transform(X_target_test)
    V_shadow_pool_in = vectorizer.transform(X_shadow_pool_in)
    V_shadow_pool_out = vectorizer.transform(X_shadow_pool_out)

    print("\n=== MUESTRA TRANSFORMACION TF-IDF ===")
    feature_names = vectorizer.get_feature_names_out()
    row = V_target_train[0]
    print(f"Shape vector TF-IDF [0]: {row.shape} | no-cero={row.nnz}")

    if row.nnz > 0:
        idx_sorted = np.argsort(row.data)[::-1][:10]
        top_cols = row.indices[idx_sorted]
        top_vals = row.data[idx_sorted]
        print("Top terminos (termino: peso):")
        for col, val in zip(top_cols, top_vals):
            print(f"  {feature_names[col]}: {val:.4f}")

    return V_target_train, V_target_test, V_shadow_pool_in, V_shadow_pool_out


def main():
    (
        X_all,
        n_classes,
        X_target_train,
        y_target_train,
        X_target_test,
        y_target_test,
        X_shadow_pool_in,
        y_shadow_pool_in,
        X_shadow_pool_out,
        y_shadow_pool_out,
    ) = load_and_split_data()

    V_target_train, V_target_test, V_shadow_pool_in, V_shadow_pool_out = vectorize_data(
        X_all,
        X_target_train,
        X_target_test,
        X_shadow_pool_in,
        X_shadow_pool_out,
    )

    # 1) Train target model.
    print("\nEntrenando modelo target...")
    target_model = train_model(V_target_train, y_target_train)
    train_acc = accuracy_score(y_target_train, target_model.predict(V_target_train))
    test_acc = accuracy_score(y_target_test, target_model.predict(V_target_test))
    print(f"Target train accuracy: {train_acc:.4f}")
    print(f"Target test  accuracy: {test_acc:.4f}")
    print(f"Overfitting gap:       {train_acc - test_acc:.4f}")

    # 2) Train shadows and build attack dataset.
    attack_X = []
    attack_y = []
    for s in range(N_SHADOWS):
        a, b = s * SHADOW_TRAIN_SIZE, (s + 1) * SHADOW_TRAIN_SIZE
        c, d = s * SHADOW_TRAIN_SIZE, (s + 1) * SHADOW_TRAIN_SIZE

        X_in, y_in = V_shadow_pool_in[a:b], y_shadow_pool_in[a:b]
        X_out, y_out = V_shadow_pool_out[c:d], y_shadow_pool_out[c:d]

        shadow = train_model(X_in, y_in)

        attack_X.append(get_probs(shadow, X_in, n_classes))
        attack_y.extend([1] * X_in.shape[0])
        attack_X.append(get_probs(shadow, X_out, n_classes))
        attack_y.extend([0] * X_out.shape[0])

        print(f"Shadow {s + 1} listo")

    attack_X = np.vstack(attack_X)
    attack_y = np.array(attack_y)

    # 3) Train attack model.
    attack_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=SEED,
        n_jobs=1,
    )
    attack_model.fit(attack_X, attack_y)
    print("Attack model entrenado")

    # 4) Attack target model on members vs non-members.
    probs_members = get_probs(target_model, V_target_train, n_classes)
    probs_nonmembers = get_probs(target_model, V_target_test, n_classes)

    X_eval = np.vstack([probs_members, probs_nonmembers])
    y_eval = np.concatenate(
        [np.ones(len(probs_members)), np.zeros(len(probs_nonmembers))]
    )

    scores = attack_model.predict_proba(X_eval)[:, 1]
    preds = (scores >= 0.5).astype(int)

    auc = roc_auc_score(y_eval, scores)
    acc = accuracy_score(y_eval, preds)

    print(f"AUC: {auc:.4f}")
    print(f"Attack Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
