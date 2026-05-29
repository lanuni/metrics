import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle

np.random.seed(42)
N_SHADOWS = 2
TARGET_TRAIN_SIZE = 500
SHADOW_TRAIN_SIZE = 500
MAX_FEATURES = 1000

print('Descargando 20 Newsgroups...')
data = fetch_20newsgroups(
    subset='all',
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)
X_all = np.array(data.data)
y_all = np.array(data.target)
X_all = X_all[:3000]
y_all = y_all[:3000]
n_classes = len(data.target_names)

print(f'Total documentos: {len(X_all)}')
print(f'Clases: {n_classes}')

# Shuffle
# idx = np.random.permutation(len(X_all))
# X_all, y_all = X_all[idx], y_all[idx]

X_all, y_all = shuffle(X_all, y_all, random_state=42)
print("hola")

# Split 4-way
t1 = TARGET_TRAIN_SIZE
t2 = t1 + TARGET_TRAIN_SIZE
t3 = t2 + SHADOW_TRAIN_SIZE * N_SHADOWS
t4 = t3 + SHADOW_TRAIN_SIZE * N_SHADOWS

X_target_train, y_target_train = X_all[:t1], y_all[:t1]
X_target_test,  y_target_test  = X_all[t1:t2], y_all[t1:t2]
X_shadow_pool_in,  y_shadow_pool_in  = X_all[t2:t3], y_all[t2:t3]
X_shadow_pool_out, y_shadow_pool_out = X_all[t3:t4], y_all[t3:t4]
print(f'\nSplits:')
print(f'  target_train:    {len(X_target_train)}')
print(f'  target_test:     {len(X_target_test)}')
print(f'  shadow_pool_in:  {len(X_shadow_pool_in)}')
print(f'  shadow_pool_out: {len(X_shadow_pool_out)}')

# Vectorización
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    stop_words='english',
    ngram_range=(1, 1),
    min_df=2
)
vectorizer.fit(X_all)

V_target_train     = vectorizer.transform(X_target_train)
V_target_test      = vectorizer.transform(X_target_test)
V_shadow_pool_in   = vectorizer.transform(X_shadow_pool_in)
V_shadow_pool_out  = vectorizer.transform(X_shadow_pool_out)

def train_model(X, y):
    model = LogisticRegression(
        C=10.0,
        max_iter=200,
        solver='lbfgs',
        n_jobs=1
    )
    model.fit(X, y)
    return model

def get_probs(model, X, n_classes):
    probs = model.predict_proba(X)
    full = np.zeros((X.shape[0], n_classes))
    for i, c in enumerate(model.classes_):
        full[:, c] = probs[:, i]
    return full

print('Entrenando modelo target...')
target_model = train_model(V_target_train, y_target_train)
train_acc = accuracy_score(y_target_train, target_model.predict(V_target_train))
test_acc  = accuracy_score(y_target_test,  target_model.predict(V_target_test))
print(f'Target train accuracy: {train_acc:.4f}')
print(f'Target test  accuracy: {test_acc:.4f}')
print(f'Overfitting gap:       {train_acc - test_acc:.4f}')

# Shadows
attack_X = []
attack_y = []
for s in range(N_SHADOWS):
    a, b = s * SHADOW_TRAIN_SIZE, (s + 1) * SHADOW_TRAIN_SIZE
    c, d = s * SHADOW_TRAIN_SIZE, (s + 1) * SHADOW_TRAIN_SIZE
    X_in,  y_in  = V_shadow_pool_in[a:b],  y_shadow_pool_in[a:b]
    X_out, y_out = V_shadow_pool_out[c:d], y_shadow_pool_out[c:d]
    shadow = train_model(X_in, y_in)
    attack_X.append(get_probs(shadow, X_in, n_classes)); attack_y.extend([1]*X_in.shape[0])
    attack_X.append(get_probs(shadow, X_out, n_classes)); attack_y.extend([0]*X_out.shape[0])
    print(f'Shadow {s+1} listo')

attack_X = np.vstack(attack_X)
attack_y = np.array(attack_y)

attack_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=1)
attack_model.fit(attack_X, attack_y)
print('Attack model entrenado')

probs_members     = get_probs(target_model, V_target_train, n_classes)
probs_nonmembers  = get_probs(target_model, V_target_test,  n_classes)
X_eval = np.vstack([probs_members, probs_nonmembers])
y_eval = np.concatenate([np.ones(len(probs_members)), np.zeros(len(probs_nonmembers))])
scores = attack_model.predict_proba(X_eval)[:, 1]
preds  = (scores >= 0.5).astype(int)

auc = roc_auc_score(y_eval, scores)
acc = accuracy_score(y_eval, preds)
print(f'AUC: {auc:.4f}')
print(f'Attack Accuracy: {acc:.4f}')