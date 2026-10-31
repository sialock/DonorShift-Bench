"""Pre-registered, reproducible baselines; no test-donor tuning."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import sparse

AVAILABLE = ('logreg', 'random_forest', 'mlp', 'transformer')


def make_sklearn_model(name: str, seed=42):
    if name == 'logreg':
        return make_pipeline(StandardScaler(with_mean=False),
                             LogisticRegression(max_iter=600, C=1.0, random_state=seed))
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=80, max_features='sqrt',
                                      class_weight='balanced_subsample', n_jobs=1,
                                      random_state=seed)
    if name == 'mlp':
        return make_pipeline(StandardScaler(with_mean=False),
                             MLPClassifier(hidden_layer_sizes=(48, 24), alpha=0.01,
                                           max_iter=180, random_state=seed))
    raise ValueError(f'Not an sklearn baseline: {name}')


def predict_model(name, X_train, y_train, X_test, seed=42, transformer_epochs=10):
    if name == 'transformer':
        return _transformer_predict(X_train, y_train, X_test, seed, transformer_epochs)
    model = make_sklearn_model(name, seed)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probs = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    return pred, np.max(probs, axis=1) if probs is not None else np.full(len(pred), np.nan)


def _transformer_predict(X_train, y_train, X_test, seed, epochs):
    """Optional compact attention baseline; fixed bins and epochs, no test tuning.

    Uses a CLS token and a small attention encoder, not a pretrained scRNA model.
    Transformer is intended for modest sample datasets; computationally expensive
    for many genes/donors. Quantization uses a constant scale, no fitted test data.
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Transformer requires pip install -e '.[transformer]'") from exc

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    classes = sorted(set(y_train))
    y_idx = np.array([classes.index(v) for v in y_train], dtype=np.int64)
    n_genes = X_train.shape[1]
    n_bins = 32

    def quantize(X):
        if sparse.issparse(X):
            X = X.toarray()
        # Fixed scale from library-size cap, independent of donor distributions.
        return torch.tensor(np.clip(np.floor(np.asarray(X) / np.log1p(10000) * (n_bins-1)),
                                    0, n_bins-1).astype(np.int64))

    xtr, xte = quantize(X_train), quantize(X_test)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            d = 32
            self.expr = nn.Embedding(n_bins, d)
            self.gene = nn.Embedding(n_genes, d)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=64,
                                               dropout=0.0, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
            self.head = nn.Linear(d, len(classes))

        def forward(self, z):
            gene_ix = torch.arange(n_genes, device=z.device)
            tokens = self.expr(z) + self.gene(gene_ix)[None, :, :]
            tokens = torch.cat([self.cls.expand(len(z), -1, -1), tokens], dim=1)
            return self.head(self.encoder(tokens)[:, 0])

    net = Encoder()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    ys = torch.tensor(y_idx, dtype=torch.long)
    net.train()
    for epoch in range(epochs):
        order = torch.randperm(len(xtr))
        for batch in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(xtr[batch]), ys[batch])
            loss.backward()
            optimizer.step()
    net.eval()
    outputs = []
    with torch.no_grad():
        for batch in xte.split(64):
            outputs.append(net(batch).softmax(-1).cpu().numpy())
    probs = np.concatenate(outputs, axis=0)
    return np.asarray(classes)[probs.argmax(axis=1)], probs.max(axis=1)