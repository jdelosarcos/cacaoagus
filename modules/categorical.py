"""Categorical encoding and k-modes; fit only inside training partitions."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import pairwise_distances, silhouette_score
class KModesSimple:
    """Batch k-modes: Hamming cost, modal centers, deterministic tie-breaking."""
    def __init__(self,n_clusters=3,n_init=5,max_iter=40,random_state=42):
        self.n_clusters=n_clusters; self.n_init=n_init
        self.max_iter=max_iter; self.random_state=random_state
    def fit(self,X):
        X=np.asarray(X,dtype=int)
        rng=np.random.default_rng(self.random_state)
        best_cost=np.inf
        for start in range(self.n_init):
            # Farthest-probability initialization, based on categorical mismatch.
            centers=[X[rng.integers(len(X))].copy()]
            while len(centers)<self.n_clusters:
                d=(X[:,None,:]!=np.asarray(centers)[None,:,:]).mean(axis=2).min(axis=1)
                index=rng.choice(len(X),p=d/d.sum()) if d.sum()>0 else rng.integers(len(X))
                centers.append(X[index].copy())
            modes=np.asarray(centers)
            for iteration in range(self.max_iter):
                dist=(X[:,None,:]!=modes[None,:,:]).mean(axis=2)
                labels=dist.argmin(axis=1)
                updated=modes.copy()
                for k in range(self.n_clusters):
                    members=X[labels==k]
                    if len(members):
                        updated[k]=[np.bincount(members[:,j]).argmax() for j in range(X.shape[1])]
                    else:
                        updated[k]=X[np.argmax(dist.min(axis=1))]
                if np.array_equal(modes,updated): break
                modes=updated
            dist=(X[:,None,:]!=modes[None,:,:]).mean(axis=2)
            labels=dist.argmin(axis=1)
            cost=float(dist.min(axis=1).sum())
            if cost<best_cost:
                best_cost=cost; self.cluster_modes_=modes.copy(); self.labels_=labels.copy()
        self.cost_=best_cost
        return self
    def predict(self,X):
        return (np.asarray(X)[:,None,:]!=self.cluster_modes_[None,:,:]).mean(axis=2).argmin(axis=1)

class CategoricalDesign(BaseEstimator,TransformerMixin):
    """One-hot predictors, optionally with training-only selected cluster dummies."""
    def __init__(self,with_cluster=False,cluster_only=False,random_state=42):
        self.with_cluster=with_cluster; self.cluster_only=cluster_only; self.random_state=random_state
    def fit(self,X,y=None):
        X=pd.DataFrame(X).astype(str)
        self.columns_=list(X.columns)
        self.ohe_=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
        self.ohe_.fit(X)
        if self.with_cluster or self.cluster_only:
            self.ordinal_=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
            Z=self.ordinal_.fit_transform(X).astype(int)
            # Bound the quadratic silhouette calculation for larger uploads.
            # All training records still fit the modes; no test row is sampled.
            probe=(np.arange(len(Z)) if len(Z)<=1000 else
                   np.sort(np.random.default_rng(self.random_state).choice(len(Z),1000,replace=False)))
            D=pairwise_distances(Z[probe],metric='hamming')
            candidates=[]
            for k in range(2,7):
                km=KModesSimple(k,random_state=self.random_state).fit(Z)
                counts=np.bincount(km.labels_,minlength=k)
                probe_labels=km.labels_[probe]
                sil=silhouette_score(D,probe_labels,metric='precomputed') if len(np.unique(probe_labels))>1 else -1
                candidates.append((sil,k,km,counts.min()))
            valid=[t for t in candidates if t[3]>=5]
            best=max(valid or candidates,key=lambda t:(t[0],-t[1]))
            self.silhouette_,self.best_k_,self.km_,_=best
            self.cluster_scores_=pd.DataFrame([{'k':k,'silhouette_hamming':s,'minimum_train_cluster_size':n,
                                                'eligible':n>=5,'training_cost':m.cost_}
                                               for s,k,m,n in candidates])
        return self
    def cluster_labels(self,X):
        return self.km_.predict(self.ordinal_.transform(pd.DataFrame(X).astype(str)).astype(int))
    def transform(self,X):
        X=pd.DataFrame(X).astype(str)
        Z=self.ohe_.transform(X)
        if self.with_cluster or self.cluster_only:
            C=np.eye(self.best_k_)[self.cluster_labels(X)]
            return C if self.cluster_only else np.column_stack([Z,C])
        return Z
    def get_feature_names_out(self,input_features=None):
        base=list(self.ohe_.get_feature_names_out(self.columns_))
        clusters=[f'cluster_{k}' for k in range(self.best_k_)] if (self.with_cluster or self.cluster_only) else []
        return np.asarray(clusters if self.cluster_only else base+clusters,dtype=object)
