class Node:
    def __init__(self,depth=0,max_depth=6,min_child_weight=1,reg_lambda=1,feature_types=None):
        self.left=None
        self.right=None
        self.split_feature=None
        self.split_threshold=None
        self.output=None
        self.feature_types=None
        self.depth=depth
        self.max_depth=max_depth
        self.min_child_weight=min_child_weight
        self.reg_lambda=reg_lambda

    def score(self,g,h):
        return (np.sum(g) ** 2) / (np.sum(h) + self.reg_lambda)


    def calc_weights(self,g,h):
        return -np.sum(g) / (np.sum(h) + self.reg_lambda)
              

    def calc_gain(self,g_left,h_left,g_right,h_right):
        return self.score(g_left,h_left) + self.score(g_right,h_right) - self.score(np.concatenate([g_left,g_right]),np.concatenate([h_left,h_right]))
        
        
    def fit(self,X,gradients,hessians,feature_types):
        m,n=X.shape
        if self.depth>self.max_depth or len(X)==0:
            self.output=self.calc_weights(gradients,hessians)
            return
        else:
            best_gain=-1
            best_feat = None
            best_thresh = None
            best_left_idx = None
            best_right_idx = None

            for feature_idx,feat_type in zip(range(n),feature_types):
                feature_data=X[:,feature_idx]
                if feat_type != 'object':
                    thresholds=np.unique(feature_data)
                    for thr in thresholds:
                        left_indices=np.where(feature_data <= thr)[0]
                        right_indices=np.where(feature_data > thr) [0]
                        if np.sum(hessians[left_indices])<self.min_child_weight or np.sum(hessians[right_indices])<self.min_child_weight:
                            continue
 
                        gain=self.calc_gain(gradients[left_indices],hessians[left_indices],gradients[right_indices],hessians[right_indices])
                        if gain > best_gain:
                            best_gain=gain
                            best_feat=feature_idx
                            best_thresh=thr
                            best_left_idx=left_indices
                            best_right_idx=right_indices

                else:
                    thresholds=np.unique(feature_data)
                    for thr in thresholds:
                        left_indices=np.where(feature_data == thr)[0]
                        right_indices=np.where(feature_data != thr) [0]
                        if np.sum(hessians[left_indices])<self.min_child_weight or np.sum(hessians[right_indices])<self.min_child_weight:
                            continue
 
                        gain=self.calc_gain(gradients[left_indices],hessians[left_indices],gradients[right_indices],hessians[right_indices])
                        # print(gain)
                        if gain > best_gain:
                            best_gain=gain
                            best_feat=feature_idx
                            best_thresh=thr
                            best_left_idx=left_indices
                            best_right_idx=right_indices

            if best_gain > 0:
                self.split_feature=best_feat
                self.split_threshold=best_thresh

                self.left=Node(depth=self.depth+1,max_depth=self.max_depth,min_child_weight=self.min_child_weight,reg_lambda=self.reg_lambda)
                self.left.fit(X[best_left_idx],gradients[best_left_idx],hessians[best_left_idx],feature_types)

                self.right=Node(depth=self.depth+1,max_depth=self.max_depth,min_child_weight=self.min_child_weight,reg_lambda=self.reg_lambda)
                self.right.fit(X[best_right_idx],gradients[best_right_idx],hessians[best_right_idx],feature_types)

            else:
                self.output=self.calc_weights(gradients,hessians)
                
 

    def predict_row(self, x, feature_types):
        if self.output is not None:
            return self.output
        if feature_types[self.split_feature] != 'object':
            # print("numerical feature")
            if x[self.split_feature] <= self.split_threshold:
                return self.left.predict_row(x,feature_types)
            else:
                return self.right.predict_row(x,feature_types)

        else:
            # print("categorical feature")
            if x[self.split_feature] == self.split_threshold:
                return self.left.predict_row(x,feature_types)
            else:
                return self.right.predict_row(x,feature_types)
            
    def predict(self, X):
        feature_types=X.dtypes
        xtest=X.to_numpy()
        return np.array([self.predict_row(row,feature_types) for row in xtest])
  



class XGBoostRegressor():
    def __init__(self,n_estimators=100,max_depth=6,min_child_weight=1,reg_lambda=1,learning_rate=0.01):
        self.trees=[]
        self.base_prediction=None
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.min_child_weight=min_child_weight
        self.reg_lambda=reg_lambda
        self.learning_rate=learning_rate

    def initial_prediction(self,y):
        return np.mean(y)
        
    def calc_gradients(self,preds,labels):
        return 2*(preds - labels)

    def calc_hessians(self,preds,labels):
        return np.full_like(labels,2)

    def build_tree(self,X,gradients,hessians,feature_types):
        root=Node(depth=0,max_depth=self.max_depth,min_child_weight=self.min_child_weight,reg_lambda=self.reg_lambda)
        root.fit(X,gradients,hessians,feature_types)
        return root
        
        
    def fit(self,X,y):
        Xt=X.to_numpy()
        yt=y.to_numpy()

        feature_types=X.dtypes

        self.base_prediction=self.initial_prediction(yt)
        y_preds=np.full(len(yt),self.base_prediction)

        for _ in range(self.n_estimators):

            grad=self.calc_gradients(y_preds,yt)
            hess=self.calc_hessians(y_preds,yt)
            tree=self.build_tree(Xt,grad,hess,feature_types)    
            self.trees.append(tree)
            y_preds+=self.learning_rate * tree.predict(X)
            
        
    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
        
    
        
