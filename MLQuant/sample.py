from MLQuant.modeling import Filter, Model

# ===================
# ===== model ======
# ===================

# 固定因子池，直接从featureParma['featureName']输入
class fixFilter(Filter):
    def filtFeature(self, data):
        return self.featureParam['featureName']

# 选取全部特征
class typeFilter(Filter):
    def filtFeature(self, data):
        numeric_features = data.columns.tolist()
        if self.featureParam.get("includeTime", False):
            return numeric_features 
        else:
            return [c for c in numeric_features if c not in ["date", "curTime"]]

# 按数据类型选取特征
class typeFilter(Filter):
    def filtFeature(self, data):
        numeric_features = data.select_dtypes(\
                include=['number']).columns.tolist()
        if self.featureParam.get("includeTime", False):
            return numeric_features 
        else:
            return [c for c in numeric_features if c not in ["date", "curTime"]]

# ===================
# ===== model ======
# ===================

import lightgbm as lgb
class lgbModel(Model):
    def train(self, Xi, Yi):
        self.model = lgb.LGBMRegressor(
            # ===== 基础参数 =====
            boosting_type=self.modelParam.get("boosting_type", 'gbdt'),          # 梯度提升算法（默认）
            objective=self.modelParam.get("objective", 'regression'),        # 回归任务目标
            metric=self.modelParam.get("metric", 'rmse'),                 # 评估指标：均方根误差

            # ===== 树结构控制 =====
            num_leaves=self.modelParam.get('num_leaves', 255),             # 单棵树最大叶子数（常用值：2^max_depth-1）
            max_depth=self.modelParam.get('max_depth', -1),                  # 树深度（-1表示无限制）
            min_child_samples=self.modelParam.get('min_child_samples', 20),   # 叶子节点最小样本数（防过拟合）

            # ===== 正则化参数 =====
            reg_alpha=self.modelParam.get('reg_alpha', 0.5),                 # L1正则化权重（别名：lambda_l1）
            reg_lambda=self.modelParam.get('reg_lambda', 8),                # L2正则化权重（别名：lambda_l2）

            # ===== 学习控制 =====
            learning_rate=self.modelParam.get('learning_rate', 0.01),            # 学习率（常用0.01-0.2）
            n_estimators=self.modelParam.get('n_estimators', 2000),              # 树的数量（迭代次数）
            min_gain_to_split=self.modelParam.get("min_gain_to_split", 0),
            subsample=self.modelParam.get('subsample', 0.8),                 # 样本采样比例（别名：bagging_fraction）
            colsample_bytree=self.modelParam.get('colsample_bytree', 0.6),          # 特征采样比例（别名：feature_fraction）

            # ===== 系统优化 =====
            n_jobs=self.modelParam.get("n_jobs", 8), device='cpu',                     # 使用所有CPU核心
            random_state=42,               # 随机种子（确保结果可复现）
            verbose=-1                     # 关闭训练日志（-1为静默模式）
        )
        Xi = Xi.astype("float32")
        Yi = Yi.astype("float32")
        self.model.fit(Xi, Yi)
        super().evaluate(Xi, Yi)
    def predict(self, Xi):
        return self.model.predict(Xi)
    
