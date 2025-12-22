import pandas as pd
from MLQuant.modeling import Filter, Model

# ===================
# ===== Filter ======
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
    """
    根据 featureParam 中的 'type' 字段筛选特征列。
    - type 可为: 
        - 字符串: 'bool', 'number', 'float', 'string', 'all'
        - 列表: 如 ['bool', 'string']，或 ['all']（等效于 'all'）
    - 若 includeNormal=False（默认），则从结果中移除名为 'symbol', 'date', 'curTime', 'legalData'的列
    - 若 type 包含 'all'，则返回所有列（再根据 includeNormal 决定是否剔除常规列）
    """

    def filtFeature(self, data):
        type_spec = self.featureParam.get("type", "number")
        include_normal = self.featureParam.get("includeNormal", False)

        # 统一转为列表
        if isinstance(type_spec, str):
            type_list = [type_spec]
        elif isinstance(type_spec, (list, tuple)):
            type_list = list(type_spec)
        else:
            raise ValueError(f"'type' must be a string or a list of strings, got {type(type_spec)}")

        # 特殊处理 "all"
        if "all" in type_list:
            selected_cols = data.columns.tolist()
        else:
            selected_cols_set = set()
            for t in type_list:
                if t == "bool":
                    cols = data.select_dtypes(include=['bool']).columns
                elif t == "number":
                    cols = data.select_dtypes(include=['number']).columns
                elif t == "float":
                    cols = data.select_dtypes(include=['floating']).columns
                elif t == "string":
                    cols = data.select_dtypes(include=['object', 'string']).columns
                else:
                    raise ValueError(f"Unsupported type: '{t}'. "
                                     f"Supported types: 'bool', 'number', 'float', 'string', 'all'.")
                selected_cols_set.update(cols)
            selected_cols = sorted(selected_cols_set)  # 保持顺序稳定

        # 仅当 includeNormal=False 时排除
        if not include_normal:
            time_columns_to_exclude = {"date", "curTime", "symbol", "legalData"}
            selected_cols = [col for col in selected_cols if col not in time_columns_to_exclude]

        return selected_cols

# ===================
# ===== Model ======
# ===================

import lightgbm as lgb
class lgbModel(Model):
    def train(self, Xi, Yi):
        import pandas as pd
        from pandas.api.types import is_categorical_dtype, is_bool_dtype
        Xi = Xi.copy()
        Yi = Yi.astype("float32")
        # 识别类别特征：bool 和 category 类型
        categorical_features = []
        for col in Xi.columns:
            if is_bool_dtype(Xi[col]):
                # 转为 category（LightGBM 推荐格式）
                Xi[col] = Xi[col].astype('category')
                categorical_features.append(col)
            elif is_categorical_dtype(Xi[col]):
                categorical_features.append(col)
            else:
                # 确保其他列为数值型
                Xi[col] = pd.to_numeric(Xi[col], errors='coerce').astype('float32')
    
        # 构建模型（注意：LGBMRegressor 不需要在初始化时传 categorical_feature）
        self.model = lgb.LGBMRegressor(
            boosting_type=self.modelParam.get("boosting_type", 'gbdt'),
            objective=self.modelParam.get("objective", 'regression'),
            metric=self.modelParam.get("metric", 'rmse'),
            num_leaves=self.modelParam.get('num_leaves', 255),
            max_depth=self.modelParam.get('max_depth', -1),
            min_child_samples=self.modelParam.get('min_child_samples', 20),
            reg_alpha=self.modelParam.get('reg_alpha', 0.5),
            reg_lambda=self.modelParam.get('reg_lambda', 8),
            learning_rate=self.modelParam.get('learning_rate', 0.01),
            n_estimators=self.modelParam.get('n_estimators', 2000),
            min_gain_to_split=self.modelParam.get("min_gain_to_split", 0),
            subsample=self.modelParam.get('subsample', 0.8),
            colsample_bytree=self.modelParam.get('colsample_bytree', 0.6),
            n_jobs=self.modelParam.get("n_jobs", 16),
            random_state=self.modelParam.get("random_state", 42),
            verbose=-1
        ) 
        if categorical_features:
            cat_feat_lgb = ["name:" + col for col in categorical_features]
            print(f"类别因子:{",".join(cat_feat_lgb)}")
        else:
            cat_feat_lgb = None  # 或 [] 
        self.model.fit(Xi, Yi, categorical_feature=cat_feat_lgb)
        super().evaluate(Xi, Yi)
    def predict(self, Xi):
        # 识别类别特征：bool 和 category 类型
        from pandas.api.types import is_categorical_dtype, is_bool_dtype
        categorical_features = []
        for col in Xi.columns:
            if is_bool_dtype(Xi[col]):
                # 转为 category（LightGBM 推荐格式）
                Xi[col] = Xi[col].astype('category')
                categorical_features.append(col)
            elif is_categorical_dtype(Xi[col]):
                categorical_features.append(col)
            else:
                # 确保其他列为数值型
                Xi[col] = pd.to_numeric(Xi[col], errors='coerce').astype('float32')
        return self.model.predict(Xi)
    
