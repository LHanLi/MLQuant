import numpy as np
import pandas as pd
from MLQuant.modeling import Filter, Model
import time

# ===================
# ===== Filter ======
# ===================

# 固定因子池，直接从featureParma['featureNames']输入
class fixFilter(Filter):
    def filtFeature(self, data):
        return self.featureParam['featureNames']

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

        if "randomSelect" in self.featureParam.keys():
            import random
            return random.sample(selected_cols, self.featureParam["randomSelect"])
        else:
            return selected_cols

# ===================
# ===== Model ======
# ===================

# modelParam = {"selectModel":"MLQuant.sample.elaModel"}
class elaModel(Model):
    def train(self, Xi, Yi):
        from sklearn.linear_model import ElasticNet
        from sklearn.linear_model import LinearRegression
        alpha = self.modelParam.get('alpha', 0)
        l1_ratio = self.modelParam.get("l1_ratio", 0.5)
        intercept = self.modelParam.get('intercept', False)
        # 模型训练
        if alpha==0:
            model = LinearRegression(fit_intercept=intercept, n_jobs=-1)
        else:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=intercept)
        self.model = model.fit(Xi, Yi)
        self.store['nonZeroRatio'] = (self.model.coef_!=0).mean()
        self.store["coef"] = list(self.model.coef_)
    def predict(self, Xi):
        return self.model.predict(Xi)

# modelParam = {"selectModel":"MLQuant.sample.lgbModel"}
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
            learning_rate=self.modelParam.get('learning_rate', 0.01),
            n_estimators=self.modelParam.get('n_estimators', 2000),
            num_leaves=self.modelParam.get('num_leaves', 255),
            max_depth=self.modelParam.get('max_depth', -1),
            min_child_samples=self.modelParam.get('min_child_samples', 20),
            min_child_weight=self.modelParam.get('min_child_weight', 0),
            reg_alpha=self.modelParam.get('reg_alpha', 0.5),
            reg_lambda=self.modelParam.get('reg_lambda', 8),
            min_split_gain=self.modelParam.get("min_split_gain", 0),
            subsample=self.modelParam.get('subsample', 0.8),
            colsample_bytree=self.modelParam.get('colsample_bytree', 0.6),
            n_jobs=self.modelParam.get("n_jobs", 16),
            random_state=self.modelParam.get("random_state", 42),
            verbose=-1
        ) 
        if categorical_features:
            cat_feat_lgb = ["name:" + col for col in categorical_features]
            self.log += f"类别因子:{','.join(cat_feat_lgb)}"
        else:
            cat_feat_lgb = None  # 或 [] 
        self.model.fit(Xi, Yi, categorical_feature=cat_feat_lgb)
        # 因子重要性
        self.store["importance_split"] = list(self.model.feature_importances_)
        self.store["importance_gain"] = list(self.model.booster_.feature_importance(importance_type='gain'))
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
    


class mlpModel(Model):
    import torch
    class MLPRegressor(torch.nn.Module):
        def __init__(self, input_dim, param):
            from torch import nn
            super().__init__()
            self.param = param
            # 可选激活函数
            activation_map = {
                'relu': nn.ReLU(inplace=True),
                'leaky_relu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
                'elu': nn.ELU(alpha=1.0, inplace=True),
                'gelu': nn.GELU(),
                'silu': nn.SiLU(),  # 又称Swish
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid()
            }
            self.activation = activation_map[param.get('activation', 'relu')]  # 默认为ReLU
            self.residual_at = param.get('residual_at', []) # 指定层添加残差连接(输出=本层正常结果+本层输入)
            self.layers = nn.ModuleList()
            dims = [input_dim] + param['hidden_dims']
            # 如果r不是整数表示这一层w维度为上一层的r倍
            for i in range(len(dims)):
                if dims[i]%1!=0:
                    dims[i] = int(dims[i]*predim)
                    predim = dims[i]
                else:
                    predim = dims[i]
            # 动态构建网络层
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i+1]
                block = self._make_block(in_dim, out_dim, param.get('use_bn', False), param.get('dropout', 0))
                self.layers.append(block)
            self.fc_out = nn.Linear(dims[-1], 1, bias=self.param.get('bias', True)) # 输出层（无激活函数）
        def _make_block(self, in_dim, out_dim, use_bn, dropout):
            from torch import nn
            layers = [nn.Linear(in_dim, out_dim, bias=self.param.get('bias', True))]
            if use_bn:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(self.activation) 
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)
        def forward(self, x):
            import torch
            # 隐藏层迭代
            x0 = x  # 是否开启输入的残差连接
            for i, layer in enumerate(self.layers):
                # 在指定层添加残差连接，不检查，如果维度不匹配直接报错
                #if i in self.residual_at and x.size(1) == layer[0].out_features:
                if i in self.residual_at:
                    identity = x
                    x = layer(x)
                    x = x + identity  # 跳跃连接
                else:
                    x = layer(x)
            if self.param.get("input_res", False):
                x = x0+x
            # 输出层
            if self.param.get('sigmoid', False):
                return torch.sigmoid(self.fc_out(x))
            else:
                return self.fc_out(x)
    def train(self, Xi, Yi):
        import torch
        # 确定训练进行设备
        self.device = torch.device(self.modelParam.get("device", 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.log += f"在 {self.device} 上执行训练\n"
        from torch.utils.data import TensorDataset, random_split, DataLoader
        # 划分训练集验证集
        #Xi = Xi.to(self.device)
        #Yi = Yi.to(self.device)
        dataset = TensorDataset(Xi, Yi)
        val_size = int(len(dataset)*self.modelParam.get("val_ratio", 0.05))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42))
        # 按batchsize划分
        batch_size = self.modelParam.get("batch_size", 3000)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # 实例化模型
        model = self.MLPRegressor(
                input_size=Xi.shape[2],
                param=self.modelParam
            ).to(self.device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log += f"model initialized with {total_params:,} trainable parameters\n"
        # 损失函数
        criterion = torch.nn.MSELoss()
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.modelParam.get("lr", 1e-3))

        # ====== 早停相关参数 ======
        patience = self.modelParam.get("patience", 10)        # 容忍多少个 epoch 没有 improvement
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None     # 保存最佳模型权重
        num_epochs = self.modelParam.get("num_epochs", 500)  # 可以设大一点，因为会早停
        # 开始epoch循环
        for epoch in range(num_epochs):
            time0 = time.time()
            model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                #time1 = time.time()
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                #self.log += f"train batch 加载至{self.device} 耗时{time.time()-time1:.1f}s\n"
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x_batch.size(0)
            train_loss /= len(train_loader.dataset)
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    pred = model(x_batch)
                    loss = criterion(pred, y_batch)
                    val_loss += loss.item() * x_batch.size(0)
            val_loss /= len(val_loader.dataset)
            # ====== 早停逻辑 ======
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # 保存当前最佳模型（深拷贝）
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
            self.log += f"Epoch {epoch+1}/{num_epochs} | 耗时: {time.time()-time0:.1f}s "+\
                  f"Train Loss: {train_loss:.8f} | "+\
                  f"Val Loss: {val_loss:.8f} | "+\
                  f"Best Val Loss: {best_val_loss:.8f}\n"
            # 检查是否触发早停
            if epochs_no_improve >= patience:
                self.log += f"Early stopping triggered after {epoch+1} epochs!\n"
                break
        # ====== 加载最佳模型权重 ======
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            self.model = model
            self.log += "Loaded best model weights based on validation loss.\n"
    def predict(self, Xi):
        import torch
        batch_size = 2000
        self.model.eval()  # 切换到评估模式
        device = next(self.model.parameters()).device
        predictions = []
        with torch.no_grad():  # 禁用梯度计算，大幅节省显存
            for i in range(0, len(Xi), batch_size):
                x_batch = Xi[i:i + batch_size].to(device)
                pred_batch = self.model(x_batch)
                predictions.append(pred_batch.cpu())  # 立即移回 CPU 节省 GPU 显存
        return torch.cat(predictions, dim=0)



class gruModel(Model):
    import torch
    class GRURegressor(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            import torch
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.output_size = output_size
            self.gru = torch.nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            self.fc = torch.nn.Linear(hidden_size, output_size)
        def forward(self, x):
            output, _ = self.gru(x) # 自动初始化h0
            last_hidden = output[:, -1, :]  # (batch, hidden_size)
            prediction = self.fc(last_hidden)  # (batch, output_size)
            return prediction
    def train(self, Xi, Yi):
        import torch
        # 确定训练进行设备
        self.device = torch.device(self.modelParam.get("device", 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.log += f"在 {self.device} 上执行训练\n"
        from torch.utils.data import TensorDataset, random_split, DataLoader
        # 划分训练集验证集
        #Xi = Xi.to(self.device)
        #Yi = Yi.to(self.device)
        dataset = TensorDataset(Xi, Yi)
        val_size = int(len(dataset)*self.modelParam.get("val_ratio", 0.05))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42))
        # 按batchsize划分
        batch_size = self.modelParam.get("batch_size", 3000)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # 实例化模型
        model = self.GRURegressor(
            input_size=Xi.shape[2],
            hidden_size=self.modelParam.get("hidden_size", 64),
            num_layers=self.modelParam.get("num_layers", 2),
            output_size=1
        ).to(self.device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log += f"model initialized with {total_params:,} trainable parameters\n"
        # 损失函数
        criterion = torch.nn.MSELoss()
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.modelParam.get("lr", 1e-3))

        # ====== 早停相关参数 ======
        patience = self.modelParam.get("patience", 10)        # 容忍多少个 epoch 没有 improvement
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None     # 保存最佳模型权重
        num_epochs = self.modelParam.get("num_epochs", 500)  # 可以设大一点，因为会早停
        # 开始epoch循环
        for epoch in range(num_epochs):
            time0 = time.time()
            model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                #time1 = time.time()
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                #self.log += f"train batch 加载至{self.device} 耗时{time.time()-time1:.1f}s\n"
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x_batch.size(0)
            train_loss /= len(train_loader.dataset)
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    pred = model(x_batch)
                    loss = criterion(pred, y_batch)
                    val_loss += loss.item() * x_batch.size(0)
            val_loss /= len(val_loader.dataset)
            # ====== 早停逻辑 ======
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # 保存当前最佳模型（深拷贝）
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
            self.log += f"Epoch {epoch+1}/{num_epochs} | 耗时: {time.time()-time0:.1f}s "+\
                  f"Train Loss: {train_loss:.8f} | "+\
                  f"Val Loss: {val_loss:.8f} | "+\
                  f"Best Val Loss: {best_val_loss:.8f}\n"
            # 检查是否触发早停
            if epochs_no_improve >= patience:
                self.log += f"Early stopping triggered after {epoch+1} epochs!\n"
                break
        # ====== 加载最佳模型权重 ======
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            self.model = model
            self.log += "Loaded best model weights based on validation loss.\n"
    def predict(self, Xi):
        import torch
        batch_size = 2000
        self.model.eval()  # 切换到评估模式
        device = next(self.model.parameters()).device
        predictions = []
        with torch.no_grad():  # 禁用梯度计算，大幅节省显存
            for i in range(0, len(Xi), batch_size):
                x_batch = Xi[i:i + batch_size].to(device)
                pred_batch = self.model(x_batch)
                predictions.append(pred_batch.cpu())  # 立即移回 CPU 节省 GPU 显存
        return torch.cat(predictions, dim=0)



class transformerModel(Model):
    import torch
    class TransformerRegressor(torch.nn.Module):
        import torch
        # 位置编码
        class PositionalEncoding(torch.nn.Module):
            def __init__(self, d_model, max_len=5000):
                import math, torch
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)  # (1, max_len, d_model)
                self.register_buffer('pe', pe)
            def forward(self, x): # x: (B, T, d_model)
                x = x + self.pe[:, :x.size(1), :]
                return x
        def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
            super().__init__()
            import torch
            self.input_proj = torch.nn.Linear(input_dim, d_model)  # 将 n → d_model
            self.pos_encoder = self.PositionalEncoding(d_model)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True  # 关键！使输入为 (B, T, d_model)
            )
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.regressor = torch.nn.Linear(d_model, 1)  # 输出标量
        def forward(self, x):
            # x: (B, T, n)
            x = self.input_proj(x)  # (B, T, d_model)
            x = self.pos_encoder(x)  # (B, T, d_model)
            x = self.transformer_encoder(x)  # (B, T, d_model)
            # 全局平均池化：对时间维度 T 求平均
            x = x.mean(dim=1)  # (B, d_model)
            out = self.regressor(x)  # (B, 1)
            return out
    def train(self, Xi, Yi):
        import torch
        # 确定训练进行设备
        self.device = torch.device(self.modelParam.get("device", 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.log += f"在 {self.device} 上执行训练\n"
        from torch.utils.data import TensorDataset, random_split, DataLoader
        # 划分训练集验证集
        #Xi = Xi.to(self.device)
        #Yi = Yi.to(self.device)
        dataset = TensorDataset(Xi, Yi)
        val_size = int(len(dataset)*self.modelParam.get("val_ratio", 0.05))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42))
        # 按batchsize划分
        batch_size = self.modelParam.get("batch_size", 3000)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # 实例化模型
        model = self.TransformerRegressor(
            input_dim=Xi.shape[2],
            d_model=self.modelParam.get("d_model", 128),
            nhead=self.modelParam.get("nhead", 8),
            num_layers=self.modelParam.get("num_layers", 2),
            dim_feedforward=self.modelParam.get("dim_feedforward", 512)
        ).to(self.device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log += f"model initialized with {total_params:,} trainable parameters\n"
        # 损失函数
        criterion = torch.nn.MSELoss()
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.modelParam.get("lr", 1e-3))

        # ====== 早停相关参数 ======
        patience = self.modelParam.get("patience", 10)        # 容忍多少个 epoch 没有 improvement
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None     # 保存最佳模型权重
        num_epochs = self.modelParam.get("num_epochs", 500)  # 可以设大一点，因为会早停
        # 开始epoch循环
        for epoch in range(num_epochs):
            time0 = time.time()
            model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                #time1 = time.time()
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                #self.log += f"train batch 加载至{self.device} 耗时{time.time()-time1:.1f}s\n"
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x_batch.size(0)
            train_loss /= len(train_loader.dataset)
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    pred = model(x_batch)
                    loss = criterion(pred, y_batch)
                    val_loss += loss.item() * x_batch.size(0)
            val_loss /= len(val_loader.dataset)
            # ====== 早停逻辑 ======
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # 保存当前最佳模型（深拷贝）
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
            self.log += f"Epoch {epoch+1}/{num_epochs} | 耗时: {time.time()-time0:.1f}s "+\
                  f"Train Loss: {train_loss:.6f} | "+\
                  f"Val Loss: {val_loss:.6f} | "+\
                  f"Best Val Loss: {best_val_loss:.6f}\n"
            # 检查是否触发早停
            if epochs_no_improve >= patience:
                self.log += f"Early stopping triggered after {epoch+1} epochs!\n"
                break
        # ====== 加载最佳模型权重 ======
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            self.model = model
            self.log += "Loaded best model weights based on validation loss.\n"
    def predict(self, Xi):
        import torch
        batch_size = 2000
        self.model.eval()  # 切换到评估模式
        device = next(self.model.parameters()).device
        predictions = []
        with torch.no_grad():  # 禁用梯度计算，大幅节省显存
            for i in range(0, len(Xi), batch_size):
                x_batch = Xi[i:i + batch_size].to(device)
                pred_batch = self.model(x_batch)
                predictions.append(pred_batch.cpu())  # 立即移回 CPU 节省 GPU 显存
        return torch.cat(predictions, dim=0)