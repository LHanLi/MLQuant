import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import duckdb, os, datetime, json

#==========================================
#========= 数据读写操作 =====================
#===========================================

# 将 DataFrame 按{output_root}/{group_col1}/{group_col2}.pqt分组写入硬盘
def saveDataFrame(df, output_root = "../data/raw", max_workers: int = None):
    """
    df : 需要包含date:int64, curTime:int64
    output_root : 输出根目录。
    max_workers : 线程池最大线程数。默认为 min(16, CPU核数 + 4)。
    """
    group_cols = ('date', 'curTime')
    if not all(col in df.columns for col in group_cols):
        raise ValueError(f"DataFrame must contain columns: {group_cols}")
    
    os.makedirs(output_root, exist_ok=True)
    
    if max_workers is None:
        max_workers = min(128, (os.cpu_count() or 4) + 4)
    def _write_single_group(args):
        key, group_df = args
        if isinstance(key, (tuple, list)):
            dir_name = str(key[0])
            file_name = str(key[1])
        else:
            # 如果只有一列分组（兼容性）
            dir_name = str(key)
            file_name = "data"
        
        dir_path = os.path.join(output_root, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f"{file_name}.pqt")
        
        data_to_write = group_df
        
        table = pa.Table.from_pandas(data_to_write, preserve_index=False)
        pq.write_table(table, file_path, compression="snappy")

    # 执行分组
    groups = list(df.groupby(list(group_cols)))
    
    # 使用线程池写入
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用 as_completed 可选地监控进度或捕获异常
        futures = [executor.submit(_write_single_group, g) for g in groups]
        for future in as_completed(futures):
            # 可在此添加日志或异常处理
            future.result()  # 触发潜在异常

    print(f"Successfully wrote df({df.shape}, {len(df['date'].unique())}dates, "\
          f"{len(df['curTime'].unique())}curTimes) to '{output_root}'")
# 加载saveDataFrame存储的数据
def loadDataFrame(
    output_root: str = "../data/raw",
    date_range: tuple[int, int] | None = None,
    curTime_range: tuple[int, int] | None = (0, 240000000),
    columns: list[str] | None = None,
    max_workers: int | None = None,
    fixed_filename: str | None = None  # 新增参数
):
    """
    从 output_root 目录下读取 Parquet 文件，支持三种格式：
      1. 新结构：{output_root}/{date}.pqt
      2. 旧结构：{output_root}/{date}/{curTime}.pqt
      3. 固定文件名结构：{output_root}/{date}/{fixed_filename} （需指定 fixed_filename）

    Parameters:
    - output_root (str): Parquet 文件根目录。
    - date_range (tuple[int, int] or None): (start_date, end_date)，包含端点。例如 (20231201, 20231205)
    - curTime_range (tuple[int, int] or None): (start_time, end_time)，仅用于旧结构。
    - columns (list[str] or None): 要读取的列。
    - max_workers (int or None): 并发线程数，默认自动设置。
    - fixed_filename (str or None): 若提供，则尝试加载 {date}/{fixed_filename} 格式的文件。

    Returns:
    - pd.DataFrame: 合并后的原始 DataFrame。
    """
    output_path = Path(output_root)
    if not output_path.exists():
        raise FileNotFoundError(f"Output root directory not found: {output_root}")

    if max_workers is None:
        max_workers = min(128, (os.cpu_count() or 4) + 4)

    # 解析 date 范围
    date_min, date_max = None, None
    if date_range is not None:
        if len(date_range) != 2:
            raise ValueError("`date_range` must be a tuple of two integers (start, end).")
        date_min, date_max = date_range

    # 解析 curTime 范围（仅用于旧结构）
    time_min, time_max = None, None
    if curTime_range is not None:
        if len(curTime_range) != 2:
            raise ValueError("`curTime_range` must be a tuple of two integers (start, end).")
        time_min, time_max = curTime_range

    file_paths = []

    # === 格式1：新结构 {date}.pqt ===
    if fixed_filename is None and curTime_range is None:
        for pqt_file in output_path.glob("*.pqt"):
            try:
                date_val = int(pqt_file.stem)
            except ValueError:
                continue
            if date_min is not None and not (date_min <= date_val <= date_max):
                continue
            file_paths.append(str(pqt_file))

    # 如果已找到新结构文件，且用户未要求 fixed_filename 或 curTime_range，则跳过其他格式
    use_other_formats = not file_paths or fixed_filename is not None or curTime_range is not None

    if use_other_formats:
        # 遍历所有可能的日期目录
        for date_dir in output_path.iterdir():
            if not date_dir.is_dir():
                continue
            try:
                date_val = int(date_dir.name)
            except ValueError:
                continue
            if date_min is not None and not (date_min <= date_val <= date_max):
                continue

            added_from_this_dir = False

            # === 格式3：固定文件名 {date}/{fixed_filename} ===
            if fixed_filename is not None:
                fixed_file = date_dir / fixed_filename
                if fixed_file.exists() and fixed_file.suffix == '.pqt':
                    file_paths.append(str(fixed_file))
                    added_from_this_dir = True  # 标记已添加，避免重复加载

            # === 格式2：旧结构 {date}/{curTime}.pqt ===
            if not added_from_this_dir and curTime_range is not None:
                for pqt_file in date_dir.glob("*.pqt"):
                    if fixed_filename and pqt_file.name == fixed_filename:
                        continue  # 避免重复（虽然 unlikely）
                    try:
                        cur_time_val = int(pqt_file.stem)
                    except ValueError:
                        continue
                    if time_min is not None and not (time_min <= cur_time_val <= time_max):
                        continue
                    file_paths.append(str(pqt_file))

    if not file_paths:
        print("No files matched the specified criteria.")
        return pd.DataFrame()

    def _read_file(file_path):
        return pq.read_table(file_path, columns=columns).to_pandas()

    # 并发读取
    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_read_file, fp) for fp in file_paths]
        for future in as_completed(futures):
            dfs.append(future.result())

    final_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(file_paths)} files → DataFrame shape: {final_df.shape}")
    return final_df
#def loadDataFrame(
#    output_root: str = "../data/raw",
#    date_range: tuple[int, int] | None = None,
#    curtime_range: tuple[int, int] | None = None,
#    columns: list[str] | None = None,
#    max_workers: int | None = None
#):
#    """
#    从 output_root 目录下读取符合 {date}/{curTime}.pqt 格式的 Parquet 文件，
#    并合并为一个 DataFrame。仅加载在指定 date 和 curTime 范围内的文件。
#    
#    Parameters:
#    - output_root (str): Parquet 文件根目录。
#    - date_range (tuple[int, int] or None): (start_date, end_date)，包含端点。例如 (20231201, 20231205)
#    - curtime_range (tuple[int, int] or None): (start_time, end_time)，包含端点。例如 (93000, 150000)
#    - 读取列
#    - max_workers (int or None): 并发线程数，默认自动设置。
#    
#    Returns:
#    - pd.DataFrame: 合并后的原始 DataFrame。
#    """
#    output_path = Path(output_root)
#    if not output_path.exists():
#        raise FileNotFoundError(f"Output root directory not found: {output_root}")
#
#    if max_workers is None:
#        max_workers = min(128, (os.cpu_count() or 4) + 4)
#
#    ## 解析 date 范围
#    #date_min, date_max = None, None
#    #if date_range is not None:
#    #    if len(date_range) != 2:
#    #        raise ValueError("`date_range` must be a tuple of two integers (start, end).")
#    #    date_min, date_max = date_range
#
#    # 解析 curTime 范围
#    time_min, time_max = None, None
#    if curtime_range is not None:
#        #if len(curtime_range) != 2:
#        #    raise ValueError("`curtime_range` must be a tuple of two integers (start, end).")
#        time_min, time_max = curtime_range
#
#    # 获取读取文件
#    file_paths = []
#    # 先尝试按新结构（单个 .pqt 文件 per date）匹配
#    if date_range is not None:
#        date_min, date_max = date_range
#    else:
#        date_min = date_max = None
#
#    # 情况1：检查是否存在 {date}.pqt 格式的文件（新结构）
#    for pqt_file in output_path.glob("*.pqt"):
#        try:
#            date_val = int(pqt_file.stem)
#        except ValueError:
#            continue  # 忽略非整数文件名
#
#        if date_min is not None and (date_val < date_min or date_val > date_max):
#            continue
#
#        # 新结构下没有 curTime 粒度，因此无法按 curtime_range 过滤！
#        # 如果用户指定了 curtime_range，则不能使用新结构（或需在读取后过滤）
#        # 此处我们假设：若存在新结构文件，且用户未指定 curtime_range，则接受
#        if curtime_range is not None:
#            # 新结构无法按 curTime 过滤，跳过（或可抛出警告/错误）
#            continue
#
#        file_paths.append(str(pqt_file))
#
#    # 情况2：如果没找到新结构文件，回退到旧结构（兼容性）
#    if not file_paths:
#        # 遍历所有 date 目录（旧结构）
#        for date_dir in output_path.iterdir():
#            if not date_dir.is_dir():
#                continue
#            try:
#                date_val = int(date_dir.name)
#            except ValueError:
#                continue
#
#            if date_min is not None and (date_val < date_min or date_val > date_max):
#                continue
#
#            for pqt_file in date_dir.glob("*.pqt"):
#                try:
#                    cur_time_val = int(pqt_file.stem)
#                except ValueError:
#                    continue
#
#                if time_min is not None and (cur_time_val < time_min or cur_time_val > time_max):
#                    continue
#
#                file_paths.append(str(pqt_file))
#
#    if not file_paths:
#        print("No files matched the specified date/curTime range.")
#        return pd.DataFrame()  # 返回空 DataFrame
#
#    def _read_file(file_path):
#        return pq.read_table(file_path, columns=columns).to_pandas()
#
#    # 并发读取
#    dfs = []
#    with ThreadPoolExecutor(max_workers=max_workers) as executor:
#        futures = [executor.submit(_read_file, fp) for fp in file_paths]
#        for future in as_completed(futures):
#            dfs.append(future.result())
#
#    final_df = pd.concat(dfs, ignore_index=True)
#    print(f"Loaded {len(file_paths)} files → DataFrame shape: {final_df.shape}")
#    return final_df


#==========================================
#==========  duckdb sql 相关操作  ==============
#==========================================
def readSqlPqt(source, startdate=None, enddate=None, columns=["*"], filter="1=1", se="cc", pqtname="/*.pqt"):
    if (startdate is None) and (enddate is None):
        paths = [source]    
    else:
        if startdate is None:
            startdate = "00000000"
        if enddate is None:
            enddate = "99999999"
        paths = [os.path.join(source, d+pqtname) for d in os.listdir(source) if 
                    ((d>startdate) if se[0]=='o' else (d>=startdate))\
                    &((d<enddate) if se[1]=='o' else (d<=enddate))]
    paths_sql = "['{}']".format("', '".join(paths))
    query = f"""
        SELECT {', '.join(columns)}
        FROM read_parquet({paths_sql})
        WHERE {filter}
    """
    print(query)
    query_result = duckdb.sql(query)
    return query_result


#==========================================
#============  json相关操作 ================  
#==========================================

# 将内存中变量转化为json可以储存格式 
def converjson(v):
    conver_v = {}
    if type(v)==dict:
        return {k:converjson(v) for k,v in v.items()}
    elif type(v)==list:
        return [converjson(i) for i in v]
    elif type(v) in [str, int, float, bool]:
        return v
    elif type(v) in [np.int8, np.int16, np.int32, np.int64]:
        return int(v)
    elif type(v) in [np.float16, np.float32, np.float64]:
        return v.astype('float')
    elif type(v)==tuple:
        conver_v['conver_type'] = 'tuple'
        conver_v['values'] = [converjson(i) for i in v]
        return conver_v
    elif type(v)==np.ndarray:
        conver_v['conver_type'] = 'array'
        conver_v['object'] = converjson(v.tolist())
        conver_v['dtype'] = repr(v.dtype)[6:-1]
        return conver_v
    elif type(v)==type(pd.Series()):
        conver_v['conver_type'] = 'Series'
        conver_v['index'] = v.index.tolist()
        conver_v['values'] = list(v.values)
        return conver_v
    elif type(v)==type(pd.DataFrame()):
        conver_v['conver_type'] = 'DataFrame'
        conver_v['dict'] = v.to_dict()
        return conver_v
    else:
        print('unknown type', type(v), v)
        raise TypeError
# 将json转化为内存变量
def jsonconver(v):
    if type(v)==dict:
        if 'conver_type' in v.keys():
            if v['conver_type']=='tuple':
                return tuple([i for i in v['values']])
            elif v['conver_type']=='array':
                return np.array(jsonconver(v['object']), dtype=eval(v['dtype']))
            elif v['conver_type']=='Series':
                return pd.Series(v['values'], index=v['index'])
            elif v['conver_type']=='DataFrame':
                return pd.DataFrame(v['dict'])
            else:
                print('unknown type', type(v), v)
                raise TypeError
        else:
            return {k:jsonconver(v) for k,v in v.items()}
    elif type(v)==list:
        return [jsonconver(i) for i in v]
    else:
        return v 
# 保存json
def savejson(d, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'w') as f:
        json.dump(d, f, indent=4)
# 读取json
def readjson(filename, encoding='utf-8'):
    with open(filename, 'r', encoding=encoding) as f:
        return json.load(f) 
# 使用字典修改json文件
def editjson(d, filename):
    with open(filename, 'r') as f:
        para = json.load(f)
    for k,v in d.items():
        para[k] = v
    with open(filename, 'w') as f:
        json.dump(para, f, indent=4)

#=========================================================================
#==========  通过字符串引入类，可以将要调用的类存储于本地文件中 ===============
#=========================================================================

# 通过字符串引入package或类  "numpy.array"
import importlib
def importMyClass(*importLoc):
    if len(importLoc)==1:  # 输入一个元素, 解析字符串 
        importLoc = importLoc[0].split('.')
        moduleStr = importLoc[0]
        classList = importLoc[1:]
        if len(importLoc)==1: # 导入package
            return importlib.import_module(moduleStr)
        return importMyClass(importMyClass(moduleStr), classList)
    else: # 输入两个元素, 一个元素是package或者类, 第二个元素是类中的类
        if len(importLoc[1])==1:
            return getattr(importLoc[0], importLoc[1][0])
        else:
            return importMyClass(getattr(importLoc[0], importLoc[1][0]), importLoc[1][1:])

#==========================================
#==============  记录日志 ==================
#==========================================

def log(*txt, logLoc=''):
    try:
        #try:
        #    f = open('log.txt','a+', encoding='gbk')
        #except:
        #    f = open('log.txt','a+', encoding='utf-8')
        f = open(os.path.join(logLoc,'log.txt'),'a+', encoding='utf-8')
        write_str = ('\n'+' '*35).join([str(i) for i in txt])
        f.write('%s,        %s\n' % \
            (datetime.datetime.now(), write_str))
        f.close()
    except PermissionError as e:
        print(f"Error: {e}. You don't have permission to access the specified file.")


