import numpy as np
import pandas as pd
import duckdb, os, datetime, json

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


