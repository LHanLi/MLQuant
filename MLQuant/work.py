from joblib import Parallel, delayed
import MLQuant as MLQ
import shutil, subprocess, json, os


# 需要一个模板文件夹 temp 其中包含单个子任务文件 job.py 以及对应的配置文件 Param.json
# paramDict keys为工作目录名, values为该目录下配置文件参数(仅需填相对模板文件修改部分)
def copyJobFiles(paramDict, tempDir='temp'):
    # 创建工作目录并提交排队任务并行
    def copy_temp(copyLoc, paramDict):
        with open(os.path.join(tempDir, 'Param.json'), 'r') as f:
            param = json.load(f)
        # 按照paramDict(可能嵌套字典)修改para.json
        def editParam(paramDict, param):
            for k,v in paramDict.items():
                if type(v) != type(dict()):
                    param[k] = v
                else:
                    editParam(v, param[k])
            return param
        param = editParam(paramDict, param)
        shutil.copytree(tempDir, copyLoc)
        with open(os.path.join(copyLoc, "Param.json"), 'w') as f:
            json.dump(param, f, indent=4)  # 复制工作目录并保存
    # 创建工作任务
    for copyLoc in paramDict.keys():
        copy_temp(copyLoc, paramDict[copyLoc])
# 提交排队任务
def subJobs(workLocs, workers=1, singleMaxTime=24*3600):
    def single(where):  # 在i目录下调用job.py
        try:
            process = subprocess.Popen(["python", "job.py"],\
                    cwd=where, stdout=subprocess.PIPE)
            process.communicate(timeout=singleMaxTime)
            return (where, False)  # 任务序号,是否超时, 
        except subprocess.TimeoutExpired:
            process.kill()
            MLQ.io.log(f"任务:'{where}'超时") 
            return (where, True)
    results = Parallel(n_jobs=workers, backend='loky')(delayed(single)(where) \
            for where in workLocs)
    return results


