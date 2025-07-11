import MLQuant as MLQ
import os

# 需要一个模板文件夹 temp 其中包含单个子任务文件 job.py 以及对应的配置文件 Param.json
def subJobs(paramDict, workers, runJobsName='subJobs', tempDir='temp', singleMaxTime=24*3600):
    from joblib import Parallel, delayed
    import shutil, subprocess, json
    # 运行单个子任务
    def single(i):  # 在i目录下调用job.py
        try:
            process = subprocess.Popen(["python", "job.py"],\
                    cwd=f"./{runJobsName}/{i}/", stdout=subprocess.PIPE)
            process.communicate(timeout=singleMaxTime)
            return (i, False)  # 任务序号,是否超时, 
        except subprocess.TimeoutExpired:
            process.kill()
            MLQ.io.log(f"任务:'{runJobsName}/{i}'超时") 
            return (i, True)
    # 创建工作目录并提交排队任务并行
    def copy_temp(i):
        with open(os.path.join(tempDir, 'Param.json'), 'r') as f:
            param = json.load(f)
        # 按照paramDict(可能嵌套字典)修改para.json
        def editParam(paramDict, param):
            for k,v in paramDict[i].items():
                if type(v) != type(dict()):
                    param[k] = v
                else:
                    editParam(v, param[k])
            return param
        param = editParam(paramDict, param)
        shutil.copytree(tempDir, f"./{runJobsName}/{i}/")
        with open(f"./{runJobsName}/{i}/Param.json", 'w') as f:
            json.dump(param, f, indent=4)  # 复制工作目录并保存
    for i in paramDict.keys():
        copy_temp(i)
    Parallel(n_jobs=workers, backend='loky')(delayed(single)(where) \
            for where in paramDict.keys())


