import math
import datetime
import sys
import numpy as np

"""
BFGS优于DFP的原因在于,BFGS有自校正的性质(self-correcting property).通俗来说，如果某一步BFGS对
Hessian阵的估计偏了,导致优化变慢,那么BFGS会在较少的数轮迭代内（取决于线搜索的质量）,校正估计的Hessian阵。
"""

class LR:
    def __init__(self, train_file_name, test_file_name, predict_result_file_name):
        self.train_file = train_file_name
        self.predict_file = test_file_name
        self.predict_result_file = predict_result_file_name
        self.max_iters = 200
        self.feats = []
        self.labels = []
        self.feats_test = []
        self.labels_predict = []
        self.param_num = 0
        self.weight = []
        self.delt_weight=[]#两次权重之差
        self.delt_g=[]#两次导数之差
        self.inv_H=[]#Hessian矩阵的逆
        self.weight_last=[]
        self.g_last=[]
        self.n_component=0

    def loadDataSet(self, file_name, label_existed_flag):
        feats = []
        labels = []
        fr = open(file_name)
        lines = fr.readlines()
        for line in lines:
            temp = []
            allInfo = line.strip().split(',')
            dims = len(allInfo)
            if label_existed_flag == 1:
                for index in range(dims-1):
                    temp.append(float(allInfo[index]))
                feats.append(temp)
                labels.append(float(allInfo[dims-1]))
            else:
                for index in range(dims):
                    temp.append(float(allInfo[index]))
                feats.append(temp)
        fr.close()
        feats = np.array(feats)
        labels = np.array(labels)
        return feats, labels

    def loadTrainData(self):
        self.feats, self.labels = self.loadDataSet(self.train_file, 1)

    def loadTestData(self):
        self.feats_test, self.labels_predict = self.loadDataSet(
            self.predict_file, 0)

    def savePredictResult(self):
        print(self.labels_predict)
        f = open(self.predict_result_file, 'w')
        for i in range(len(self.labels_predict)):
            f.write(str(self.labels_predict[i])+"\n")
        f.close()

    def sigmod(self, x):
        return 1/(1+np.exp(-x))

    def printInfo(self):
        print(self.train_file)
        print(self.predict_file)
        print(self.predict_result_file)
        print(self.feats)
        print(self.labels)
        print(self.feats_test)
        print(self.labels_predict)

    def initParams(self):
        self.weight = np.ones((self.param_num,), dtype=np.float)
        self.inv_H=np.identity(self.param_num)

    def compute_ypred(self, recNum, param_num, feats, w):
        return self.sigmod(np.dot(feats, w))

    def error_rate(self, recNum, label, preval):
        return np.power(label - preval, 2).sum()

    def hessian_bfgs(self,delt_weight,delt_g):
        d=np.dot(delt_g.T,delt_weight)
        if math.fabs(d)<1e-4:#分母为0，返回单位阵
            return np.identity(self.param_num)

        A=np.zeros((self.param_num,self.param_num))
        for i in range(self.param_num):
            A[i]=delt_weight[i]*delt_g
        A /= d
        A =np.identity(self.param_num)-A

        B=np.zeros((self.param_num,self.param_num))
        for i in range(self.param_num):
            B[i]=delt_g[i]*delt_weight
        B/=d
        B=np.identity(self.param_num)-B

        C = np.zeros((self.param_num, self.param_num))
        for i in range(self.param_num):
            C[i]=delt_weight[i]*delt_weight
        C/=d

        self.inv_H=np.dot(np.dot(A,self.inv_H),B)+C

    def pca_fit_transform(self,feas,n_component):
        self.n_component=n_component
        self.mean=np.mean(feas,axis=0)
        feas -=self.mean
        U,D,Vt=np.linalg.svd(feas)
        self.V=Vt.T
        return feas.dot(self.V[:,0:self.n_component])

    def pca_transform(self,feas):
        feas -=self.mean
        return feas.dot(self.V[:,0:self.n_component])

    def train(self):
        self.loadTrainData()
        recNum = len(self.feats)
        # self.feats=self.pca_fit_transform(self.feats,500)
        self.param_num = len(self.feats[0])
        self.initParams()
        ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S,f'
        for i in range(self.max_iters):
            preval = self.compute_ypred(recNum, self.param_num,
                                  self.feats, self.weight)
            sum_err = self.error_rate(recNum, self.labels, preval)
            if i%30 == 0:
                print("Iters:" + str(i) + " error:" + str(sum_err))
                theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
                print(theTime)
            if i==0:
                err = preval - self.labels
                # 计算损失函数一阶导
                g_w = np.dot(self.feats.T, err)
                g_w /= recNum
                #保存g_0
                self.g_last=g_w
                #保存w_0
                self.weight_last=self.weight.copy()
                #更新weight得
                self.weight -= np.dot(self.inv_H, g_w)
            else:
                err = preval - self.labels
                # 计算g(t+1)
                g_w = np.dot(self.feats.T, err)
                g_w /= recNum

                #计算两次导数差值 g(t+1)-g(t)
                self.delt_g=g_w-self.g_last
                #保存g(t)
                self.g_last=g_w

                #计算两次权重插值w(t+1)-w(t)
                self.delt_weight=self.weight-self.weight_last
                # dfp求hessian矩阵的逆
                self.hessian_bfgs(self.delt_weight,self.delt_g)

                #保存w(t)
                self.weight_last = self.weight.copy()
                #牛顿法更新w(t+1)
                self.weight =self.weight- np.dot(self.inv_H, g_w)

    def predict(self):
        self.loadTestData()
        # self.feats_test=self.pca_transform(self.feats_test)
        preval = self.compute_ypred(len(self.feats_test),
                              self.param_num, self.feats_test, self.weight)
        self.labels_predict = (preval+0.5).astype(np.int)
        self.savePredictResult()

def print_help_and_exit():
    print("usage:python3 main.py train_data.txt test_data.txt predict.txt [debug]")
    sys.exit(-1)

def parse_args():
    debug = False
    if len(sys.argv) == 2:
        if sys.argv[1] == 'debug':
            print("test mode")
            debug = False
        else:
            print_help_and_exit()
    return debug

if __name__ == "__main__":
    debug = parse_args()

    train_file = r".\data\train_data.txt"
    test_file = r".\data\test_data.txt"
    predict_file = r".\data\result.txt"
    lr = LR(train_file, test_file, predict_file)
    lr.train()
    lr.predict()

    if debug:
        answer_file = r".\data\answer.txt"
        f_a = open(answer_file, 'r')
        f_p = open(predict_file, 'r')
        a = []
        p = []
        lines = f_a.readlines()
        for line in lines:
            a.append(int(float(line.strip())))
        f_a.close()

        lines = f_p.readlines()
        for line in lines:
            p.append(int(float(line.strip())))
        f_p.close()

        print("answer lines:%d" % (len(a)))
        print("predict lines:%d" % (len(p)))

        errline = 0
        for i in range(len(a)):
            if a[i] != p[i]:
                errline += 1

        accuracy = (len(a)-errline)/len(a)
        print("accuracy:%f" %(accuracy))

