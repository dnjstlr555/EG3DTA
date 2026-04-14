import os
import pickle as pkl
import pandas as pd
import joblib
import numpy as np

from os import listdir
from sklearn.model_selection import KFold

sbjinfo = '''
e01 (deep squat) - 90 repetitions
	Subject 1  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
	Subject 2  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 3  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 4  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 5  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 6  (10 repetitions): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 7  (8 repetitions):  2, 3, 5, 6, 7, 8, 9, 10
    Subject 8  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 9  (8 repetitions):  2, 3, 4, 6, 7, 8, 9, 10
    Subject 10 (10 repetitions): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

e02 (hurdle step) - 55 repetitions
	Subject 1  (6 repetitions):  2, 5, 6, 7, 8, 9 
	Subject 2  (0 repetitions):  
    Subject 3  (8 repetitions):  2, 3, 4, 5, 6, 8, 9, 10
    Subject 4  (7 repetitions):  2, 3, 4, 5, 6, 8, 9
    Subject 5  (7 repetitions):  3, 5, 6, 7, 8, 9, 10
    Subject 6  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 7  (0 repetitions):  
    Subject 8  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 9  (9 repetitions):  1, 2, 3, 4, 5, 6, 7, 8, 9
    Subject 10 (0 repetitions):

e03 (inline lunge) - 51 repetitions
	Subject 1  (6 repetitions):  4, 5, 6, 8, 9, 10
    Subject 2  (0 repetitions): 
    Subject 3  (0 repetitions):
    Subject 4  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 5  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 6  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 7  (0 repetitions):  
    Subject 8  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 9  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 10 (0 repetitions):

e04 (Side lunge) - 70 repetitions
	Subject 1  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 2  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 3  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 4  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 5  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 6  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 7  (0 repetitions):
    Subject 8  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 9  (7 repetitions):  2, 3, 4, 5, 6, 7, 8
    Subject 10 (0 repetitions):

e05 (Sit to stand) - 84 repetitions
	Subject 1  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 2  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 3  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 4  (7 repetitions):  2, 3, 5, 6, 7, 9, 10
    Subject 5  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 6  (8 repetitions):  2, 3, 4, 6, 7, 8, 9, 10
    Subject 7  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 8  (8 repetitions):  2, 3, 4, 5, 6, 7, 8, 9
    Subject 9  (7 repetitions):  2, 3, 4, 5, 6, 7, 8
    Subject 10 (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10

e06 (Standing active straight leg raise) - 73 repetitions
	Subject 1  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 2  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 3  (7 repetitions):  3, 4, 5, 6, 7, 8, 9
    Subject 4  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 5  (10 repetitions): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 6  (10 repetitions): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 7  (0 repetitions):  
    Subject 8  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 9  (10 repetitions): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 10 (0 repetitions):  

e07 (Standing shoulder abduction) - 63 repetitions
	Subject 1  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 2  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 3  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 4  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 5  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 6  (0 repetitions):  
    Subject 7  (0 repetitions):  
    Subject 8  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 9  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 10 (0 repetitions):  

e08 (Standing shoulder extension) - 63 repetitions
	Subject 1  (5 repetitions):  2, 4, 5, 6, 9
    Subject 2  (8 repetitions):  2, 3, 4, 5, 6, 7, 8, 9
    Subject 3  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 4  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 5  (8 repetitions):  2, 3, 4, 5, 6, 7, 8, 9
    Subject 6  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 7  (0 repetitions):  
    Subject 8  (6 repetitions):  3, 4, 6, 7, 8, 9
    Subject 9  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 10 (0 repetitions):  

e09 (Standing shoulder internal external rotation) - 60 repetitions
	Subject 1  (0 repetitions):  
    Subject 2  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 3  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 4  (7 repetitions):  2, 3, 4, 5, 6, 7, 8
    Subject 5  (10 repetitions): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 6  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 7  (0 repetitions):  
    Subject 8  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 9  (7 repetitions):  2, 3, 4, 5, 8, 9, 10
    Subject 10 (0 repetitions):

e10 (Standing shoulder scaption) - 54 repetitions
	Subject 1  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 2  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 3  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 4  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 5  (0 repetitions):  
    Subject 6  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 7  (0 repetitions):  
    Subject 8  (9 repetitions):  2, 3, 4, 5, 6, 7, 8, 9, 10
    Subject 9  (0 repetitions):  
    Subject 10 (0 repetitions):'''


def extract_rep(s):
    lines = s.strip().split('\n')[1:]  # Skip the first line
    dct = {}
    for line in lines:
        parts = line.strip().split('):')
        sbjno = int(parts[0].split('(')[0].split()[-1])
        repno = int(parts[0].split('(')[1][0])
        reps = list(map(int,parts[1].split(','))) if repno>0 else []
        dct[sbjno] = reps
    return dct
def divideblocks(s):
    exercises = s.strip().split('\ne')[:]  # Split by exercise blocks
    ex_dct = {}
    for ex in exercises:
        ex_lines = ex.strip().split('\n')
        ex_name = ex_lines[0].split('(')[0].strip()
        ex_name = int(ex_name) if ex_name[0]!='e' else int(ex_name[1:])
        ex_info = '\n'.join(ex_lines)
        ex_dct[ex_name] = extract_rep(ex_info)
    return ex_dct

def save_exmaps():
    exmaps = divideblocks(sbjinfo)
    with open('./data/ex_sbj_map.pkl', 'wb') as f:
        pkl.dump(exmaps, f)

def getdata(path):
    return pd.read_csv(path, header=None, delim_whitespace=True)
def getscore(ex, prefix):
    assert prefix in ['Correct', 'Incorrect']
    scores = pd.read_csv(f'./data/uiprmd/Data and Scores csv/{prefix}_score_S{ex}.csv', header=None).iloc[:, 0].values
    return scores
def getsbjids(exmap, ex):
    crex = exmap[ex].copy()
    for k in exmap:
        crex[k] = len(crex[k])
    pair = sorted([(k, crex[k]) for k in crex], key=lambda x: x[0])
    ids = []
    for k, v in pair:
        ids += [k] * v
    return ids
def constructex(exmap, ex, prefix):
    assert prefix in ['Correct', 'Incorrect']
    datapath = f"./data/uiprmd/{prefix} Movements"
    filename = f"e{ex:02d}_r"
    data = []
    datacounts = sum([len(v) for k, v in exmap[ex].items()])
    for f in range(1, datacounts + 1):
        mypath = f"{datapath}/{filename}{f}{'_inc' if prefix == 'Incorrect' else ''}.txt"
        df = getdata(mypath).to_numpy()
        data.append(df)
    lbs = getscore(ex, prefix)
    ids = getsbjids(exmap, ex)
    zipdata = list(zip(data, lbs, ids))
    return zipdata
def constructall(exmap):
    dct = {}
    for ex in exmap:
        correctdata = constructex(exmap, ex, 'Correct')
        incorrectdata = constructex(exmap, ex, 'Incorrect')
        dct[ex] = correctdata + incorrectdata
    return dct

def save_raw():
    with open('./data/ex_sbj_map.pkl', 'rb') as f:
        exmap = pkl.load(f)
    alldata = constructall(exmap)
    joblib.dump(alldata, './data/uiprmd_raw.pkl')

class Data_Loader():
    def __init__(self, path):
        self.exs = joblib.load(path)
    
    def getsubjects(self, exs, exercise):
        subjects = [i[2] for i in exs[exercise]]
        return list(set(subjects))
    
    def select_k(self, subjects, k, random_state=0):
        #subject에서 k-fold로 나누는 함수
        stf=KFold(n_splits=k, shuffle=True, random_state=random_state)
        selected_subjects = []
        train_subjects = []
        iter_split=stf.split(subjects)
        for train, test in iter_split:
            selected_subjects.append([subjects[i] for i in test])
            train_subjects.append([subjects[i] for i in train])
        return selected_subjects, train_subjects
    
    def getdata(self, exs, exercise, subjects):
        data = []
        labels = []
        for i in exs[exercise]:
            if i[2] in subjects:
                mydata=np.stack(i[0]).reshape(-1,39,3)
                #mydata is (time, joints*3)                
                timelimit=(mydata.shape[0]//100)*100
                mydata=mydata[:timelimit]
                mydata=mydata.reshape((-1, 100, 39, 3))
                #mydata is (n, 100, joints*3)
                for j in range(mydata.shape[0]):
                    data.append(mydata[j])
                    labels.append(i[1])
        return np.stack(data), np.array(labels)
    
    def getkdata(self, exercise, k=None):
        sbs=self.getsubjects(self.exs, exercise)
        selected_subjects, train_subjects = self.select_k(sbs, k)
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for i in selected_subjects:
            data, labels = self.getdata(self.exs, exercise, i)
            test_data.append(data)
            test_labels.append(labels)
        for i in train_subjects:
            data, labels = self.getdata(self.exs, exercise, i)
            train_data.append(data)
            train_labels.append(labels)

        return train_data, train_labels, test_data, test_labels, selected_subjects, train_subjects
    
if __name__ == "__main__":
    if not os.path.exists('./data/ex_sbj_map.pkl'):
        save_exmaps()
    if not os.path.exists('./data/uiprmd_raw.pkl'):
        save_raw()

    loader=Data_Loader(path="./data/uiprmd_raw.pkl")
    ex_kfold={}
    for i in range(1, 11):
        train_data, train_labels, test_data, test_labels, selected_subjects, train_subjects = loader.getkdata(exercise=i, k=5)
        print(train_data[0].shape)
        ex_kfold[i]={
            "train_data": train_data,
            "train_labels": train_labels,
            "test_data": test_data,
            "test_labels": test_labels,
            "selected_subjects": selected_subjects,
            "train_subjects": train_subjects
        }
        print(f"Exercise {i} k-fold data loaded")
    joblib.dump(ex_kfold, './data/uiprmd_kfold.pkl')