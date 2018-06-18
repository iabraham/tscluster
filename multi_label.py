from tsCluster import tsCluster as tsc
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import RobustScaler, Normalizer, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.cross_decomposition import PLSCanonical
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics.scorer import make_scorer
import pickle, gzip
from numpy import s_
from sklearn import metrics
from matplotlib import pyplot as mpl
from matplotlib_venn import venn3
import numpy as np
import sys
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, hamming_loss
from sklearn.decomposition import  PCA
from datetime import datetime

# =============================== LOAD DATA
# ---- Run following one time and save the class instance to a file using pickle and gzip

# all_data = tsc.TsCluster()
# all_data.from_directory()
# # 1. ROI removal  -- Faster this way, instead of at run-time
# all_data.remove_roi('lcuneus')  # Desctructive, cannot add it back in
# all_data.remove_roi('rcuneus')
# all_data.recompute()
# all_data.save_to_file()

# # np.seterr(all='raise')
# # # -----Afterwards just load the data from the binary file. Much faster!
file_name = './data_binary_roi31.bin'
with gzip.open(file_name, 'rb') as file:
    all_data = pickle.load(file)
#
# # ============================== HOUSEKEEPING
#
# # 2. Un-fixable Motion removal
remList = ['sub_157', 'sub_158', 'sub_158', 'sub_162', 'sub_169', 'sub_182', 'sub_188', 'sub_210', 'sub_217', 'sub_903']
all_data.mod_samples(op='rem', idxs=remList)
all_data.mod_data_rem(idx_session='s1', idx_run='run3')

# 3. Fix other motion and reset
all_data.edit_time_series(sub='sub_132', run='run2', session='s2', edit_idx=s_[0:180])
all_data.edit_time_series(sub='sub_120', run='run1', session='s1', edit_idx=s_[0:290])
all_data.edit_time_series(sub='sub_135', run='run2', session='s1', edit_idx=s_[0:135, 180:230, 180:300])
all_data.recompute()
# print(all_data)


# ============================== HELPER FUNCTIONS
def label_transform(label):
    if label == 'ctr':
        return ()
    elif label == 'tin':
        return ('tin',)
    elif label == 'hl':
        return ('hl',)
    elif label == 'tin_hl':
        return ('hl', 'tin')
    else:
        raise ValueError


def visualize(label_list, title=None, block = False):
    Abc, aBc, ABc, abC, AbC, aBC, ABC = 0,0,0,0,0,0,0
    for label in label_list:
        if label == ():
            Abc += 1
        elif label == ('hl',):
            aBc += 1
            ABc += 1
        elif label == ('tin',):
            abC += 1
        elif label == ('tin', 'hl') or label == ('hl', 'tin'):
            aBC += 1
    fig = mpl.figure()
    venn3([Abc, aBc, ABc, abC, AbC, aBC, ABC], set_labels=('Controls', 'Hearing Loss', 'Tinnitus'))
    if title:
        mpl.title(title)
    mpl.show(block = block)


def my_score(true_label_list, pred_label_list):
    null_predas_null, null_predas_tin, null_predas_hl, null_predas_tinhl = 0, 0, 0, 0
    tin_predas_null, tin_predas_tin, tin_predas_hl, tin_predas_tinhl = 0, 0, 0, 0
    hl_predas_null, hl_predas_tin, hl_predas_hl, hl_predas_tinhl = 0, 0, 0, 0
    tinhl_predas_null, tinhl_predas_tin, tinhl_predas_hl, tinhl_predas_tinhl = 0, 0, 0, 0
    total_tin, total_ctr = 0,0

    def a_fun(ls):
        st = ''
        for item in ls:
            st = st + str(item)
        return st

    true_label_list = list(map(a_fun, true_label_list))
    pred_label_list = list(map(a_fun, pred_label_list))

    total_null = len([item for item in true_label_list if item == '00'])
    total_ptin = len([item for item in true_label_list if item == '01'])
    total_hl = len([item for item in true_label_list if item == '10'])
    total_tinhl = len([item for item in true_label_list if item == '11'])

    for true_label, pred_label in zip(true_label_list, pred_label_list):
        if (true_label, pred_label) == ('00','00'):
            null_predas_null, total_ctr = (val + delta  for val, delta in zip((null_predas_null, total_ctr), (1,1)))
        elif (true_label, pred_label) == ('00', '01'):
            null_predas_tin, total_ctr = (val + delta for val, delta in zip((null_predas_tin, total_ctr), (1, 1)))
        elif (true_label, pred_label) == ('00', '10'):
            null_predas_hl, total_ctr = (val + delta for val, delta in zip((null_predas_hl, total_ctr), (1, 1)))
        elif (true_label, pred_label) == ('00', '11'):
            null_predas_tinhl, total_ctr = (val + delta for val, delta in zip((null_predas_tinhl, total_ctr), (1, 1)))
        elif (true_label, pred_label) == ('01', '00'):
            tin_predas_null, total_tin = (val + delta for val, delta in zip((tin_predas_null, total_tin), (1, 1)))
        elif (true_label, pred_label) == ('01', '01'):
            tin_predas_tin, total_tin = (val + delta for val, delta in zip((tin_predas_tin, total_tin), (1, 1)))
        elif (true_label, pred_label) == ('01', '10'):
            tin_predas_hl, total_tin = (val + delta for val, delta in zip((tin_predas_hl, total_tin), (1, 1)))
        elif (true_label, pred_label) == ('01', '11'):
            tin_predas_tinhl, total_tin = (val + delta for val, delta in zip((tin_predas_tinhl, total_tin), (1, 1)))
        elif (true_label, pred_label) == ('10','00'):
            hl_predas_null, total_ctr = (val + delta for val, delta in zip((hl_predas_null, total_ctr), (1, 1)))
        elif (true_label, pred_label) == ('10', '01'):
            hl_predas_tin, total_ctr = (val + delta for val, delta in zip((hl_predas_tin, total_ctr), (1, 1)))
        elif (true_label, pred_label) == ('10', '10'):
            hl_predas_hl, total_ctr = (val + delta for val, delta in zip((hl_predas_hl, total_ctr), (1, 1)))
        elif (true_label, pred_label) == ('10', '11'):
            hl_predas_tinhl, total_ctr = (val + delta for val, delta in zip((hl_predas_tinhl, total_ctr), (1, 1)))
        elif (true_label, pred_label) == ('11', '00'):
            tinhl_predas_null, total_tin = (val + delta for val, delta in zip((tinhl_predas_null, total_tin), (1, 1)))
        elif (true_label, pred_label) == ('11','01'):
            tinhl_predas_tin, total_tin = (val + delta for val, delta in zip((tinhl_predas_tin, total_tin), (1, 1)))
        elif (true_label, pred_label) == ('11', '10'):
            tinhl_predas_hl, total_tin = (val + delta for val, delta in zip((tinhl_predas_hl, total_tin), (1, 1)))
        elif (true_label, pred_label) == ('11', '11'):
            tinhl_predas_tinhl, total_tin = (val + delta for val, delta in zip((tinhl_predas_tinhl, total_tin), (1, 1)))
        else:
            raise ValueError

    ctr_idas_ctr = null_predas_hl + null_predas_null + hl_predas_null + hl_predas_hl
    tin_idas_tin = tinhl_predas_tinhl + tin_predas_tin + tinhl_predas_tin + tin_predas_tinhl
    null = [(null_predas_null, total_null), (null_predas_tin, total_null), (null_predas_hl, total_null),
           (null_predas_tinhl, total_null)]
    tin = [(tin_predas_null, total_ptin), (tin_predas_tin,total_ptin), (tin_predas_hl,total_ptin),
            (tin_predas_tinhl,total_ptin)]
    hl = [(hl_predas_null, total_hl), (hl_predas_tin, total_hl), (hl_predas_hl,total_hl), (hl_predas_tinhl, total_hl)]
    tinhl = [(tinhl_predas_null, total_tinhl), (tinhl_predas_tin,total_tinhl), (tinhl_predas_hl,total_tinhl),
             (tinhl_predas_tinhl, total_tinhl)]
    return null, tin, hl, tinhl, [(ctr_idas_ctr,total_ctr), (tin_idas_tin, total_tin)]


def my_score_for_grid(true_label, pred_label, **kwargs):
    _, _, _, _, (ctr,tin) = my_score(true_label, pred_label)
    return (ctr[0]+ tin[0])/(tin[1]+ctr[1])


def my_score_report(null, tin, hl, tinhl, succ, name):
    print('\n+++++++++++++++ My score report: ' + name + ' +++++++++++++++ \n')
    helper = ['NULL:\t\t\t', 'TIN:\t\t\t', 'HL:\t\t\t', 'TIN-HL:\t\t\t']
    for pred_count, item in zip(null, helper):
        print ('NULL pred as ' + item + str(pred_count).zfill(2))
    print('-------------------------------------------')
    for pred_count, item in zip(tin, helper):
        print('TIN pred as ' + item  + str(pred_count).zfill(2))
    print('-------------------------------------------')
    for pred_count, item in zip(hl, helper):
        print('HL pred as ' + item + str(pred_count).zfill(2))
    print('-------------------------------------------')
    for pred_count, item in zip(tinhl, helper):
        print('TIN-HL pred as ' + item + str(pred_count).zfill(2))
    print('-------------------------------------------')
    print('CTR id-ed as CTR:\t' + str(succ[0]))
    print('TIN id-ed as TIN:\t' + str(succ[1]))
    print('---------------------------------------------------------')


class Tee(object):

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately

    def flush(self) :
        for f in self.files:
            f.flush()


# #============================== CLASSIFICATION
# # ------- Generate test train data set

train_label, train_data, groups, test_label, test_data = all_data.get_test_train(mat='FlatULM', ctr=0.6, hl=0.5,
                                                                                 tin=0.6, tin_hl = 0.35)
train_label = list(map(label_transform, train_label))
test_label = list(map(label_transform, test_label))
#
# # ------- Put class labels in binary format
MBL = MultiLabelBinarizer()
trans_train_label = MBL.fit_transform(train_label)
trans_test_label = MBL.fit_transform(test_label)
#
# # ------- Pre-process the data (PCA or CCA)
# 1. Normalize
pre_process = Normalizer()
train_data=pre_process.fit_transform(train_data, trans_train_label)
test_data = pre_process.fit_transform(test_data, trans_test_label)

# # 2.PCA
pca = PCA()
full_data_set = np.concatenate((train_data, test_data))
pca.fit(full_data_set)
train_data = pca.transform(train_data)
test_data = pca.transform(test_data)

# # 3. CCA
# # pls = PLSCanonical(n_components=20)
# # train_data, _ = pls.fit_transform(train_data, trans_train_label)
# # test_data = pls.transform(test_data)

# visualize(train_label, title='Actual populations')

# ------- Fit classifier and predict

# clf3 = OneVsRestClassifier(GaussianProcessClassifier(max_iter_predict= 1000, multi_class='one_vs_rest'))
# clf4 = OneVsRestClassifier(SGDClassifier(class_weight='balanced', warm_start=True, tol=0.001))

# clf3.fit(train_data, trans_train_label)
# clf4.fit(train_data, trans_train_label)

tp_gbc = [{'estimator__max_depth': range(2,6), 'estimator__min_samples_split': range(2,10),
           'estimator__min_samples_leaf': range(2,6), 'estimator__loss': ['deviance', 'exponential'],
           'estimator__learning_rate': [0.05, 0.1, 0.2], 'estimator__n_estimators': [100,1000],
           'estimator__min_impurity_decrease': [0, 0.25]}]

tp_svm = [{'estimator__kernel': ['poly', 'sigmoid'], 'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
           'estimator__gamma': [1, 10, 100, 1000,10000], 'estimator__coef0':[0.001, 0.01, 0.1, 0, 1, 10, 100],
           'estimator__degree': [1,2,3,4]}]

tp_knn = [{'n_neighbors': [1,2,3,4,5], 'weights': ['uniform', 'distance'], 'p': [1,2,3,4,5],
           'metric':['manhattan','chebyshev','cosine', 'hamming', 'canberra', 'braycurtis']}]

scorer = make_scorer(my_score_for_grid, greater_is_better=True)
gkf = GroupKFold(n_splits=5)
clf1 = GridSearchCV(OneVsRestClassifier(GradientBoostingClassifier(warm_start=True)), param_grid=tp_gbc,
                    cv=gkf, iid=False,n_jobs=8, verbose=True, scoring=scorer)
clf2 = GridSearchCV(OneVsRestClassifier(SVC(class_weight='balanced', tol=0.1)), param_grid=tp_svm, cv=gkf, iid=False,
                    n_jobs=4, verbose=True, scoring=scorer)

clf3 = GridSearchCV(KNeighborsClassifier(), param_grid=tp_knn, cv=gkf, iid=False, n_jobs=4, verbose=True, scoring=scorer)

# to_optimize = {'Gradient Boost': clf1 'C-SVM': clf2, 'k-NN Classifier': clf3}
to_optimize = {'C-SVM': clf2, 'k-NN Classifier': clf3}

# with open('./optimal_classifier.bin', 'rb') as file:
#    optimal = pickle.load(file)

# optimal['k-NN Classifier'].fit(train_data, trans_train_label)
# predicted = optimal['k-NN Classifier'].predict(test_data)
# print(predicted)

logfile = open('test.txt', 'w')

original_stderr = sys.stderr
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, logfile)
sys.stderr = sys.stdout

print('Process starting at: ' + str(datetime.now()))
print('\n{{{{{{{{{{{{{{{{{{\t \t Data in use \t \t}}}}}}}}}}}}}}}}}}')
print(all_data)

optimal_dict = dict()

for (name, classififer) in to_optimize.items():
    print('\n ==============> Beginning grid search on ' + name + ' over parameters: \n')
    print(str(classififer.param_grid) + '\n')
    classififer.fit(train_data, trans_train_label, groups=groups)
    print('Best parameters: ')
    print(classififer.best_params_)
    optimal = classififer.best_estimator_
    optimal_dict[name] = optimal
    optimal.fit(train_data, trans_train_label)
    prediction = optimal.predict(test_data)
    pred_labels = MBL.inverse_transform(prediction)
    null, tin, hl, tinhl, succ = my_score(trans_test_label, prediction)
    my_score_report(null, tin, hl, tinhl, succ, name)
    print(classification_report(trans_test_label, prediction))
    print(my_score_for_grid(trans_test_label, prediction))

print('Process ended at: ' + str(datetime.now()))

optimal_to_file = './optimal_classifier.bin'
with open(optimal_to_file, 'wb') as file:
    pickle.dump(optimal_dict, file, protocol=-1)

sys.stdout = original_stdout
sys.stderr = original_stderr
logfile.close()
