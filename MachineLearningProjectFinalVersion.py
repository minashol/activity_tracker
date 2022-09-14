# Τελική εργασία στο μάθημα: Ειδικά Θέματα Ι: Μηχανική Μάθηση, των φοιτητών: Δαμιανίδου Μαριάννας, Μάρκου Παντελή και Χολέβα Μηνά

# ΠΕΡΙΕΧΟΜΕΝΑ:

# block 1: επεξεργασία δεδομένων και εξαγωγή χαρακτηριστικών
# block 2: γραφήματα στατιστικής ανάλυσης
# block 3: μοντέλο - SVM
# block 4: μοντέλο - Δέντρα Απόφασης
# block 5: μοντέλο - MLP




# ΟΔΗΓΙΕΣ ΕΚΤΕΛΕΣΗΣ ΤΟΥ ΚΩΔΙΚΑ:


# 1. για να διαβάσει το πρόγραμμα τα csv αρχεία με τα δεδομένα των 10 συμμετεχόντων πρέπει να δημιουργηθεί το εξής μονοπάτι:
#
#                  "~/Desktop/ML_assignment/Clean Data/Participant_x.csv"    όπου x = 1, 2, ..., 10 οι συμμετέχοντες
# 
#    τα 10 csv αρχεία ανήκουν στον φάκελο Clean Data, ο οποίος είναι υποφάκελος του ML_assignment, ο οποίος βρίσκεται στην επιφάνεια εργασίας.
#
# 2. το πρώτο block που δεν είναι σημειωμένο σαν σχόλιο να τρέχει πάντα.
#
# 3. τα υπόλοιπα blocks: 2,3,4,5 τα οποία περιέχουν:    στατιστικά γραφήματα: [2], και 
#
#                                                       τα 3 μοντέλα: [3], [4], [5] 
#
#                                                                                           να τρέχουν ξεχωριστά το καθένα. 
#
# 4. εάν για κάποιο block υπάρχουν επιπλέον οδηγίες, θα βρίσκονται στην αρχή του block.



# Τα blocks διαχωρίζονται με το ακόλουθο σχέδιο:



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



''' block κώδικα νούμερο 1 - ΑΡΧΗ '''

""" οδηγίες: να τρέχει πάντα αυτό το block """

# ----------------------------------------1. Επεξεργασία Δεδομένων -----------------------------------------------------------

## 1.1 importing packages ######################################################################
from pyexpat import model
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import scipy
from scipy import stats, signal
import statistics
import csv
import graphviz
# Machine learning libraries
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV

# 1.2 φόρτωση συναρτήσεων #####################################################################

# νόρμα - συνάρτηση για το μέτρο του 3d διανύσματος
def norm(x, y, z):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)

# φόρτωση test set
def load_testdata(X, y, i):  
    ix = i*1241
    # Επιλέγουμε τα X, y δεδομένα για τον i συμμετέχων
    testX = X.iloc[ix:ix+1241, :]
    testy = y.iloc[ix:ix+1241]
    return testX, testy 

# φόρτωση train set
def load_traindata(X, y, i):
    ix = i*1241
    # Επιλέγουμε τα X, y δεδομένα για όλους τους συμμετέχοντες εκτός του i συμμετέχων
    trainX = pd.concat((X.iloc[:ix, :], X.iloc[ix+1241:, :]))
    trainy = pd.concat((y.iloc[:ix], y.iloc[ix+1241:]))
    return trainX, trainy

# Γράφημα accuracy σε σχέση με τις τιμές C-gamma, που έτρεξε η συνάρτηση GridSeachCV
# Source: stackoverflow (https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv)
def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1), order='F')

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('SVM with RBF kernel Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

# 1.2 φόρτωση συνόλου δεδομένων  ##############################################################

# Δημιουργία λεξικού για τα αρχεία
files = {0: "~/Desktop/ML_assignment/Clean Data/Participant_1.csv",
         1: "~/Desktop/ML_assignment/Clean Data/Participant_2.csv",
         2: "~/Desktop/ML_assignment/Clean Data/Participant_3.csv",
         3: "~/Desktop/ML_assignment/Clean Data/Participant_4.csv",
         4: "~/Desktop/ML_assignment/Clean Data/Participant_5.csv",
         5: "~/Desktop/ML_assignment/Clean Data/Participant_6.csv",
         6: "~/Desktop/ML_assignment/Clean Data/Participant_7.csv",
         7: "~/Desktop/ML_assignment/Clean Data/Participant_8.csv",
         8: "~/Desktop/ML_assignment/Clean Data/Participant_9.csv",
         9: "~/Desktop/ML_assignment/Clean Data/Participant_10.csv"}

X = pd.DataFrame(columns=['Mean', 
                          'STD', 
                          'Skew', 
                          'Max', 
                          'Min', 
                          'Range',
                          'Welch1','Welch2','Welch3','Welch4','Welch5', 'Welch6','Welch7','Welch8' ])  
y = pd.DataFrame(columns=['Activity'])

for i in range(10):
    data = pd.read_csv(files[i], skiprows=1)
    data.rename(columns={'Ax.1': 'Ax1', 'Ay.1': 'Ay1', 'Az.1': 'Az1'}, inplace=True)
    data['A'] = norm(data.Ax1, data.Ay1, data.Az1)
    data.rename(columns={'Unnamed: 69': 'Activity'}, inplace=True)
    data = data.iloc[:, -2:]
    data['Activity'] = data['Activity'].map(
        {'walking': 0, 
        'standing': 1, 
        'jogging': 2, 
        'sitting': 3, 
        'biking': 4, 
        'upstairs': 5,
        'upsatirs': 5,            # λανθασμένο όνομα κλάσης στα δεδομένα
        'downstairs': 6})
    print("Are there values > 1000: ",np.any(data.A > 1000))  # έλεγχος για τιμές > 1000

    # εάν οι τιμές είναι > 1000 αντικαθίστανται από την προηγούμενη τιμή μέτρου
    for j in range(63000):
        if data.iloc[j,1]>1000:
            data.iloc[j,1]=data.iloc[j-1,1]
        j=j+1
    
    print("missing values:", np.any(np.isnan(data.Activity)))  # έλεγχος για missing values
    print("data shape:", data.shape)                           # διαστάσεις δεδομένων
    print(i+1, "to 10" "\n")


# ----------------------------------------2. Εξαγωγή Χαρακτηριστικών -----------------------------------------------------------

# 2.1 Δημιουργία παραθύρων ####################################################################

    for l in range(0, 62001, 50):                 
        window = data.iloc[l:l + 1001, :] 
        mean = np.mean(window.A)
        std = np.std(window.A)
        skew = stats.skew(window.A)
        max = np.max(window.A)
        min = np.min(window.A)
        fx, Pxx = signal.welch(window.A)
        r = max - min
        Pxx1, Pxx2, Pxx3, Pxx4, Pxx5, Pxx6, Pxx7, Pxx8 = np.max(Pxx[:15]), np.max(Pxx[16:30]), np.max(Pxx[31:45]), np.max(Pxx[46:60]), np.max(Pxx[61:75]), np.max(Pxx[76:90]), np.max(Pxx[91:105]), np.max(Pxx[106:])
        m=window['Activity'].value_counts().idxmax()
        X.loc[X.shape[0]] = [mean, std, skew, max, min, r, Pxx1, Pxx2, Pxx3, Pxx4, Pxx5, Pxx6, Pxx7, Pxx8]
        y.loc[y.shape[0]] = m

print('Data shape and head()')
print(X.shape)
print(y.shape)

print(X.head())
print(y.head())

# Κανονικοποίηση δεδομένων
scaler = StandardScaler()
scaler.fit(X)       
d = scaler.transform(X)
X_scaled = pd.DataFrame(d, columns=['Mean', 
                                    'STD', 
                                    'Skew', 
                                    'Max', 
                                    'Min', 
                                    'Range', 
                                    'Welch1','Welch2','Welch3','Welch4','Welch5', 'Welch6','Welch7','Welch8'])

# Αποθήκευση σε αρχείο csv (προαιρετικό)
'''
X_scaled.to_csv(r'~/Desktop/ML_assignment/X_new.csv', index=False)
y.to_csv(r'~/Desktop/ML_assignment/y_new.csv', index=False)
'''


''' block κώδικα νούμερο 1 - ΤΕΛΟΣ '''

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


''' block κώδικα νούμερο 2 - ΑΡΧΗ '''

# 2.2 Έλεγχοι και οπτικοποίηση των δεδομένων ##################################################

# ΠΡΟΕΙΔΟΠΟΙΗΣΗ: ΠΟΛΛΑ ΠΑΡΑΘΥΡΑ !
# RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) 
# are retained until explicitly closed and may consume too much memory

'''

# 1. (Raw data plot) Έλεγχουμε το ποσοστό που κατέχει η κάθε δραστηριότητα στα δεδομένα και τα απεικονίζουμε σε bar plot
activities = ['walking', 'standing', 'jogging', 'sitting' , 'biking', 'upstairs', 'downstairs']
values = data.groupby(['Activity']).count().stack().tolist()
plt.figure(figsize=(8, 6))
# Horizontal Bar Plot
plt.xlabel("Δραστηριότητες")
plt.ylabel("Αριθμός καταγραφών")
plt.title("Καταγραφές για κάθε δραστηριότητα")
plt.ylim([0, 10000])
plt.bar(activities, values)
plt.show() 

# 2. Γράφημα (Line plot) του μέτρου της επιτάχυνσης (Raw data) ανα δραστηριότητα ως προς τον χρόνο
fig, axs = plt.subplots(2)
fig.suptitle('Σήμα του επιταχυνσιόμετρου - Δραστηριότητα')
axs[0].plot(data['A'])
axs[0].set_title('Μέτρο επιτάχυνσης')
axs[1].plot(data['Activity'])
axs[1].set_title('Δραστηριότητα', y=-0.01)


X_scaledy = X_scaled.join(y)
# 3. Ιστόγραμμα (Histogram) όλων των χαρακτηριστικών ανα δραστηριότητα
for l in range(len(X_scaled.columns)):
    plt.figure(figsize=(8, 6))
    for i in range(7):
        plt.suptitle('Ιστόγραμμα ανά δραστηριότητα για ' + X_scaled.columns[l])
        plt.subplot(7, 1, i+1)
        plt.hist(X_scaledy.loc[X_scaledy['Activity'] == i][X_scaled.columns[l]], bins=100)
        plt.title('Histogram of ' + X_scaled.columns[l] +' for ' +activities[i], y=0, loc='right', size=7)
        plt.yticks([])
    plt.show()

# 4. Θηκόγραμμα (Boxplot) του μέτρου της επιτάχυνσης ανα δραστηριότητα
for l in range(len(X_scaled.columns)):
    plt.figure(figsize=(8, 6))
    plt.suptitle('Θηκόγραμμα για το χαρακτηριστικό ' + X_scaled.columns[l])
    dataplt=list()
    for i in range(7):
        dataplt.append(X_scaledy.loc[X_scaledy['Activity'] == i][X_scaled.columns[l]])      
    plt.boxplot(dataplt, labels=activities)
    plt.show()

'''

''' block κώδικα νούμερο 2 - ΤΕΛΟΣ '''


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


''' block κώδικα νούμερο 3 - ΑΡΧΗ '''

""" οδηγίες: μπορεί να τρέξει ολόκληρο το block μονομιάς - ενδέχεται να πάρει ~ έως 10 λεπτά (μαρκαρισμένες με σχόλια είναι οι διαδικασίες
    εύρεσης βέλτιστων τιμών παραμέτρων <<RandomizedSearchCV>> και  <<GridSearchCV>>). """



# ----------------------------------------3. Δημιουργία μοντέλων -----------------------------------------------------------


####################################################################################################################
# 3.1) SVM με πυρήνα Radial Basis Function (RBF)
####################################################################################################################


# Ακολουθούμε τη μέθοδο αξιολόγησης Leave-One-Subject-Out (loso)
gp=[]
for i in range(10):
    for l in range(1241):
        gp.append(i)

gp = np.array(gp)
loso = LeaveOneGroupOut()


SVM_Model = SVC()
'''
# Για να κάνουμε μια αρχική αναζήτηση των βέλτιστων παραμέτρων C και gamma για τη μέθοδο ταξινόμησης SVM με πυρήνα Radial Basis Function (RBF)
# Ως τιμές των C - gamma θα παρέχουμε μια σειρά δυνάμεων του 2 και θα χρησιμοποιήσουμε την συνάρτηση RandomizedSearchCV
# ,ώστε να βρούμε κάποιες αρχικές βέλτιστες τιμές χωρίς μεγάλο υπολογιστικό κόστος
C_range = 2. ** np.arange(-3, 4+1)
gamma_range = 2. ** np.arange(-6, 1+1)
param_distr = {'C': C_range, 'gamma':gamma_range, 'kernel': ['rbf']}

svm_Grid = RandomizedSearchCV(estimator = SVM_Model, param_distributions = param_distr, cv=loso, n_jobs=-1, n_iter=60)
svm_Grid.fit(X_scaled, np.ravel(y), groups=gp)
print("The best parameters for SVM are: ", svm_Grid.best_params_, "with a score", svm_Grid.best_score_)
# Από το RandomizedSearch παίρνουμε το πρακάτων αποτέλεσμα
#The best parameters for SVM are:  {'kernel': 'rbf', 'gamma': 0.125, 'C': 2.0} with a score 0.8116841257050765

# Συνεχίζουμε την αναζήτηση των βέλτιστων παραμέτρων C - gamma πιο στοχευμένα πλέον
# μέσω της συνάρτησης GridSearchCV
C_range = 2. ** np.arange(-2, 2+1)
gamma_range = 2. ** np.arange(-5, -1+1)
param_grid = {'C': C_range, 'gamma': gamma_range, 'kernel': ['rbf']}

svm_grid = GridSearchCV(estimator = SVM_Model, param_grid = param_grid, cv=loso, n_jobs=-1)
svm_grid.fit(X_scaled, np.ravel(y), groups=gp)
print("The best parameters for SVM are: ", svm_grid.best_params_, "with a score", svm_grid.best_score_)
# Επιβεβαίωση ότι οι βέλτιστες παράμετροι C - gamma είναι:
# The best parameters for SVM are:  {'C': 2.0, 'gamma': 0.125, 'kernel': 'rbf'} with a score 0.8116841257050765

# Καλούμε τη συνάρτηση για να πάρουμε γράφημα της ακρίβειας σε σχέση με C - gamma
plot_grid_search(svm_grid.cv_results_, C_range, gamma_range, 'C', 'gamma')
'''

# Καλούμε τη μέθοδο ταξινόμησης SVM με πυρήνα Radial Basis Function (RBF) με παραμέτρους C=2 - gamma=0.125
clf_svm = SVC(kernel='rbf', C=2., gamma=0.125)

cf=[]
train_score=[]
test_score=[]
reports=[]

for i in range(10):
    trainX, trainy = load_traindata(X_scaled, y, i)
    testX, testy = load_testdata(X_scaled, y, i)
    clf_svm.fit(trainX,np.ravel(trainy))
    cf.append(confusion_matrix(testy, clf_svm.predict(testX)))
    # Train and test accuracy
    y_train_pred = clf_svm.predict(trainX)
    y_test_pred = clf_svm.predict(testX)
    train_score.append(accuracy_score(y_train_pred, trainy))
    test_score.append(accuracy_score(y_test_pred, testy))
    clsf_report = pd.DataFrame(classification_report( testy, y_test_pred, zero_division=0, output_dict=True))
    reports.append(clsf_report)

sum_cf = np.array(cf).sum(axis=0)
reports_sum = reports[0].copy()
for i in range(1, len(reports)):
    reports_sum += reports[i]
reports_sum=reports_sum/len(reports)
reports_sum.rename(columns={'0': 'walking', '1': 'standing', '2': 'jogging', '3': 'sitting', '4': 'biking', '5': 'upstairs', '6': 'downstairs'}, inplace=True)

sns.heatmap(reports_sum.iloc[:-1, :].T, annot=True)

# plot train and test accuracy 
X = range(10)
Y1 = train_score
Y2 = test_score 
plt.grid()
plt.ylabel("accuracy")
plt.plot(X, Y1, label = "train set", color = 'blue', marker = 'o', ms = 5, mec = 'black', mfc = 'black')
plt.plot(X, Y2, label = "test set", color = 'green', marker = 'o', ms = 5, mec = 'black', mfc = 'black')
plt.legend()
plt.title("Accuracy on train and test set")
plt.show()

cm=sum_cf
target_names=['walking', 'standing', 'jogging', 'sitting' , 'biking', 'upstairs', 'downstairs']

plot_confusion_matrix(cm, target_names, normalize=False)
plot_confusion_matrix(cm, target_names)

####################################################################################################################
# 3.1.1) Ομαδοποίηση δραστηριοτήτων standing-sitting

# Ομαδοποιούμε τις δραστηριότητες 1 (standing) και 3 (sitting) σε μία δραστηριότητα 1 (standing-sitting)
# Ομαδοποιούμε τις δραστηριότητες 1 (standing) και 3 (sitting) σε μία δραστηριότητα 1 (standing-sitting)

'''
# Κάνουμε αναζήτηση των βέλτιστων παραμέτρων C - gamma με ομαδοποιημένη δραστηριότητα (standing-sitting) 
# μέσω της συνάρτησης GridSearchCV
target_names=['walking', 'standing-sitting', 'jogging', 'biking', 'upstairs-downstairs']
y['Activity'] = y['Activity'].map({0: 0,1: 1,2: 2,3: 1,4: 4,5: 5,6: 5})

C_range = 2. ** np.arange(-2, 3+1)
gamma_range = 2. ** np.arange(-5, -1+1)
param_grid = {'C': C_range, 'gamma': gamma_range, 'kernel': ['rbf']}
SVM_Model = SVC()

svm_grid = GridSearchCV(estimator = SVM_Model, param_grid = param_grid, cv=loso, n_jobs=-1)
svm_grid.fit(X_scaled, np.ravel(y), groups=gp)
print("The best parameters for SVM are: ", svm_grid.best_params_, "with a score", svm_grid.best_score_)
# Επιβεβαίωση ότι οι βέλτιστες παράμετροι C - gamma είναι:
# The best parameters for SVM are:  The best parameters for SVM are:  {'C': 1.0, 'gamma': 0.125, 'kernel': 'rbf'} with a score 0.9734085414987913

plot_grid_search(svm_grid.cv_results_, C_range, gamma_range, 'C', 'γ')
'''

target_names=['walking', 'standing-sitting', 'jogging', 'biking', 'upstairs-downstairs']
y['Activity'] = y['Activity'].map({0: 0,1: 1,2: 2,3: 1,4: 4,5: 5,6: 5})

# Καλούμε τη μέθοδο ταξινόμησης SVM με πυρήνα Radial Basis Function (RBF) με παραμέτρους C=1 - gamma=0.125 για τις ομαδοποιημένες δραστηριότητες
clf_svm = SVC(kernel='rbf', C=1., gamma=0.125)

cf=[]
train_score=[]
test_score=[]
reports=[]

for i in range(10):
    trainX, trainy = load_traindata(X_scaled, y, i)
    testX, testy = load_testdata(X_scaled, y, i)
    clf_svm.fit(trainX,np.ravel(trainy))
    cf.append(confusion_matrix(testy, clf_svm.predict(testX)))
    # Train and test accuracy
    y_train_pred = clf_svm.predict(trainX)
    y_test_pred = clf_svm.predict(testX)
    train_score.append(accuracy_score(y_train_pred, trainy))
    test_score.append(accuracy_score(y_test_pred, testy))
    clsf_report = pd.DataFrame(classification_report( testy, y_test_pred, zero_division=0, output_dict=True))
    reports.append(clsf_report)

sum_cf = np.array(cf).sum(axis=0)
reports_sum = reports[0].copy()
for i in range(1, len(reports)):
    reports_sum += reports[i]
reports_sum=reports_sum/len(reports)
reports_sum.rename(columns={'0': 'walking', '1': 'standing-sitting', '2': 'jogging', '4': 'biking', '5': 'upstairs-downstairs'}, inplace=True)

sns.heatmap(reports_sum.iloc[:-1, :].T, annot=True)

cm=sum_cf

plot_confusion_matrix(cm, target_names, normalize=False)
plot_confusion_matrix(cm, target_names)


''' block κώδικα νούμερο 3 - ΤΕΛΟΣ '''



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


''' block κώδικα νούμερο 4 - ΑΡΧΗ '''

""" οδηγίες: 
     1. πρώτα τρέχει το τμήμα <<Grid Search>> μόνο του, και μετά το μαρκάρουμε ξανά ως σχόλιο (πρόταση: το τμήμα <<Grid Search>>
        παραθέτεται ενδεικτικά διότι αργεί πολύ να τρέξει - οι παράμετροι που βρέθηκαν ως βέλτιστες σημειώνονται δίπλα). 
     2. οι ενότητες 3.2.1, 3.2.2 και 3.3.3 τρέχουν μαζί (διαδικασία εύρεσης βέλτιστων παραμέτρων)
     3. η ενότητα 3.4.4 τρέχει μόνη της στο τέλος (ομαδοποίηση δραστηριοτήτων που συγχέονται)
     """


####################################################################################################################
# 3.2) Δέντρο Απόφασης (Decision Tree)
####################################################################################################################

np.random.seed(1234) # για να λαμβάνουμε τα "ίδια τυχαία" αποτελέσματα

# 3.2.1 βέλτιστες παράμετροι ##################################################################

# πραγματοποίηση Grid Search σε συνδυασμό με χειροκίνητο έλεγχο για την εύρεση βέλτιστων τιμών παραμέτρων

#-------------------Grid Search------------------------------------------------------------

'''
# Leave One Group Out διαχωρισμός
gp=[]
for i in range(10):
    for l in range(1241):
        gp.append(i)
gp = np.array(gp)
loso = LeaveOneGroupOut()
loso.get_n_splits(X_scaled.to_numpy(), y.to_numpy, gp)

# παράμετροι προς εξέταση
param_grid = {'max_depth': list(np.arrange(3,11,2)),                           # βέλτιστη τιμή: 6
              "criterion": ['entropy', 'gini'],                              # βέλτιστη τιμή: 'entropy'
              'splitter': ['best', 'random'],                                # βέλτιστη τιμή: 'best'
              'min_samples_leaf': list(np.arrange(100, 400, 50)),              # βέλτιστη τιμή: 346
              'min_samples_split': list(np.arrange(2, 150,50)),              # βέλτιστη τιμή: 2
              'max_leaf_nodes': list(np.arrange(10, 18)),                     # βέλτιστη τιμή: 15
              'ccp_alpha': list(np.arrange(0, 0.02, 0.001)) }                # βέλτιστη τιμή: 0.014

# grid search
TREE_Model = tree.DecisionTreeClassifier()
rf_Grid = GridSearchCV(estimator = TREE_Model, param_grid = param_grid, cv=loso)
rf_Grid.fit(X_scaled, np.ravel(y), groups=gp)

# εκτύπωση αποτελεσμάτων
print(rf_Grid.best_params_)
print("\n The best score across ALL searched params:\n",rf_Grid.best_score_)

'''

# ####################################################################################################################
# #-------------------------------- τελικό μοντέλο με τις βέλτιστες τιμές παραμέτρων ---------------------------------
# ####################################################################################################################

Acc = []
ClassificationReport = []
reports = []
not_ovf=0; overfitting=0
ConfMatrix=[]
Train = []; Test = []

# βέλτιστες τιμές
depth=6 
min_leaf=346        
min_split=2 
nodes=15           
alpha=0.014       
             
# μοντέλο
clf_tree = tree.DecisionTreeClassifier(criterion="entropy",              
                                        splitter="best",               
                                        max_depth=depth,                    
                                        min_samples_leaf=min_leaf,
                                        min_samples_split=min_split,
                                        max_leaf_nodes=nodes,
                                        ccp_alpha=alpha) 

# LOSO εκπαίδευση
for i in range(10):
    
    # διαχωρισμός συνόλου δεδομένων, fit στο μοντέλο, πραγματοποίηση προβλέψεων, πίνακας Classifiation Report, πίνακας σύγχησης
    trainX, trainy = load_traindata(X_scaled, y, i)
    testX, testy = load_testdata(X_scaled, y, i)
    clf_tree.fit(trainX,trainy)
    pred_tree = clf_tree.predict(testX)
    Predictions = classification_report(testy,pred_tree,zero_division=0)
    clsf_report = pd.DataFrame(classification_report( testy, pred_tree, zero_division=0, output_dict=True)).transpose()
    clsf_report.iloc[7] = [None, None, clsf_report.iloc[7][2], 1241]
    reports.append(clsf_report)
    ConfMatrix.append(confusion_matrix(testy, clf_tree.predict(testX)))
    # υπολογισμός ακρίβειας train και test set
    y_train_pred = clf_tree.predict(trainX)
    y_test_pred = clf_tree.predict(testX)
    train_score = accuracy_score(y_train_pred, trainy)
    test_score = accuracy_score(y_test_pred, testy)

    # εκτύπωση των τιμών train & test score (προαιρετικό)
    ''''
    print(' Train score', train_score)
    print(' Test score', test_score)
    print("\n")
    Train.append(train_score)
    Test.append(test_score)
    '''

    # έλεγχος υπερπροσαρμογής
    if accuracy_score(y_train_pred, trainy) == 1:
        print("overfitting")
        break
    elif accuracy_score(y_train_pred, trainy)-accuracy_score(y_test_pred, testy) < 0.05:
        not_ovf = not_ovf + 1
    else:
        overfitting = overfitting + 1

    # οπτικοποίηση του δέντρου απόφασης
    if i == 5:
        tree.plot_tree(clf_tree, feature_names=['Mean', 'STD', 'Skew', 'Max', 'Min', 'Range', 'Welch1','Welch2','Welch3','Welch4','Welch5', 
                                                    'Welch6','Welch7','Welch8'], 
                                 class_names=["walking", "standing", "jogging", "sitting", "biking", "upstairs", "downstairs"],
                                 filled = True,
                                 fontsize=11,
                                 label = "root",
                                 impurity=False)
        plt.show()

# υπολογισμός αθροίσματος πινάκων Classifiation Report
reports_sum = reports[0].copy()
for i in range(1, len(reports)):
    reports_sum += reports[i]
reports_sum=reports_sum/len(reports_sum)
reports_sum.rename(columns={'0': 'walking', '1': 'standing', '2': 'jogging', '3': 'sitting', '4': 'biking', '5': 'upstairs', '6': 'downstairs'}, inplace=True)

# οπτικοποίηση με heatmap του πίνακα Classifiation Report
sns.heatmap(reports_sum.iloc[0:10, 0:3], annot=True, linewidths=.5, yticklabels=["walking", "standing", "jogging", "sitting", 
                    "biking", "upstairs", "downstairs", "accuracy", "macro avg", "weighted avg "], vmin=0.5, vmax=1)

sum_ConfMatrix = np.array(ConfMatrix).sum(axis=0)

acc = reports_sum.iloc[7][2]
Acc.append(acc)

# εκτύπωση classification report και υπερπροσαρμογής (προαιρετικό)
'''
ClassificationReport = reports_sum
print(ClassificationReport)
print("\n", "note: walking=0, standing=1, jogging=2, sitting=3, biking=4, upstairs=5, downstairs=6")
print("\n", "model accuracy:", Acc)
print("\n", "overfitting (acc(train) - acc(test) < 0.05) : ", overfitting, "out of: ", not_ovf +  overfitting )
'''




# 3.2.2 γράφημα που απεικονίζει την ακρίβεια στο    ###########################################
#       σύνολο εκπαίδευσης και δοκιμής για τα δέντρα απόφασης  ################################

# ΣΗΜΕΙΩΣΗ: ΟΤΑΝ ΤΡΕΧΟΥΝ ΟΛΑ ΜΑΖΙ ΣΧΕΔΙΑΖΟΝΤΑΙ ΤΟ ΕΝΑ ΠΑΝΩ ΣΤΟ ΑΛΛΟ, ΓΙΑ ΑΥΤΟ ΕΙΝΑΙ ΜΑΡΚΑΡΙΣΜΕΝΟ ΣΑΝ ΣΧΟΛΙΟ
# το γράφημα υπάρχει και στο αρχείο word της αναφοράς (Ορθότητα Ταξινόμησης)

'''
# plot train and test accuracy 
X = range(10)
Y1 = Train
Y2 = Test 
plt.grid()
plt.ylabel("accuracy")
plt.plot(X, Y1, label = "train set", color = 'blue', marker = 'o', ms = 5, mec = 'black', mfc = 'black')
plt.plot(X, Y2, label = "test set", color = 'green', marker = 'o', ms = 5, mec = 'black', mfc = 'black')
plt.legend()
plt.title("Accuracy on train and test set")
plt.savefig('TrainAndTestAccuracy.png')
plt.show()
'''




# 3.3.3 Πίνακας Σύγχυσης ######################################################################

plot_confusion_matrix(cm=sum_ConfMatrix, target_names=['walking', 'standing', 'jogging', 'sitting' , 'biking', 'upstairs', 'downstairs'], 
                        normalize=False)
plot_confusion_matrix(cm=sum_ConfMatrix, target_names=['walking', 'standing', 'jogging', 'sitting' , 'biking', 'upstairs', 'downstairs']) # περιέχει κανονικοποιημένες τιμές


# 3.3.4 Ομαδοποίηση Δραστηριοτήτων ############################################################


# Ερώτημα 3, 4: Ποιες δραστηριότητες μπερδεύει ο αλγόριθμος; -> ομαδοποίηση
# Ο αλγόριθμος φαίνεται να δυσκολεύεται περισσότερο στην ταξινόμηση των παρακάτω:
# standing = sitting
# upstairs = downstairs 

# Το παρακάτω τμήμα κώδικα ομαδοποιεί σε μία κλάση τις δραστηριότητες που δεν ταξινομεί καλά το μοντέλο και εμφανίζει 
# πίνακες σύγχησης και ακρίβεια 


'''
target_names=['walking', 'standing/ sitting', 'jogging', 'biking', 'upstairs/ downstairs']
y['Activity'] = y['Activity'].map({0: 0,1: 1,2: 2,3: 1,4: 4,5: 5,6: 5})

# μοντέλο
clf_tree = tree.DecisionTreeClassifier(criterion="entropy",              
                                        splitter="best",               
                                        max_depth=depth,                    
                                        min_samples_leaf=min_leaf,
                                        min_samples_split=min_split,
                                        max_leaf_nodes=nodes,
                                        ccp_alpha=alpha) 

cf=[]
train_score=[]
test_score=[]
reports=[]

for i in range(10):
    trainX, trainy = load_traindata(X_scaled, y, i)
    testX, testy = load_testdata(X_scaled, y, i)
    clf_tree.fit(trainX,np.ravel(trainy))
    cf.append(confusion_matrix(testy, clf_tree.predict(testX)))
    # Train and test accuracy
    y_train_pred = clf_tree.predict(trainX)
    y_test_pred = clf_tree.predict(testX)
    train_score.append(accuracy_score(y_train_pred, trainy))
    test_score.append(accuracy_score(y_test_pred, testy))
    clsf_report = pd.DataFrame(classification_report( testy, y_test_pred, zero_division=0, output_dict=True))
    reports.append(clsf_report)


sum_cf = np.array(cf).sum(axis=0)
reports_sum = reports[0].copy()
for i in range(1, len(reports)):
    reports_sum += reports[i]
reports_sum=reports_sum/len(reports)
reports_sum.rename(columns={'0': 'walking', '1': 'standing/ sitting', '2': 'jogging', '4': 'biking', '5': 'upstairs/ downstairs'}, inplace=True)

sns.heatmap(reports_sum.iloc[:-1, :].T,linewidths=.5, annot=True, vmin=0.5, vmax=1)

cm=sum_cf

plot_confusion_matrix(cm, target_names, normalize=False)
plot_confusion_matrix(cm, target_names)
'''





''' block κώδικα νούμερο 4 - ΤΕΛΟΣ '''



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


''' block κώδικα νούμερο 5 - ΑΡΧΗ '''


####################################################################################################################
# 3.3) MLP
####################################################################################################################
from sklearn.neural_network import MLPClassifier
# Σε όλα τα MLP χρησιμοποιήθηκε stochastic gradient decent μέθοδος βελτιστοποίησης.
#3.3.1 MLP με  ΕΝΑ στρώμα, συνάρτηση ενεργοποίησης ReLU και δοκιμές για τιμές του momemntum.


cf=[]
singleacc=[]
singleind=[]
singlemom=[]
for moment in [0.1 , 0.5, 0.9, 0.99]:  #Οι 4 τιμές ορμής προς διερεύνηση

    for neurons in range(50,500,50):   #Έλεγχος για ένα στρώμα νευρώνων με τιμές από 50 έως 450
        clf=MLPClassifier(hidden_layer_sizes=neurons , activation='relu', solver='sgd', max_iter=200, momentum=moment)

        acc=[]

        for i in range(10):

            trainX, trainy = load_traindata(X_scaled, y, i)
            testX, testy = load_testdata(X_scaled, y, i)
            clf.fit(trainX,np.ravel(trainy))

            print(confusion_matrix(testy, clf.predict(testX)))
            cf.append(confusion_matrix(testy,clf.predict(testX)))
            acc.append(clf.score(testX,testy))

        sum_cf=np.array(cf).sum(axis=0)
        print(sum_cf)

        singleacc.append(np.round(np.mean(acc),2))

        singleind.append(clf.hidden_layer_sizes)
        singlemom.append(moment)
        print(singleind,singleacc)


        plot_confusion_matrix(cm=sum_cf,
                              target_names=['walking', 'standing', 'jogging', 'sitting', 'biking', 'upstairs',
                                            'downstairs'], normalize=False)   #Απόλυτες τιμές



        plot_confusion_matrix(cm=sum_cf,
                              target_names=['walking', 'standing', 'jogging', 'sitting', 'biking', 'upstairs',
                                            'downstairs'])  #Υπό μορφή ποσοστού





        acc.clear()
        sum_cf=[]
        cf.clear()

    plt.scatter(singleind, singleacc) #Διάγραμμα για τις ακρίβειες που βρήκαμε
    plt.plot(singleind, singleacc, label='Momentum={}'.format(moment))
    plt.legend()
    singleind.clear()
    singleacc.clear()


plt.title('Accuracy vs. Single layer \n  ReLU function')
plt.xlabel('Number of neurons')
plt.ylabel('Accuracy')
plt.ylim((0,1))
plt.show()

#===========================================================================================================================
#3.3.2 MLP με ΕΝΑ στρώμα, συνάρτηση ενεργοποίησης την Υπερβολική Εφαπτομένη (tanh) και δοκιμές για τιμές του momentum.
#Η μόνη ουσιαστική αλλαγή σε σχέση με το #3.3.1 είναι η συνάρτηση ενεργοποίησης, ενώ το υπόλοιπο σώμα παραμένει ίδιο.


cf=[]
singleacc=[]
singleind=[]
singlemom=[]
for moment in [0.1 , 0.5, 0.9, 0.99]:

    for neurons in range(50,500,50):
        clf=MLPClassifier(hidden_layer_sizes=neurons , activation='tanh', solver='sgd', max_iter=200, momentum=moment)

        acc=[]

        for i in range(10):

            trainX, trainy = load_traindata(X_scaled, y, i)
            testX, testy = load_testdata(X_scaled, y, i)
            clf.fit(trainX,np.ravel(trainy))
            print(confusion_matrix(testy, clf.predict(testX)))
            cf.append(confusion_matrix(testy, clf.predict(testX)))
            acc.append(clf.score(testX,testy))

        sum_cf=np.array(cf).sum(axis=0)
        print(sum_cf)

        singleacc.append(np.round(np.mean(acc),2))

        singleind.append(clf.hidden_layer_sizes)
        singlemom.append(moment)
        print(singleind,singleacc)



        plot_confusion_matrix(cm=sum_cf,
                              target_names=['walking', 'standing', 'jogging', 'sitting', 'biking', 'upstairs',
                                            'downstairs'], normalize=False)



        plot_confusion_matrix(cm=sum_cf,
                              target_names=['walking', 'standing', 'jogging', 'sitting', 'biking', 'upstairs',
                                            'downstairs'])




        acc.clear()
        sum_cf=[]
        cf.clear()

    plt.scatter(singleind, singleacc)
    plt.plot(singleind, singleacc, label='Momentum={}'.format(moment))
    plt.legend()
    singleind.clear()
    singleacc.clear()


plt.title('Accuracy vs. Single layer \n tanh function')
plt.xlabel('Number of neurons')
plt.ylabel('Accuracy')
plt.ylim((0,1))

plt.show()

#========================================================================================================================================
#3.3.3 MLP με ΕΝΑ στρώμα, συνάρτηση ενεργοποίησης τη λογιστική απεικόνιση και δοκιμές με τις τιμές του momentum.
#O σχολιασμός δε διαφέρει σε τίποτα από τα #3.3.1 και #3.3.2 με εξαίρεση τη χρήση άλλης συνάρτησης ενεργοποίησης.

cf=[]
singleacc=[]
singleind=[]
singlemom=[]
for moment in [0.1 , 0.5, 0.9, 0.99]:

    for neurons in range(50,500,50):
        clf=MLPClassifier(hidden_layer_sizes=neurons , activation='logistic', solver='sgd', max_iter=200, momentum=moment)

        acc=[]

        for i in range(10):

            trainX, trainy = load_traindata(X_scaled, y, i)
            testX, testy = load_testdata(X_scaled, y, i)
            clf.fit(trainX,np.ravel(trainy))
            print(confusion_matrix(testy, clf.predict(testX)))
            cf.append(confusion_matrix(testy, clf.predict(testX)))
            acc.append(clf.score(testX,testy))

        sum_cf=np.array(cf).sum(axis=0)
        print(sum_cf)

        singleacc.append(np.round(np.mean(acc),2))

        singleind.append(clf.hidden_layer_sizes)
        singlemom.append(moment)
        print(singleind,singleacc)


        plot_confusion_matrix(cm=sum_cf,
                              target_names=['walking', 'standing', 'jogging', 'sitting', 'biking', 'upstairs',
                                            'downstairs'], normalize=False)



        plot_confusion_matrix(cm=sum_cf,
                              target_names=['walking', 'standing', 'jogging', 'sitting', 'biking', 'upstairs',
                                            'downstairs'])


        acc.clear()
        sum_cf=[]
        cf.clear()

    plt.scatter(singleind, singleacc)
    plt.plot(singleind, singleacc, label='Momentum={}'.format(moment))
    plt.legend()
    singleind.clear()
    singleacc.clear()


plt.title('Accuracy vs. Single layer \n  Logistic function')
plt.xlabel('Number of neurons')
plt.ylabel('Accuracy')
plt.ylim((0,1))
plt.show()

#============================================================================================================================
#3.3.4 MLP με ΔΥΟ κρυφά στρώματα, συνάρτηση ενεργοποίησης ReLU και δοκιμές για διάφορες τιμές του momentum.


neur=[(50,50), (100,100), (150,150), (200,200), (500,500)]  #Τα μεγέθη των προς διερεύνηση στρωμάτων.
cf=[]
singleacc=[]
singleind=[]
singlemom=[]
for moment in [0.1 , 0.5, 0.9, 0.99]:  #Οι προς διερεύνηση τιμές της ορμής.

    for neurons in neur:
        clf=MLPClassifier(hidden_layer_sizes=neurons, activation='relu', solver='sgd', max_iter=200, momentum=moment)

        acc=[]

        for i in range(10):

            trainX, trainy = load_traindata(X_scaled, y, i)
            testX, testy = load_testdata(X_scaled, y, i)
            clf.fit(trainX,np.ravel(trainy))
            print(confusion_matrix(testy, clf.predict(testX)))
            cf.append(confusion_matrix(testy, clf.predict(testX)))
            acc.append(clf.score(testX,testy))

        sum_cf=np.array(cf).sum(axis=0)
        print(sum_cf)

        singleacc.append(np.round(np.mean(acc),2))

        singleind.append(neurons[0])
        singlemom.append(moment)
        print(singleind,singleacc)


        plot_confusion_matrix(cm=sum_cf,
                              target_names=['walking', 'standing', 'jogging', 'sitting', 'biking', 'upstairs',
                                            'downstairs'], normalize=False)



        plot_confusion_matrix(cm=sum_cf,
                              target_names=['walking', 'standing', 'jogging', 'sitting', 'biking', 'upstairs',
                                            'downstairs'])


        acc.clear()
        sum_cf=[]
        cf.clear()

    plt.scatter(singleind, singleacc)
    plt.plot(singleind, singleacc, label='Momentum={}'.format(moment))
    plt.legend()
    singleind.clear()
    singleacc.clear()


plt.title('Accuracy vs. Double layer \n ReLU function')
plt.xlabel('Number of neurons') #Στον άξονα χ φαίνεται η μία τιμή από τις δύο των στρωμάτων, καθώς και οι δύο είναι ίδιες π.χ. (50,50) και όχι (50,120).
plt.ylabel('Accuracy')
plt.ylim((0,1))

plt.show()

#==========================================================================================================================================
#3.3.5 MLP με ΔΥΟ κρυφά στρώματα, συνάρτηση ενεργοποίησης την Υπερβολική Εφαπτομένη και δοκιμές για το momentum.
#Τα σχόλια ίδια με το τμήμα #3.3.4 με μόνη αλλαγή τη συνάρτηση ενεργοποίησης.


target_names=['walking', 'standing/ sitting', 'jogging', 'biking', 'upstairs/ downstairs']
y['Activity'] = y['Activity'].map({0: 0,1: 1,2: 2,3: 1,4: 4,5: 5,6: 5})  #Ομαδοποίση των δραστηριοτήτων που συγχέει ο κώδικας


neur=[(50,50), (100,100), (150,150), (200,200), (500,500)]
cf=[]
singleacc=[]
singleind=[]
singlemom=[]
for moment in [0.1 , 0.5, 0.9, 0.99]:

    for neurons in neur:
        clf=MLPClassifier(hidden_layer_sizes=neurons, activation='tanh', solver='sgd', max_iter=200, momentum=moment)

        acc=[]

        for i in range(10):

            trainX, trainy = load_traindata(X_scaled, y, i)
            testX, testy = load_testdata(X_scaled, y, i)
            clf.fit(trainX,np.ravel(trainy))
            print(confusion_matrix(testy, clf.predict(testX)))
            cf.append(confusion_matrix(testy,clf.predict(testX)))
            acc.append(clf.score(testX,testy))

        sum_cf=np.array(cf).sum(axis=0)
        print(sum_cf)

        singleacc.append(np.round(np.mean(acc),2))

        singleind.append(neurons[0])
        singlemom.append(moment)
        print(singleind,singleacc)

        plot_confusion_matrix(cm=sum_cf,
                              target_names=['walking', 'standing/sitting', 'jogging', 'biking', 'upstairs/downstairs',
                                            ], normalize=False)



        plot_confusion_matrix(cm=sum_cf,
                              target_names=['walking', 'standing/sitting', 'jogging', 'biking', 'upstairs/downstairs',
                                            ])


        acc.clear()
        sum_cf=[]
        cf.clear()

    plt.scatter(singleind, singleacc)
    plt.plot(singleind, singleacc, label='Momentum={}'.format(moment))
    plt.legend()
    singleind.clear()
    singleacc.clear()


plt.title('Accuracy vs. Double layer \n tanh function')
plt.xlabel('Number of neurons')
plt.ylabel('Accuracy')
plt.ylim((0,1))

plt.show()


#=============================================================================================================================
#3.3.6 MLP με ΔΥΟ κρυφά στρώματα, συνάρτηση ενεργοποίησης τη Λογιστική απεικόνιση και δοκιμές για διάφορες τιμές του momentum.
#Ο σχολιασμός είναι ακριβώς ο ίδιος με τα τμήματα #3.3.4 και #3.3.5 με μόνη διαφορά τη χρήση την Λογιστικής απεικόνισης για ενεργοποίηση.


neur=[(50,50), (100,100), (150,150), (200,200), (500,500)]
cf=[]
singleacc=[]
singleind=[]
singlemom=[]
for moment in [0.1 , 0.5, 0.9, 0.99]:

    for neurons in neur:
        clf=MLPClassifier(hidden_layer_sizes=neurons, activation='logistic', solver='sgd', max_iter=200, momentum=moment)

        acc=[]

        for i in range(10):

            trainX, trainy = load_traindata(X_scaled, y, i)
            testX, testy = load_testdata(X_scaled, y, i)
            clf.fit(trainX,np.ravel(trainy))
            print(confusion_matrix(testy, clf.predict(testX)))
            cf.append(confusion_matrix(testy, clf.predict(testX)))
            acc.append(clf.score(testX,testy))

        sum_cf=np.array(cf).sum(axis=0)
        print(sum_cf)

        singleacc.append(np.round(np.mean(acc),2))

        singleind.append(neurons[0])
        singlemom.append(moment)
        print(singleind,singleacc)


        plot_confusion_matrix(cm=sum_cf,
                              target_names=['walking', 'standing', 'jogging', 'sitting', 'biking', 'upstairs',
                                            'downstairs'], normalize=False)



        plot_confusion_matrix(cm=sum_cf,
                              target_names=['walking', 'standing', 'jogging', 'sitting', 'biking', 'upstairs',
                                            'downstairs'])


        acc.clear()
        sum_cf=[]
        cf.clear()

    plt.scatter(singleind, singleacc)
    plt.plot(singleind, singleacc, label='Momentum={}'.format(moment))
    plt.legend()
    singleind.clear()
    singleacc.clear()



plt.title('Accuracy vs. Double layer \n Logistic function')
plt.xlabel('Number of neurons')
plt.ylabel('Accuracy')
plt.ylim((0,1))

plt.show()

#=======================================================================================================================
#3.3.7 ΔΡΑΣΤΗΡΙΟΤΗΕΣ ΠΟΥ ΣΥΓΧΕΕΙ Ο ΑΛΓΟΡΙΘΜΟΣ. Συγκεκριμένα οι 'sitting' και 'standing' και οι 'upstairs' και downstairs.
#Τρέχουμε MLP ΕΝΟΣ στρώματος με συνάρτηση ενεργοποίησης τη ReLU και δοκιμές για το momentum.

cf=[]
singleacc=[]
singleind=[]
singlemom=[]
for moment in [0.1 , 0.5, 0.9, 0.99]:

    for neurons in range(50,500,50):
        clf=MLPClassifier(hidden_layer_sizes=neurons , activation='relu', solver='sgd', max_iter=200, momentum=moment)

        acc=[]

        for i in range(10):

            trainX, trainy = load_traindata(X_scaled, y, i)
            testX, testy = load_testdata(X_scaled, y, i)
            clf.fit(trainX,np.ravel(trainy))

            print(confusion_matrix(testy, clf.predict(testX)))
            cf.append(confusion_matrix(testy,clf.predict(testX)))
            acc.append(clf.score(testX,testy))

        sum_cf=np.array(cf).sum(axis=0)
        print(sum_cf)

        singleacc.append(np.round(np.mean(acc),2))

        singleind.append(clf.hidden_layer_sizes)
        singlemom.append(moment)
        print(singleind,singleacc)


        plot_confusion_matrix(cm=sum_cf,
                              target_names=['walking', 'standing and sitting', 'jogging', 'biking', 'upstairs and downstairs',
                                            ], normalize=False)



        plot_confusion_matrix(cm=sum_cf,
                              target_names=['walking', 'standing and sitting', 'jogging', 'biking', 'upstairs and downstairs',
                                            ])


        acc.clear()
        sum_cf=[]
        cf.clear()

    plt.scatter(singleind, singleacc)
    plt.plot(singleind, singleacc, label='Momentum={}'.format(moment))
    plt.legend()
    singleind.clear()
    singleacc.clear()


plt.title('Accuracy vs. Single layer \n  ReLU function')
plt.xlabel('Number of neurons')
plt.ylabel('Accuracy')
plt.ylim((0,1))
plt.show()


#====================================== ΤΕΛΟΣ MLP CLASSIFIER ====================================================================




''' block κώδικα νούμερο 5 - ΤΕΛΟΣ '''






