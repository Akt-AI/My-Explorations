from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


joblib.dump(my_model, "my_model.pkl")
my_model_loaded = joblib.load("my_model.pkl")
param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                          {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]
                          
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                                    scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)
print(grid_search.best_estimator_)


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score), params)
	
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)


final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)


final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# => evaluates to 48,209.6

print(clf.strategy)
print(clf.statistics_)
#convert these text labels to numbers.
#Scikit-Learn provides a transformer for this task called LabelEncoder :

encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
#array([1, 1, 4, ..., 1, 0, 3])

print(encoder.classes_)
#['<1H OCEAN' 'INLAND' 'ISLAND' 'NEAR BAY' 'NEAR OCEAN']


"""
OneHotEncoder encoder to convert integer categorical values
into one-hot vectors. Let’s encode the categories as one-hot vectors. Note that
fit_transform() expects a 2D array, but housing_cat_encoded is a 1D array, so we
need to reshape it
"""

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
print(housing_cat_1hot)
housing_cat_1hot.toarray()


"""
We can apply both transformations (from text categories to integer categories, then
from integer categories to one-hot vectors) in one shot using the LabelBinarizer
class:
"""

encoder = LabelBinarizer(sparse_output=True)
housing_cat_1hot = encoder.fit_transform(housing_cat)
print(housing_cat_1hot)

#Cross-validation:
scores = cross_val_score(clf, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict(['some_digit'])


skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
	clone_clf = clone(sgd_clf)
	X_train_folds = X_train[train_index]
	y_train_folds = (y_train_5[train_index])
	X_test_fold = X_train[test_index]
	y_test_fold = (y_train_5[test_index])
	clone_clf.fit(X_train_folds, y_train_folds)
	y_pred = clone_clf.predict(X_test_fold)
	n_correct = sum(y_pred == y_test_fold)
	print(n_correct / len(y_pred)) # prints 0.9502, 0.96565 and 0.96495
	
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
#array([ 0.9502 , 0.96565, 0.96495])

confusion_matrix(y_train_5, y_train_pred)
confusion_matrix(y_train_5, y_train_perfect_predictions)

"""
Equation 3-1. Precision
precision = TP/TP + FP
precision = TP/TP + FN
"""

"""
The F 1 score is
the harmonic mean of precision and recall (Equation 3-3). Whereas the regular mean
treats all values equally, the harmonic mean gives much more weight to low values.
As a result, the classifier will only get a high F1 score if both recall and precision are
high.

Equation 3-3. F 1 score
F1 =2/(1/precision)+(1/recall)
"""


precision_score(y_train_5, y_pred)
# == 4344 / (4344 + 1307)
#0.76871350203503808
recall_score(y_train_5, y_train_pred) # == 4344 / (4344 + 1077)
#0.79136690647482011

from sklearn.metrics import f1_score
f1_score(y_train_5, y_pred)


y_scores = sgd_clf.decision_function([some_digit])
y_scores
#array([ 161855.74572176])
threshold = 0
y_some_digit_pred = (y_scores > threshold)


"""
So how can you decide which threshold to use? For this you will first need to get the
scores of all instances in the training set using the cross_val_predict() function
again, but this time specifying that you want it to return decision scores instead of
predictions:
"""

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method="decision_function")

"""
Now with these scores you can compute precision and recall for all possible thresh‐
olds using the precision_recall_curve() function:
"""

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# Choice of decision threshold is decided by PR curve. Best threshold is where precision starts droping.

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
	plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
	plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
	plt.xlabel("Threshold")
	plt.legend(loc="upper left")
	plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


"""
Another way to select a good precision/recall tradeoff is to plot
precision directly against recall.
"""

"""
Another way to select a good precision/recall tradeoff is to plot
precision directly against recall

If someone says “let’s reach 99% precision,” you should ask, “at
what recall?”
"""

"""
The ROC Curve
The receiver operating characteristic (ROC) curve is another common tool used with
binary classifiers. It is very similar to the precision/recall curve, but instead of plot‐
ting precision versus recall, the ROC curve plots the true positive rate (another name
for recall) against the false positive rate. The FPR is the ratio of negative instances that
are incorrectly classified as positive. It is equal to one minus the true negative rate,
which is the ratio of negative instances that are correctly classified as negative. The
TNR is also called specificity. Hence the ROC curve plots sensitivity (recall) versus
1 – specificity.
To plot the ROC curve, you first need to compute the TPR and FPR for various thres‐
hold values, using the roc_curve() function:
"""

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
	plt.plot(fpr, tpr, linewidth=2, label=label)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.axis([0, 1, 0, 1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()



"""
Once again there is a tradeoff: the higher the recall (TPR), the more false positives
(FPR) the classifier produces. The dotted line represents the ROC curve of a purely
random classifier; a good classifier stays as far away from that line as possible (toward
the top-left corner).
One way to compare classifiers is to measure the area under the curve (AUC). A per‐
fect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will
have a ROC AUC equal to 0.5. Scikit-Learn provides a function to compute the ROC
AUC:
"""

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
#0.97061072797174941


"""
Since the ROC curve is so similar to the precision/recall (or PR)
curve, you may wonder how to decide which one to use. As a rule
of thumb, you should prefer the PR curve whenever the positive
class is rare or when you care more about the false positives than
the false negatives, and the ROC curve otherwise. For example,
looking at the previous ROC curve (and the ROC AUC score), you
may think that the classifier is really good. But this is mostly
because there are few positives (5s) compared to the negatives
(non-5s). In contrast, the PR curve makes it clear that the classifier
has room for improvement (the curve could be closer to the top-
right corner).
"""

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="bottom right")
plt.show()

roc_auc_score(y_train_5, y_scores_forest)


#########################################
"""
>>> some_digit_scores = sgd_clf.decision_function([some_digit])
>>> some_digit_scores

>>> np.argmax(some_digit_scores)
5
>>> sgd_clf.classes_
array([ 0., 1., 2., 3., 4., 5.,
>>> sgd_clf.classes[5]
5.0"""
#########################################

"""
The Bias/Variance Tradeoff
An important theoretical result of statistics and Machine Learning is the fact that a
model’s generalization error can be expressed as the sum of three very different
errors:
Bias
This part of the generalization error is due to wrong assumptions, such as assum‐
ing that the data is linear when it is actually quadratic. A high-bias model is most
likely to underfit the training data. 10
Variance
This part is due to the model’s excessive sensitivity to small variations in the
training data. A model with many degrees of freedom (such as a high-degree pol‐
ynomial model) is likely to have high variance, and thus to overfit the training
data.

Irreducible error
This part is due to the noisiness of the data itself. The only way to reduce this
part of the error is to clean up the data (e.g., fix the data sources, such as broken
sensors, or detect and remove outliers).
Increasing a model’s complexity will typically increase its variance and reduce its bias.
Conversely, reducing a model’s complexity increases its bias and reduces its variance.
This is why it is called a tradeoff.
"""

"""
Notice that when there are just two classes (K = 2), this cost function is equivalent to
the Logistic Regression’s cost function (log loss; see Equation 4-17).
"""


#Voting Classifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
                                                         voting='hard')

voting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
	
"""LogisticRegression 0.864
RandomForestClassifier 0.872
SVC 0.888
VotingClassifier 0.896"""


ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,
                                                     algorithm="SAMME.R", learning_rate=0.5)
                                                     
ada_clf.fit(X_train, y_train)























