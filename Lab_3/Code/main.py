import pandas as pd
import numpy as np
import math
import random
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt

drive.mount('/content/drive/')
data = pd.read_csv("/***/dat.csv")

# Отбираем нужное количество рандомных признаков
n = data.shape[1]-1
k = math.ceil(math.sqrt(n))
rand = []
i=0
while i<n-k:
  r = random.randint(0, n-1)
  if r not in rand:
    rand.append(r)
    i+=1

# Удаляем ненужные признаки из датасета
data=data.drop(data.columns[rand], axis=1)

# Класс узла
class Node():
  def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
    self.feature_index = feature_index
    self.threshold = threshold
    self.left = left
    self.right = right
    self.info_gain = info_gain
    self.value = value

# Основной класс дерева
class Tree():
    def __init__(self, min_split=2, max_depth=2):
        self.root = None
        self.min_split = min_split
        self.max_depth = max_depth

    # Сборка дерева
    def build_tree(self, df, curr_depth=0):
        X, Y = df[:, :-1], df[:, -1]
        num_samples, num_features = np.shape(X)

        if num_samples >= self.min_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(df, num_samples, num_features)

            if best_split["info_gain"] > 0:
                left_subtree = self.build_tree(best_split["df_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["df_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree,
                            best_split["info_gain"])

        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    # Поиск наилучшего критерия для разделения
    def get_best_split(self, df, num_samples, num_features):
        best_split = {"info_gain": -1}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_values = df[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds:
                df_left, df_right = self.split(df, feature_index, threshold)
                if len(df_left) > 0 and len(df_right) > 0:
                    y, left_y, right_y = df[:, -1], df_left[:, -1], df_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["df_left"] = df_left
                        best_split["df_right"] = df_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split

    # Разделение по найденному критерию
    def split(self, df, feature_index, threshold):
        df_left = np.array([row for row in df if row[feature_index] <= threshold])
        df_right = np.array([row for row in df if row[feature_index] > threshold])
        return df_left, df_right

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls ** 2
        return 1 - gini

    # Вычисляем значение конечного узла (класс большинства)
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        k = 0
        for y in Y:
            if y == 'e':
                k += 1

        pred = k / len(Y)
        return max(Y, key=Y.count), pred

    # Визуализация дерева решений
    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print(str(data.columns[tree.feature_index]), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    # Основной метод - обучение модели (с него запускается процесс построения дерева)
    def fit(self, X, Y):
        df = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(df)

    # Методы для тестирования модели
    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, tree):
        if tree.value != None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

# Разделение данных на тренировочную и тестовую выборку
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

# Обучение модели
classifier = Tree(min_split=2, max_depth=4)
classifier.fit(X_train, Y_train)
classifier.print_tree()

# Разделение полученного массива значений на классы и их вероятности
Y_pred = classifier.predict(X_test)
Y_res, Y_proba = [], []
for y in Y_pred:
  Y_res.append(y[0])
  Y_proba.append(y[1])

# Расчёт метрик (accuracy, precision, recall)
accuracy = sklearn.metrics.accuracy_score(Y_test, Y_res)
precision = sklearn.metrics.precision_score(Y_test, Y_res, pos_label="e")
recall = sklearn.metrics.recall_score(Y_test, Y_res, pos_label="e")
print('accuracy = ' + str(accuracy))
print('precision = ' + str(precision))
print('recall = ' + str(recall))

# Обработка данных (замена символьных классов на числовые значения)
y_t = []
for y in Y_test:
  if y=='e':
    y_t.append(1)
  else:
    y_t.append(0)

# Расчёт м построение AUC-ROC
lr_auc = roc_auc_score(y_t, Y_proba)
print('ROC AUC=%.3f' % (lr_auc))

fpr, tpr, treshold = roc_curve(y_t, Y_proba, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange',
         label='ROC кривая (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.show()

# Расчёт м построение AUC-PR
precision, recall, _ = precision_recall_curve(y_t, Y_proba)

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])