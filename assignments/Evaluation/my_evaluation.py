from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class my_evaluation:
    def __init__(self, y_pred, y_true, y_probs):
        self.y_pred = y_pred
        self.y_true = y_true
        self.y_probs = y_probs
        self.classes = sorted(list(set(y_true)))  # Extract unique classes

    def precision(self, target):
        return precision_score(self.y_true, self.y_pred, labels=[target], average=None)

    def recall(self, target):
        return recall_score(self.y_true, self.y_pred, labels=[target], average=None)

    def f1(self, target=None, average=None):
        # Handles per-class F1 and macro/micro/weighted averages
        if target:
            return f1_score(self.y_true, self.y_pred, labels=[target], average=None)
        else:
            return f1_score(self.y_true, self.y_pred, average=average)

    def auc(self, target):
        # Calculate AUC per class
        return roc_auc_score(self.y_true == target, self.y_probs[target])
