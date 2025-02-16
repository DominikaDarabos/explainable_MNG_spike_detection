import numpy as np


def normalize(data, min_val, max_val):
    """
    Normalize an array between the given range.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * (max_val - min_val) + min_val

def upsample(data, factor):
    """
    Upsamples a list of data points by linear interpolating n new points between the original ones.
    """
    if len(data) == 0:
        return np.array([])

    new_points = factor - 1
    interpolated_data = np.empty((len(data) - 1) * new_points + len(data))
    interpolated_data[::factor] = data
    for i in range(len(data) - 1):
        for n in range(1, new_points + 1):
            interpolated_value = data[i] + (data[i + 1] - data[i]) * n / factor
            interpolated_data[i * factor + n] = interpolated_value
    return interpolated_data

class XAIProject():
    def __init__(self, model, samples, binary_labels, multiple_labels, probabilities, decision_boundary):
        self.decision_boundary = decision_boundary
        self.model = model
        self.samples = samples
        self.binary_labels = binary_labels
        self.multiple_labels = multiple_labels
        self.probabilities = probabilities
        self.prediction_labels = self.get_predicted_classes()
        self.analyzer_output = None

    ##########################################################
    ############### Prediction Qualities #####################
    ##########################################################


    def get_predicted_classes(self):
        """
        Returns a one-dimensional array with the predicted classes.
        """
        class_labels = (self.probabilities[:, 1] >= self.decision_boundary).astype(int)
        return class_labels


    def get_truth_class_binary_indices_for_class(self, class_num):
        """
        Returns a one-dimensional array with the indices where the sample actually belongs to the given class.
        """
        return np.where(self.binary_labels == class_num)[0]
    
    def get_truth_class_multiple_indices_for_class(self, class_num):
        """
        Returns a one-dimensional array with the indices where the sample actually belongs to the given class.
        """
        return np.where(self.multiple_labels == class_num)[0]


    def get_pred_class_indices(self, class_num):
        """
        Returns a one-dimensional array with the indices where the sample, based on the prediction, belongs to the given class.
        """
        return np.where(self.prediction_labels == class_num)[0]


    def get_correct_prediction_indices(self):
        """
        Returns a one-dimensional array containing the indices of the samples for which the prediction was correct.
        """
        return np.where(self.prediction_labels == self.binary_labels)[0]


    def get_correct_pred_indices_for_class(self, class_num):
        """
        Returns a one-dimensional array containing the indices of the samples
        for which the prediction was correct and the sample belongs to the given class.
        """
        return np.intersect1d(self.get_correct_prediction_indices(), self.get_truth_class_binary_indices_for_class(class_num))


    def get_incorrect_prediction_indices(self):
        """
        Returns a one-dimensional array containing the indices of the samples for which the prediction was incorrect.
        """
        return np.where(self.binary_labels != self.get_predicted_classes())[0]


    def get_incorrect_prediction_indices_for_class(self, class_num):
        """
        Returns a one-dimensional array containing the indices of the samples
        for which the prediction was incorrect and the sample belongs to the given class.
        """
        incorrect_indices = self.get_incorrect_prediction_indices()
        class_indices = self.get_truth_class_binary_indices_for_class(class_num)
        return np.intersect1d(incorrect_indices, class_indices)


    def get_true_pos_prediction_indices(self):
        """
        Returns a one-dimensional array containing the indices of the samples
        for which the prediction was correctly positive and the sample belongs to the given class.
        """
        return np.intersect1d(np.where(self.binary_labels == 1)[0], np.where(self.prediction_labels == 1)[0])


    def get_ture_neg_prediction_indices(self):
        """
        Returns a one-dimensional array containing the indices of the samples
        for which the prediction was correctly negative and the sample belongs to the given class.
        """
        return np.intersect1d(np.where(self.binary_labels == 0)[0], np.where(self.prediction_labels == 0)[0])


    def get_false_positive_indices(self):
        """
        Returns a one-dimensional array containing the indices of the samples
        for which the prediction was falsely positive and the sample belongs to the given class.
        """
        return np.intersect1d(np.where(self.binary_labels == 0)[0],np.where(self.prediction_labels == 1)[0])


    def get_false_negative_indices(self):
        """
        Returns a one-dimensional array containing the indices of the samples
        for which the prediction was falsely negative and the sample belongs to the given class.
        """
        return np.intersect1d(np.where(self.binary_labels == 1)[0],np.where(self.prediction_labels == 0)[0])
    
    def get_true_positive_indices_for_multiple_class(self, class_num):
        """
        Returns a one-dimensional array containing the indices of the samples
        for which the prediction was correctly positive and the sample belongs to the given multuple label class.
        """
        return np.intersect1d(self.get_correct_prediction_indices(), self.get_truth_class_multiple_indices_for_class(class_num))

    def get_false_negative_indices_for_multiple_class(self, class_num):
        """
        Returns a one-dimensional array containing the indices of the samples
        for which the prediction was falsely negative and the sample belongs to the given multiple label class.
        """
        return np.intersect1d(np.where(self.multiple_labels == class_num)[0],\
                                  np.where(self.prediction_labels == 0)[0])