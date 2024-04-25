class DataSplits:
    def __init__(self,
                 train_x, train_y,
                 validation_x, validation_y,
                 test_for_train_x, test_for_train_y,
                 test_for_test_x, test_for_test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.test_for_train_x = test_for_train_x
        self.test_for_train_y = test_for_train_y
        self.test_for_test_x = test_for_test_x
        self.test_for_test_y = test_for_test_y