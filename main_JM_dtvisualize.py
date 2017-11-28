import os

from sklearn import tree

import trainer

if __name__ == "__main__":
    trainer = trainer.Trainer()

    train_features, train_activity_labels, train_subject_labels, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

    decTree = tree.DecisionTreeClassifier()
    decTree = decTree.fit(train_features, train_subject_labels)
    tree.export_graphviz(decTree, out_file = 'tree.dot')
