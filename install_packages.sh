while read requirement; do conda install --yes $requirement; done < requirements.txt


conda install -c glemaitre imbalanced-learn
