from os.path import dirname, join, normpath

import pandas as pd
from sklearn.metrics import accuracy_score

from dialogue_agent_with_preprocessing import DialogueAgent

if __name__ == '__main__':
    BASE_DIR = normpath(dirname(__file__))

    training_data = pd.read_csv(join(BASE_DIR, './training_data.csv'))

    dialogue_agent = DialogueAgent()
    dialogue_agent.train(training_data['text'], training_data['label'])

    test_data = pd.read_csv(join(BASE_DIR, './test_data.csv'))

    predictions = dialogue_agent.predict(test_data['text'])

    print(accuracy_score(test_data['label'], predictions))
