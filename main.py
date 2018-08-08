import Neuro as nModule
import numpy as np

data = [
    ([0, 0, 0], 0),
    ([0, 0, 1], 1),
    ([0, 1, 0], 0),
    ([0, 1, 1], 0),
    ([1, 0, 0], 1),
    ([1, 0, 1], 1),
    ([1, 1, 0], 0),
    ([1, 1, 1], 1),
]

times = 162

neuro = nModule.Neuro()

results = {'expected': [], 'real': []}

for i in range(times):
    for inputs, expected in data:
        results['expected'].append(expected)
        results['real'].append(neuro.train(np.array(inputs), expected))

number_of_predictions = 0
number_of_mistakes = 0
for i in range(times):
    for inputs, expected in data:
        result = 1 if (neuro.predict(inputs)[0] > 0.5) else 0
        number_of_predictions += 1
        if result != expected:
            number_of_mistakes += 1

print(number_of_predictions, '  ', number_of_mistakes)
