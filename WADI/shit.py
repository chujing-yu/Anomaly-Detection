from methods import *

data = [1,None, 3, None, 5, None, 7, None, 9]
filled_data = []

previous_value = None

for value in data:
    if value is None:
        filled_data.append(previous_value)
    else:
        filled_data.append(value)
        previous_value = value

print(filled_data)
