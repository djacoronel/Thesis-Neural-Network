import csv
import numpy as np
import statsmodels.api as sm


def load_data(dataset_source):
    with open(dataset_source, 'rU') as infile:
        reader = csv.DictReader(infile)
        data = {}
        for row in reader:
            for header, value in row.items():
                try:
                    data[header].append(value)
                except KeyError:
                    data[header] = [value]

    return data


def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))

    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()

    return results


def print_coefficients(attribute_names, coefficients):
    attribute_names.append("Intercept")
    for i in range(len(attribute_names)):
        print(str(attribute_names[i]) + ": " + str(coefficients[i]))
    attribute_names.remove("Intercept")
    print("\n\n")


def print_values(attribute_names, coefficients):
    for i in range(len(attribute_names)):
        print(str(attribute_names[i]) + ": " + str(coefficients[i]))

    print("\n\n")


def get_coefficients(dataset_source, predictor_list, dependent):
    data = load_data(dataset_source)

    data_x = [np.array(data[x]).astype(np.float).tolist() for x in predictor_list]
    data_y = (np.array(data[dependent]).astype(np.float).tolist())

    results = reg_m(data_y, data_x)
    coefficients = results.params

    print_coefficients(predictor_list, coefficients)
    # print(results.summary())

    return coefficients


def compute(coefficients, inputs):
    values = []

    for i in range(len(inputs)):
        sum = 0

        for j in range(len(inputs)):
            sum += coefficients[j] * inputs[j]

        sum += coefficients[-1]
        sum -= coefficients[i]

        values.append(sum / -coefficients[i])

    return values


x_list = ["YEAR", "INTENSITY", "WIND", "SIGNAL", "POP", "DR", "FLR", "HP", "HMB"]
inputs = [1, 1, 1, 1, 1, 1, 1, 1, 1]
coef = get_coefficients("Q-CAS.csv", x_list, "CASUALTIES")
val = compute(coef, inputs)
print_values(x_list, val)

x_list = ["YEAR", "SIGNAL", "POP", "AI", "DR", "FLR", "HP", "HS", "HMA", "HMB"]
# get_coefficients("Q-DAH.csv", x_list, "DAMAGED HOUSES")

x_list = ["YEAR", "DURATION", "SIGNAL", "AI", "DR", "SR", "HP", "HS"]
# get_coefficients("Q-DAP.csv", x_list, "DAMAGED PROPERTIES")
