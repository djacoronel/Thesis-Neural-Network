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

    # print(results.predict([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    return results


def print_coefficients(attribute_names, coefficients):
    attribute_names.append("Intercept")
    for i in range(len(attribute_names)):
        print(str(attribute_names[i]) + ": " + str(coefficients[i]))
    attribute_names.remove("Intercept")
    print("\n\n")


def print_suggested(suggested):
    print("Set variable to the suggested value")
    print("to reduce risk to zero.")
    print("************")

    print(suggested)
    print("\n\n")


def get_coefficients(dataset_source, predictor_list, dependent):
    data = load_data(dataset_source)

    data_x = [np.array(data[x]).astype(np.float).tolist() for x in predictor_list]
    data_y = (np.array(data[dependent]).astype(np.float).tolist())

    from sklearn.preprocessing import normalize
    data_x = normalize(data_x, axis=1)

    from sklearn.preprocessing import scale
    data_x = scale(data_x, axis=1)

    results = reg_m(data_y, data_x)
    coefficients = results.params

    print_coefficients(predictor_list, coefficients)
    #print(results.summary())

    return coefficients


def compute_suggested(coefficients, inputs):
    values = []

    for i in range(len(inputs)):
        sum = 0

        for j in range(len(inputs)):
            sum += coefficients[j] * inputs[j]

        sum += coefficients[-1]
        sum -= coefficients[i]

        values.append(sum / -coefficients[i])

    return values


def eval(coefficients, inputs):
    sum = 0

    for j in range(len(inputs)):
        sum += coefficients[j] * inputs[j]

    sum += coefficients[-1]

    return sum


def get_suggested(dataset_source, x_list, y, inputs):
    coefficients = get_coefficients(dataset_source, x_list, y)
    suggested = compute_suggested(coefficients, inputs)

    adjustable = ["AI", "PR", "HP", "HMB", "HMC"]
    suggestable = {}

    for i in range(len(x_list)):
        if (suggested[i] > 0 and x_list[i] in adjustable):
            suggestable[x_list[i]] = suggested[i]

    return suggestable


dataset_source = "Quantile/Q-CAS.csv"
x_list = ["WIND", "POP", "AI", "PR", "HP", "HMB"]
y = "CASUALTIES"
inputs = [160, 2882408, 148641, 30, 86, 267643]

suggested = get_suggested(dataset_source, x_list, y, inputs)
print_suggested(suggested)

inputs = [160, 2882408, 148641, 30, 86, 267643]
coeff = get_coefficients(dataset_source,x_list,y)

print("risk: " + str(eval(coeff, inputs)))


